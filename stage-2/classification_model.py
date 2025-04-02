import csv
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from gensim.models.doc2vec import Doc2Vec


def encode_labels(label_list):
    label_set = sorted(set(label_list))
    label2idx = {label: idx for idx, label in enumerate(label_set)}
    encoded = [label2idx[label] for label in label_list]
    return encoded, label2idx


class TextDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        x = torch.tensor(self.embeddings[idx], dtype=torch.float)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def load_csv_data(csv_file):
    all_tokens = []
    all_labels = []
    print("Loading CSV data ...")
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tokens = row["title"].split()
            all_tokens.append([row["category"]] + tokens)
            all_labels.append(row["board"])
    return all_tokens, all_labels


def load_doc2vec_model(model_path):
    print("Loading pre-trained Doc2Vec model ...")
    d2v_model = Doc2Vec.load(model_path)
    vector_size = d2v_model.vector_size
    return d2v_model, vector_size


def get_embeddings(all_tokens, d2v_model, epochs=50):
    print("Obtaining embeddings for each document ...")
    embeddings = []
    for tokens in all_tokens:
        vec = d2v_model.infer_vector(tokens, epochs=epochs)
        embeddings.append(vec)
    return embeddings


def split_data(embeddings, labels, train_ratio=0.8):
    data = list(zip(embeddings, labels))
    random.shuffle(data)
    split_idx = int(train_ratio * len(data))
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    return train_data, test_data


def create_dataloaders(train_data, test_data, batch_size=64):
    train_embeddings, train_labels = zip(*train_data)
    test_embeddings, test_labels = zip(*test_data)
    train_dataset = TextDataset(train_embeddings, train_labels)
    test_dataset = TextDataset(test_embeddings, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader


def build_classifier(vector_size, hidden_dim, num_classes):
    model = Classifier(
        input_dim=vector_size, hidden_dim=hidden_dim, num_classes=num_classes
    )
    return model


def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct_top1 = 0
    correct_top2 = 0
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            batch_size = batch_x.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            _, top1 = torch.max(outputs, 1)
            correct_top1 += (top1 == batch_y).sum().item()

            _, top2 = torch.topk(outputs, k=2, dim=1)
            for i in range(batch_size):
                if batch_y[i] in top2[i]:
                    correct_top2 += 1

    avg_loss = total_loss / total_samples
    top1_acc = correct_top1 / total_samples
    top2_acc = correct_top2 / total_samples
    return avg_loss, top1_acc, top2_acc


def train_and_evaluate(
    model, train_loader, test_loader, criterion, optimizer, num_epochs=20
):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        test_loss, top1_acc, top2_acc = evaluate_model(model, test_loader, criterion)

        print(f"===== Epoch {epoch+1} =====")
        print(f"Average Loss in Training Data {train_loss:.6f}")
        print(f"Average Loss {test_loss:.6f}")
        print(f"First Match {top1_acc:.6f}")
        print(f"Second Match {top2_acc:.6f}")


def main():
    random.seed(42)
    torch.manual_seed(42)

    csv_file = "train-data/data-clean-token.csv"
    doc2vec_model_path = "doc2vec_model/doc2vec_model.bin"
    hidden_dim = 64
    num_epochs = 20
    batch_size = 64

    all_tokens, all_labels = load_csv_data(csv_file)
    print(f"Total documents: {len(all_tokens)}")

    encoded_labels, label2idx = encode_labels(all_labels)
    num_classes = len(label2idx)
    print(f"Number of classes: {num_classes}")

    d2v_model, vector_size = load_doc2vec_model(doc2vec_model_path)
    embeddings = get_embeddings(all_tokens, d2v_model, epochs=50)

    train_data, test_data = split_data(embeddings, encoded_labels, train_ratio=0.8)
    train_loader, test_loader = create_dataloaders(
        train_data, test_data, batch_size=batch_size
    )

    classifier = build_classifier(vector_size, hidden_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    train_and_evaluate(
        classifier,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        num_epochs=num_epochs,
    )


if __name__ == "__main__":
    main()
