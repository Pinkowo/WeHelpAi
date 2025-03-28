import csv
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

folder = "test-data"  # "train-data" or "test-data"


def train_model():
    csv_file = f"{folder}/data-clean-token.csv"
    documents = []

    print("Reading CSV and preparing documents...")
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            tokens = row["title"].split()
            documents.append(TaggedDocument(words=tokens, tags=[str(idx)]))
            if idx > 0 and idx % 100000 == 0:
                print(f"Processed {idx} documents...")

    print(f"Total documents: {len(documents)}")

    print("Building Doc2Vec model vocabulary...")
    model = Doc2Vec(vector_size=100, window=5, min_count=2, dm=1, workers=4, epochs=500)
    model.build_vocab(documents)

    print("Training Doc2Vec model...")
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

    print("Saving model to 'doc2vec_model.bin'...")
    model.save("doc2vec_model/doc2vec_model.bin")
    print("Model saved successfully!")


if __name__ == "__main__":
    train_model()
