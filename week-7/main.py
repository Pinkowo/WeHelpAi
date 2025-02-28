import os

# import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

zscore_params = {}


def zscore(values, field):
    if field not in zscore_params:
        mean = np.mean(values)
        std = np.std(values)
        zscore_params[field] = (mean, std)
    else:
        mean, std = zscore_params[field]
    return (values - mean) / (std if std != 0 else 1e-8)


def dataSplit(dataset, val_split=0.25, shuffle=False, random_seed=0):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    return train_sampler, valid_sampler


###########################


class GenderHeightWeightDataset(Dataset):
    def __init__(self):
        xy = np.genfromtxt(
            "data/gender-height-weight.csv",
            delimiter=",",
            dtype=None,
            names=True,
            encoding=None,
        )

        genders = np.array(
            [0 if gender.strip('"') == "Male" else 1 for gender in xy["Gender"]],
            dtype=np.float64,
        )
        heights = zscore(xy["Height"], "Height")
        weights = zscore(xy["Weight"], "Weight")

        self.x = np.column_stack((genders, heights))
        self.y = weights
        self.n_samples = xy.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class RegressionNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RegressionNeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)

        return out


def task1():
    dataset = GenderHeightWeightDataset()

    batch_size = 1000
    val_split = 0.2
    shuffle_dataset = True
    random_seed = 1234

    train_sampler, test_sampler = dataSplit(
        dataset, val_split=val_split, shuffle=shuffle_dataset, random_seed=random_seed
    )

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    input_size = 2
    hidden_size = 4
    output_size = 1
    model = RegressionNeuralNet(input_size, hidden_size, output_size)

    epochs = 15
    batch_size = 4
    learning_rate = 0.1
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for i, (inputs, expects) in enumerate(train_loader):
            inputs = inputs.float()
            expects = expects.float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, expects)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        mse_sum = 0.0
        samples = 0
        for inputs, expects in test_loader:
            inputs = inputs.float()
            expects = expects.float().unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, expects)
            mse_sum += loss.item() * len(expects)
            samples += len(expects)

        mean_mse = mse_sum / samples
        print("Test MSE:", mean_mse)
        test_rmse_z = mean_mse**0.5
        print("Test RMSE:", test_rmse_z)
        mean, std = zscore_params["Weight"]
        test_rmse_pounds = test_rmse_z * std
        print("Test RMSE (pounds):", test_rmse_pounds)


###########################


def main():
    task1()


if __name__ == "__main__":
    main()
