from nn.layer import Layer
from nn.network import Network
from nn.losses.mse_loss import MSELoss
from nn.losses.bce_loss import BinaryCrossEntropy
from nn.activations import sigmoid
import csv
import numpy as np
import torch

zscore_params = {}


def read_dataset_from_csv(path):
    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset.append(row)
    return dataset


def random_weights(shape):
    return np.random.uniform(-1, 1, size=shape).tolist()


def get_list(dataset, field):
    return np.array([row[field] for row in dataset])


def transform_to_integer(value, target):
    return 1 if value == target else 0


def transform_field_zscore(list, field):
    pstdev = np.std(list, ddof=1)
    mean = np.mean(list)
    zscore_params[field + "_pstdev"] = pstdev
    zscore_params[field + "_mean"] = mean
    z_scores = ((list - mean) / pstdev).tolist()
    return z_scores


def revert_zscore(avg_loss, field):
    pstdev = zscore_params[field + "_pstdev"]
    return np.sqrt(avg_loss * (pstdev**2))


###########################


def process_task1_dataset(dataset):
    genders = []
    heights = []
    weights = []

    for row in dataset:
        gender = transform_to_integer(row["Gender"], "Male")
        height = float(row["Height"])
        weight = float(row["Weight"])

        genders.append(gender)
        heights.append(height)
        weights.append(weight)

    height_z_scores = transform_field_zscore(heights, "Height")
    weight_z_scores = transform_field_zscore(weights, "Weight")

    xs = np.column_stack((genders, height_z_scores))
    es = np.array(weight_z_scores).reshape(-1, 1)

    return xs, es


def task1():
    print("----- Predict Weight by Gender and Height -----")
    dataset = read_dataset_from_csv("data/gender-height-weight.csv")

    # Training Procedure
    nn = Network(
        Layer(
            weights=random_weights((2, 2)),
            bias=random_weights((2,)),
            activation="linear",
        ),
        Layer(
            weights=random_weights((1, 2)),
            bias=random_weights((1,)),
            activation="linear",
        ),
        Layer(
            weights=random_weights((1, 1)),
            bias=random_weights((1,)),
            activation="linear",
        ),
    )

    xs, es = process_task1_dataset(dataset)

    loss_fn = MSELoss()
    learning_rate = 0.01
    repeat_times = 10

    loss_sum = 0
    for x, e in zip(xs, es):
        outputs = nn.forward(*x)
        loss = loss_fn.get_total_loss(e, outputs)
        loss_sum += loss
    avg_loss = loss_sum / len(xs)
    print(
        "Before Training, Average Loss:",
        revert_zscore(avg_loss, "Weight"),
    )

    for _ in range(repeat_times):
        for x, e in zip(xs, es):
            outputs = nn.forward(*x)
            loss_fn.get_total_loss(e, outputs)
            output_losses = loss_fn.get_output_losses()
            nn.backward(output_losses)
            nn.zero_grad(learning_rate)

    # Evaluating Procedure
    loss_sum = 0
    for x, e in zip(xs, es):
        outputs = nn.forward(*x)
        loss = loss_fn.get_total_loss(e, outputs)
        loss_sum += loss
    avg_loss = loss_sum / len(xs)
    print("After Training, Average Loss:", revert_zscore(avg_loss, "Weight"))


###########################


def transform_age_to_integer(age, title):
    child_titles = {"Miss", "Master"}
    try:
        age = float(age)
    except (TypeError, ValueError):
        return 1 if title in child_titles else 0
    return 1 if age < 12 else 0


def check_has_family(sibs, parch):
    return 1 if int(sibs) + int(parch) > 0 else 0


def process_task2_dataset(dataset):
    genders = []
    children = []
    pclasses = []
    families = []
    fares = []
    survived_list = []

    for row in dataset:
        survived = int(row["Survived"])
        pclass = int(row["Pclass"])
        name = row["Name"]
        title = name.split(",")[1].split(".")[0].strip()
        is_child = transform_age_to_integer(row["Age"], title)
        sex = transform_to_integer(row["Sex"], "female")
        has_family = check_has_family(row["SibSp"], row["Parch"])
        fare = float(row["Fare"])

        genders.append(sex)
        children.append(is_child)
        pclasses.append(pclass)
        families.append(has_family)
        fares.append(fare)
        survived_list.append(survived)

    fare_z_scores = transform_field_zscore(fares, "Fare")

    xs = np.column_stack((pclasses, genders, children, families, fare_z_scores))
    es = np.array(survived_list)

    return xs, es


def task2():
    print("\n----- Predict Survival Status of Passengers on Titanic -----")
    dataset = read_dataset_from_csv("data/titanic.csv")

    # Training Procedure
    nn = Network(
        Layer(
            weights=random_weights((2, 5)),
            bias=random_weights((2,)),
            activation="relu",
        ),
        Layer(
            weights=random_weights((1, 2)),
            bias=random_weights((1,)),
            activation="sigmoid",
        ),
    )

    xs, es = dataset = process_task2_dataset(dataset)

    loss_fn = MSELoss()
    learning_rate = 0.01
    repeat_times = 20

    correct_count = 0
    threshold = 0.5
    for x, e in zip(xs, es):
        output = nn.forward(*x)
        survival_status = 0
        if output > threshold:
            survival_status = 1
        if survival_status == e:
            correct_count += 1
    correct_rate = correct_count / len(xs)
    print("Before Training, Correct Rate:", correct_rate)

    for _ in range(repeat_times):
        for x, e in zip(xs, es):
            outputs = nn.forward(*x)
            loss_fn.get_total_loss(e, outputs)
            output_losses = loss_fn.get_output_losses()
            nn.backward(output_losses)
            nn.zero_grad(learning_rate)

    # Evaluating Procedure
    correct_count = 0
    threshold = 0.5
    for x, e in zip(xs, es):
        output = nn.forward(*x)
        survival_status = 0
        if output > threshold:
            survival_status = 1
        if survival_status == e:
            correct_count += 1
    correct_rate = correct_count / len(xs)
    print("After Training, Correct Rate:", correct_rate)


###########################


def task3():
    print("\n----- Task 3-1 -----")
    x = torch.tensor([[2, 3, 1], [5, -2, 1]])
    print(x.shape, x.dtype)

    print("\n----- Task 3-2 -----")
    x = torch.rand(3, 4, 2)
    print(x.shape)
    print(x)

    print("\n----- Task 3-3 -----")
    y = x.new_ones((2, 1, 5))
    print(y.shape)
    print(y)

    print("\n----- Task 3-4 -----")
    x = torch.tensor([[1, 2, 4], [2, 1, 3]])
    y = torch.tensor([[5], [2], [1]])
    print(torch.mm(x, y))

    print("\n----- Task 3-5 -----")
    x = torch.tensor([[1, 2], [2, 3], [-1, 3]])
    y = torch.tensor([[5, 4], [2, 1], [1, -5]])
    print(torch.mul(x, y))


###########################


def main():
    task1()
    task2()
    task3()


if __name__ == "__main__":
    main()
