import math


def linear(x):
    return x


def relu(x):
    return max(0, x)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


activation_map = {
    "linear": linear,
    "relu": relu,
    "sigmoid": sigmoid,
}


def derivative_linear(x):
    return 1


def derivative_relu(x):
    return 1 if x > 0 else 0


def derivative_sigmoid(x):
    return x * (1 - x)


derivative_map = {
    "linear": derivative_linear,
    "relu": derivative_relu,
    "sigmoid": derivative_sigmoid,
}
