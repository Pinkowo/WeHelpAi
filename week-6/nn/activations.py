import numpy as np
import math


def linear(x):
    return x


def relu(x):
    if isinstance(x, np.ndarray):
        return np.maximum(0, x)
    else:
        return x if x > 0 else 0


def sigmoid(x):
    if isinstance(x, np.ndarray):
        return 1 / (1 + np.exp(-x))
    else:
        return 1 / (1 + math.exp(-x))


activation_map = {
    "linear": linear,
    "relu": relu,
    "sigmoid": sigmoid,
}


def derivative_linear(x):
    if isinstance(x, np.ndarray):
        return np.ones_like(x)
    else:
        return 1


def derivative_relu(x):
    if isinstance(x, np.ndarray):
        return np.where(x > 0, 1, 0)
    else:
        return 1 if x > 0 else 0


def derivative_sigmoid(x):
    if isinstance(x, np.ndarray):
        return x * (1 - x)
    else:
        return x * (1 - x)


derivative_map = {
    "linear": derivative_linear,
    "relu": derivative_relu,
    "sigmoid": derivative_sigmoid,
}
