import math


def linear(x):
    return x


def relu(x):
    return max(0, x)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def softmax(xs):
    max_x = max(xs)
    exp_x = [math.exp(v - max_x) for v in xs]
    sum_exp_x = sum(exp_x)
    return [v / sum_exp_x for v in exp_x]
