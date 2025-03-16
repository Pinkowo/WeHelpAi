import math


def mse(expects, outputs):
    mse_loss = 0.0
    for e, o in zip(expects, outputs):
        mse_loss += (e - o) ** 2
    mse_loss /= len(outputs)
    return mse_loss


def binary_cross_entropy(expects, outputs):
    bce_loss = 0.0
    for e, o in zip(expects, outputs):
        bce_loss += e * math.log(o) + (1 - e) * math.log(1 - o)
    bce_loss *= -1
    return bce_loss


def categorical_cross_entropy(expects, outputs):
    cce_loss = 0.0
    for e, o in zip(expects, outputs):
        cce_loss += e * math.log(o)
    cce_loss *= -1
    return cce_loss
