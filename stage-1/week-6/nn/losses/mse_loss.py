import numpy as np


class MSELoss:
    def __init__(self):
        self.expects = None
        self.outputs = None

    def get_total_loss(self, expects, outputs):
        self.expects = np.array(expects)
        self.outputs = np.array(outputs)
        loss = np.mean((self.expects - self.outputs) ** 2)
        return loss

    def get_output_losses(self):
        if self.expects is None or self.outputs is None:
            raise ValueError("need to call get_total_loss() first")
        output_losses = 2 * (self.outputs - self.expects)
        return output_losses.tolist()
