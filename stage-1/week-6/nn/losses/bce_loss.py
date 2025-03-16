import numpy as np

eps = 1e-9


class BinaryCrossEntropy:
    def __init__(self):
        self.expects = None
        self.outputs = None

    def get_total_loss(self, expects, outputs):
        self.expects = np.array(expects)
        self.outputs = np.array(outputs)
        loss = -(
            self.expects * np.log(self.outputs + eps)
            + (1 - self.expects) * np.log(1 - self.outputs + eps)
        )
        total_loss = np.mean(loss)
        return total_loss

    def get_output_losses(self):
        if self.expects is None or self.outputs is None:
            raise ValueError("need to call get_total_loss() first")
        output_losses = -self.expects / (self.outputs + eps) + (1 - self.expects) / (
            1 - self.outputs + eps
        )
        return output_losses.tolist()
