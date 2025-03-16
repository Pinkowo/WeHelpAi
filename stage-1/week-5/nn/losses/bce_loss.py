import math

eps = 1e-9


class BinaryCrossEntropy:
    def __init__(self):
        self.expects = []
        self.outputs = []

    def get_total_loss(self, expects, outputs):
        loss = 0
        self.expects = expects
        self.outputs = outputs
        for e, o in zip(self.expects, self.outputs):
            result = e * math.log(o + eps) + (1 - e) * math.log(1 - o + eps)
            loss += result
        return loss * -1

    def get_output_losses(self):
        output_losses = []
        for e, o in zip(self.expects, self.outputs):
            result = -1 * e / (o + eps) + (1 - e) / (1 - o + eps)
            output_losses.append(result)
        return output_losses
