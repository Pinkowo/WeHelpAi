import math


class BinaryCrossEntropy:
    def __init__(self):
        self.loss = 0.0
        self.output_losses = []

    def cal_loss(self, expects, outputs):
        for e, o in zip(expects, outputs):
            result = e * math.log(o) + (1 - e) * math.log(1 - o)
            self.loss += result
            self.output_losses.append(result)
        self.loss *= -1

    def get_total_loss(self, expects, outputs):
        self.cal_loss(expects, outputs)
        return self.loss

    def get_output_losses(self):
        return self.output_losses
