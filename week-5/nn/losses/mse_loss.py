class MSELoss:
    def __init__(self):
        self.loss = 0.0
        self.output_losses = []

    def cal_loss(self, expects, outputs):
        for e, o in zip(expects, outputs):
            self.loss += (e - o) ** 2
            self.output_losses.append(2 * (e - o))
        self.loss /= len(outputs)

    def get_total_loss(self, expects, outputs):
        self.cal_loss(expects, outputs)
        return self.loss

    def get_output_losses(self):
        return self.output_losses
