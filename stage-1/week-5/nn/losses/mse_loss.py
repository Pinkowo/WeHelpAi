class MSELoss:
    def __init__(self):
        self.expects = []
        self.outputs = []

    def get_total_loss(self, expects, outputs):
        loss = 0
        self.expects = expects
        self.outputs = outputs
        for e, o in zip(self.expects, self.outputs):
            loss += (e - o) ** 2
        return loss / len(self.outputs)

    def get_output_losses(self):
        output_losses = []
        for e, o in zip(self.expects, self.outputs):
            output_losses.append((o - e))
        return output_losses
