class Layer:
    def __init__(self, weights, bias, activation="linear"):
        self.weights = weights
        self.bias = bias
        self.activation = activation

    def __repr__(self):
        return f"Layer(weights={self.weights}, bias={self.bias}, activation={self.activation})"

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def get_activation(self):
        return self.activation
