from .activations import linear, relu, sigmoid, softmax

# 用字典對應可用的激活函式
activation_map = {
    "linear": linear,
    "relu": relu,
    "sigmoid": sigmoid,
    "softmax": softmax,
}


class Network:
    """
    多層網路，僅負責 forward( ) 前向傳播。
    """

    def __init__(self, *layers):
        self.layers = layers

    def unit(self, inputs, weights, bias):
        return sum(i * w for i, w in zip(inputs, weights)) + bias

    def forward(self, *xs):
        outputs = list(xs)
        for layer in self.layers:
            new_outputs = []
            act_func = activation_map.get(layer.activation, linear)

            if layer.activation == "softmax":
                logits = [
                    self.unit(outputs, w, b) for w, b in zip(layer.weights, layer.bias)
                ]
                new_outputs = softmax(logits)
            else:
                for weights, bias in zip(layer.weights, layer.bias):
                    z = self.unit(outputs, weights, bias)
                    new_outputs.append(act_func(z))

            outputs = new_outputs

        return outputs

    def backward(self, output_losses):
        pass
