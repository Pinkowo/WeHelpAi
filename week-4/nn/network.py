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

    def unit(self, inputs, weights):
        return sum(a * b for a, b in zip(inputs, weights))

    def forward(self, *xs):
        outputs = list(xs)
        bias = 1

        for layer in self.layers:
            augmented_inputs = outputs + [bias]  # 加上 bias
            new_outputs = []

            act_func = activation_map.get(layer.activation, linear)

            if layer.activation == "softmax":
                logits = [self.unit(augmented_inputs, w) for w in layer.units]
                new_outputs = softmax(logits)
            else:
                for weights in layer.units:
                    z = self.unit(augmented_inputs, weights)
                    new_outputs.append(act_func(z))

            outputs = new_outputs

        return outputs
