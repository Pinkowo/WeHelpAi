class Layer:
    """
    表示網路中的一層 (Layer)。
    - weights: list，每個 element 是一個 unit 的權重，例如:
        [(0.5, 0.2, 0.3), (0.6, -0.6, 0.25)]
    - bias: list，每個 element 是一個 bias，例如:
        [0.3, 0.25]
    - activation: 字串，對應於使用哪種激活函式 (linear, relu, sigmoid, softmax ...)
    """

    def __init__(self, weights, bias, activation="linear"):
        self.weights = weights
        self.bias = bias
        self.activation = activation

    def __repr__(self):
        return f"Layer(weights={self.weights}, bias={self.bias}, activation={self.activation})"
