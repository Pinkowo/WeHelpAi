class Layer:
    """
    表示網路中的一層 (Layer)。
    - units: list/tuple，每個 element 是一個 unit 的權重，例如:
        [(0.5, 0.2, 0.3), (0.6, -0.6, 0.25)]
    - activation: 字串，對應於使用哪種激活函式 (linear, relu, sigmoid, softmax ...)
    """

    def __init__(self, units, activation="linear"):
        self.units = units
        self.activation = activation

    def __repr__(self):
        return f"Layer(units={self.units}, activation={self.activation})"
