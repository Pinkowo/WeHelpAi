from nn.layer import Layer
from nn.network import Network
from nn.losses import mse, binary_cross_entropy, categorical_cross_entropy


def main():
    print("----- Regression -----")
    nn1 = Network(
        Layer([(0.5, 0.2, 0.3), (0.6, -0.6, 0.25)], "relu"),
        Layer([(0.8, -0.5, 0.6), (0.4, 0.5, -0.25)], "linear"),
    )
    outputs = nn1.forward(1.5, 0.5)
    expects = (0.8, 1)
    print("Outputs", outputs)
    print("Total Loss", mse(expects, outputs))

    outputs = nn1.forward(0, 1)
    expects = (0.5, 0.5)
    print("Outputs", outputs)
    print("Total Loss", mse(expects, outputs))

    ###################################################

    print("----- Binary Classification -----")
    nn2 = Network(
        Layer([(0.5, 0.2, 0.3), (0.6, -0.6, 0.25)], "relu"),
        Layer([(0.8, 0.4, -0.5)], "sigmoid"),
    )
    outputs = nn2.forward(0.75, 1.25)
    expects = (1,)
    print("Outputs", outputs)
    print("Total Loss", binary_cross_entropy(expects, outputs))

    outputs = nn2.forward(-1, 0.5)
    expects = (0,)
    print("Outputs", outputs)
    print("Total Loss", binary_cross_entropy(expects, outputs))

    ###################################################

    print("----- Multi-Label Classification -----")
    nn3 = Network(
        Layer([(0.5, 0.2, 0.3), (0.6, -0.6, 0.25)], "relu"),
        Layer([(0.8, -0.4, 0.6), (0.5, 0.4, 0.5), (0.3, 0.75, -0.5)], "sigmoid"),
    )
    outputs = nn3.forward(1.5, 0.5)
    expects = (1, 0, 1)
    print("Outputs", outputs)
    print("Total Loss", binary_cross_entropy(expects, outputs))

    outputs = nn3.forward(0, 1)
    expects = (1, 1, 0)
    print("Outputs", outputs)
    print("Total Loss", binary_cross_entropy(expects, outputs))

    ###################################################

    print("----- Multi-Class Classification -----")
    nn4 = Network(
        Layer([(0.5, 0.2, 0.3), (0.6, -0.6, 0.25)], "relu"),
        Layer([(0.8, -0.4, 0.6), (0.5, 0.4, 0.5), (0.3, 0.75, -0.5)], "softmax"),
    )
    outputs = nn4.forward(1.5, 0.5)
    expects = (1, 0, 0)
    print("Outputs", outputs)
    print("Total Loss", categorical_cross_entropy(expects, outputs))

    outputs = nn4.forward(0, 1)
    expects = (0, 0, 1)
    print("Outputs", outputs)
    print("Total Loss", categorical_cross_entropy(expects, outputs))


if __name__ == "__main__":
    main()
