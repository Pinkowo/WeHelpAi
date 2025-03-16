class Layer:
    def __init__(self, units):
        self.units = units


class Network:
    def __init__(self, *layers):
        self.layers = layers

    def unit(self, inputs, weights):
        return sum(a * b for a, b in zip(inputs, weights))

    def forward(self, *xs):
        outputs = list(xs)
        bias = 1

        for layer in self.layers:
            augmented_inputs = outputs + [bias]
            new_outputs = []
            for weights in layer.units:
                new_outputs.append(self.unit(augmented_inputs, weights))
            outputs = new_outputs

        return outputs


def main():
    nn1 = Network(
        Layer([[0.5, 0.2, 0.3], [0.6, -0.6, 0.25]]), Layer([[0.8, 0.4, -0.5]])
    )
    print(nn1.forward(1.5, 0.5))
    print(nn1.forward(0, 1))

    nn2 = Network(
        Layer([[0.5, 1.5, 0.3], [0.6, -0.8, 1.25]]),
        Layer([[0.6, -0.8, 0.3]]),
        Layer([[0.5, 0.2], [-0.4, 0.5]]),
    )
    print(nn2.forward(0.75, 1.25))
    print(nn2.forward(-1, 0.5))


if __name__ == "__main__":
    main()
