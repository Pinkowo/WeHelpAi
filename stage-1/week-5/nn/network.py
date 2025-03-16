from .activations import linear, activation_map, derivative_map


class Network:
    def __init__(self, *layers):
        self.layers = layers
        self.neurons = []
        self.gradients = []

    def _unit(self, inputs, weights, bias):
        return sum(i * w for i, w in zip(inputs, weights)) + bias

    def forward(self, *xs):
        self.neurons.append(list(xs))

        for i, layer in enumerate(self.layers):
            new_outputs = []
            act_func = activation_map.get(layer.activation, linear)

            for weights, bias in zip(layer.weights, layer.bias):
                neuron = self._unit(self.neurons[i], weights, bias)
                new_outputs.append(act_func(neuron))

            self.neurons.append(new_outputs)

        return self.neurons[-1]

    def backward(self, output_losses):
        reversed_neurons = list(reversed(self.neurons))
        reversed_layers = list(reversed(self.layers))
        gradients = []

        output_layer = reversed_layers[0]
        act_func = derivative_map.get(output_layer.get_activation(), linear)
        delta = [
            loss * act_func(neuron)
            for loss, neuron in zip(output_losses, reversed_neurons[0])
        ]

        for i, layer in enumerate(reversed_layers):
            if i + 1 < len(reversed_neurons):
                input_activation = reversed_neurons[i + 1]
                weight_gradients = [[d * a for a in input_activation] for d in delta]
            else:
                weight_gradients = None

            bias_gradients = delta

            gradients.append({"weights": weight_gradients, "bias": bias_gradients})

            if i + 1 < len(reversed_layers):
                weights = layer.get_weights()
                previous_layer = reversed_layers[i + 1]
                next_act_func = derivative_map.get(
                    previous_layer.get_activation(), linear
                )
                previous_activations = reversed_neurons[i + 1]

                new_delta = []
                for j in range(len(previous_activations)):
                    weighted_sum = sum(
                        weights[k][j] * delta[k] for k in range(len(delta))
                    )
                    new_delta.append(
                        weighted_sum * next_act_func(previous_activations[j])
                    )
                delta = new_delta

        self.gradients = list(reversed(gradients))

    def zero_grad(self, learning_rate):
        for i, layer in enumerate(self.layers):
            weights = layer.get_weights()
            bias = layer.get_bias()
            for j in range(len(weights)):
                for k in range(len(weights[j])):
                    weights[j][k] -= learning_rate * self.gradients[i]["weights"][j][k]
            for j in range(len(bias)):
                bias[j] -= learning_rate * self.gradients[i]["bias"][j]
            layer.weights = weights
            layer.bias = bias
        self.gradients = []

    def get_layers(self):
        return self.layers
