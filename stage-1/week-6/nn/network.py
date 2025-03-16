from nn.activations import linear, activation_map, derivative_map
import numpy as np


class Network:
    def __init__(self, *layers):
        self.layers = layers
        self.neurons = []
        self.gradients = []

    def _unit(self, inputs, weights, bias):
        return np.dot(inputs, weights) + bias

    def forward(self, *xs):
        x = np.array(xs)
        self.neurons = [x]
        for layer in self.layers:
            W = np.array(layer.weights)
            b = np.array(layer.bias)
            z = np.dot(W, self.neurons[-1]) + b
            act_func = activation_map.get(layer.activation, linear)
            a = act_func(z)
            self.neurons.append(a)
        return self.neurons[-1]

    def backward(self, output_losses):
        output_losses = np.array(output_losses)
        last_layer = self.layers[-1]
        output_acts = self.neurons[-1]
        d_act = derivative_map.get(last_layer.activation, lambda x: 1)(output_acts)
        delta = output_losses * d_act

        gradients = []
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            prev_acts = self.neurons[i]
            dW = np.outer(delta, prev_acts)
            db = delta
            gradients.insert(0, {"weights": dW, "bias": db})

            if i > 0:
                W = np.array(layer.weights)
                new_delta = np.dot(W.T, delta)
                prev_layer = self.layers[i - 1]
                d_act_prev = derivative_map.get(prev_layer.activation, lambda x: 1)(
                    self.neurons[i]
                )
                delta = new_delta * d_act_prev

        self.gradients = gradients

    def zero_grad(self, learning_rate):
        for i, layer in enumerate(self.layers):
            layer.weights = np.array(layer.weights) - learning_rate * np.array(
                self.gradients[i]["weights"]
            )
            layer.bias = np.array(layer.bias) - learning_rate * np.array(
                self.gradients[i]["bias"]
            )

        self.gradients = []

    def get_layers(self):
        return self.layers
