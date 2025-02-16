from nn.layer import Layer
from nn.network import Network
from nn.losses.mse_loss import MSELoss
from nn.losses.bce_loss import BinaryCrossEntropy
from nn.activations import sigmoid

repeat_times = 1000


def show_layers(nn):
    for i, layer in enumerate(nn.layers):
        print(f"Layer {i}")
        print(layer.weights)
        print(layer.bias)


def calculate(nn, inputs, expects, loss_fn, learning_rate):
    for i in range(repeat_times):
        outputs = nn.forward(*inputs)
        loss = loss_fn.get_total_loss(expects, outputs)
        if i == repeat_times - 1:
            print("Total Loss", loss)
        output_losses = loss_fn.get_output_losses()
        nn.backward(output_losses)
        nn.zero_grad(learning_rate)
        if i == 0:
            show_layers(nn)


def main():
    print("----- Regression -----")
    nn = Network(
        Layer(weights=[[0.5, 0.2], [0.6, -0.6]], bias=[0.3, 0.25], activation="relu"),
        Layer(weights=[[0.8, -0.5]], bias=[0.6], activation="linear"),
        Layer(weights=[[0.6], [-0.3]], bias=[0.4, 0.75], activation="linear"),
    )
    inputs = [1.5, 0.5]
    expects = [0.8, 1.0]
    loss_fn = MSELoss()
    learning_rate = 0.01

    calculate(nn, inputs, expects, loss_fn, learning_rate)

    ###################################################

    print("----- Binary Classification -----")
    nn = Network(
        Layer(weights=[[0.5, 0.2], [0.6, -0.6]], bias=[0.3, 0.25], activation="relu"),
        Layer(weights=[[0.8, 0.4]], bias=[-0.5], activation="sigmoid"),
    )
    inputs = [0.75, 1.25]
    expects = [1.0]
    loss_fn = BinaryCrossEntropy()
    learning_rate = 0.1

    calculate(nn, inputs, expects, loss_fn, learning_rate)


if __name__ == "__main__":
    main()
