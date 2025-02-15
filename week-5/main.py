from nn.layer import Layer
from nn.network import Network
from nn.losses.mse_loss import MSELoss
from nn.losses.bce_loss import BinaryCrossEntropy
from nn.activations import sigmoid

repeat_times = 1


def main():
    print("----- Regression -----")
    nn = Network(
        Layer(weights=[[0.5, 0.2], [0.6, -0.6]], bias=[0.3, 0.25], activation="relu"),
        Layer(weights=[[0.8, -0.5]], bias=[0.6], activation="linear"),
        Layer(weights=[[0.6], [-0.3]], bias=[0.4, 0.75], activation="linear"),
    )
    expects = (0.8, 1)
    loss_fn = MSELoss()
    learning_rate = 0.01

    for i in range(repeat_times):
        if i == 0:
            pass  # for task1
        outputs = nn.forward(1.5, 0.5)
        loss = loss_fn.get_total_loss(expects, outputs)
        output_losses = loss_fn.get_output_losses()
        nn.backward(output_losses)
        nn.zero_grad(learning_rate)

    ###################################################

    print("----- Binary Classification -----")
    nn = Network(
        Layer(weights=[[0.5, 0.2], [0.6, -0.6]], bias=[0.3, 0.25], activation="relu"),
        Layer(weights=[[0.8, 0.4]], bias=[-0.5], activation="sigmoid"),
    )
    expects = (1,)
    loss_fn = BinaryCrossEntropy()
    learning_rate = 0.1

    for i in range(repeat_times):
        if i == 0:
            pass  # for task1
        outputs = nn.forward(0.75, 1.25)
        loss = loss_fn.get_total_loss(expects, outputs)
        output_losses = loss_fn.get_output_losses(loss)
        nn.backward(output_losses)
        nn.zero_grad(learning_rate)


if __name__ == "__main__":
    main()
