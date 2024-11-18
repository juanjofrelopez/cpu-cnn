import numpy as np
import time
from progress.bar import Bar

from src.layer import Layer
from src.softmax import softmax


def train(
    network: list[Layer],
    epochs: int,
    learning_rate: float,
    input_data: np.ndarray,
    labels: np.ndarray,
    loss=softmax,
):
    for i in range(epochs):
        main_loss = 0
        start_time = time.time()

        l = input_data.shape[0]
        bar = Bar("Processing", max=l)

        for j in range(l):
            output = predict(input_data[j], network)

            # calculate loss and gradient
            gradient, total_loss = loss(output, labels[j])
            main_loss += total_loss

            input_gradient = gradient
            for n in reversed(network):
                input_gradient = n.backward(input_gradient, learning_rate)

            bar.next()

        main_loss /= l
        bar.finish()
        print(
            f"Epoch {i+1} Loss= {round(main_loss,6)} Run in: {round((time.time() - start_time),2)} seconds"
        )
    return


def predict(input, network):
    output = input
    for n in network:
        output = n.forward(output)
    return output