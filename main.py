import time
from mnist import MNIST
import numpy as np
from src.ops import class_to_one_hot

from src.conv import ConvNet
from src.relu import ReLU
from src.reshape import Reshape
from src.dense import Dense
from src.train import train, predict

IMAGE_SIZE = 28
TRAINING_SET_SIZE = 10000
TEST_SET_SIZE = 10
N_CLASSES = 10
EPOCHS = 20
LEARNING_RATE = 0.001

FILTER_SIZE = 3
FILTER_DEPTH = 3


def process_data(x, y, size):
    x = x[:size]
    y = y[:size]
    x = x.astype("float64").reshape((size, 1, IMAGE_SIZE, IMAGE_SIZE))
    x /= 255
    x -= x.mean()
    y = class_to_one_hot(y, 10)
    return x, y


def main() -> None:
    mndata = MNIST("./data/mnist", return_type="numpy")

    print("processing data...")

    x, y = mndata.load_training()
    train_images, train_labels = process_data(x, y, TRAINING_SET_SIZE)

    x, y = mndata.load_testing()
    test_images, test_labels = process_data(x, y, TEST_SET_SIZE)

    print("data processed succesfully")

    network = [
        ConvNet((1, 28), 3, 2),
        ReLU(),
        ConvNet((2, 26), 2, 2),
        ReLU(),
        Reshape((2, 25, 25), (2 * 25 * 25, 1)),
        Dense(2 * 25 * 25, N_CLASSES),
    ]

    print("training network...")

    train(network, EPOCHS, LEARNING_RATE, train_images, train_labels)

    print("network trained")

    for i in range(TEST_SET_SIZE):
        input = test_images[i, :, :]
        predicted_output = predict(input, network)
        print(
            "pred:", np.argmax(predicted_output), "\ttrue:", np.argmax(test_labels[i])
        )


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("\n >>> Program run in %s seconds <<<" % round((time.time() - start_time), 2))
