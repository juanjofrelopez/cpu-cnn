import numpy as np


class Reshape:
    def __init__(self, input_dim, output_dim) -> None:
        self.input_dim, self.output_dim = input_dim, output_dim
        return

    def forward(self, input: np.ndarray):
        self.input = input
        return input.reshape(self.output_dim)

    def backward(self, output_gradient, _):
        return output_gradient.reshape(self.input_dim)
