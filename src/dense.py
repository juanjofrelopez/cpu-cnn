import numpy as np

from src.layer import Layer


class Dense(Layer):
    def __init__(self, input_size: int, kernel_size: int) -> None:
        scale = np.sqrt(2.0 / (input_size + kernel_size))
        self.weights = np.random.normal(0, scale, (kernel_size, input_size))
        self.biases = np.zeros((kernel_size, 1))
        super().__init__()

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        return np.dot(self.weights, self.input) + self.biases

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        weights_gradient = np.dot(output_gradient, self.input.T)  # scalar
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient
