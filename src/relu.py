from src.layer import Layer
import numpy as np


class ReLU(Layer):
    def __init__(self) -> None:
        pass

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        return np.maximum(0, input)

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        return np.multiply(output_gradient, np.where(self.input > 0, 1, 0))
