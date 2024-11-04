import numpy as np

from src.ops import valid_cross_corr, full_convolution
from src.layer import Layer

# assume that all inputs and outputs are square
# input dim is in_size X in_size X in_depth
# output dim is size x size x depth
# biases dim is size x size x depth
# kernel dim is depth x k_size x k_size x in_depth
# no stride applied


class ConvNet(Layer):
    def __init__(self, input_dim, kernel_size: int, n_kernels: int) -> None:
        input_depth, input_size = input_dim
        self.input_size = input_size
        self.input_depth = input_depth
        self.kernel_size = kernel_size
        self.output_size = (input_size - kernel_size) + 1
        self.output_depth = n_kernels

        fan_in = input_depth * kernel_size * kernel_size
        fan_out = n_kernels * kernel_size * kernel_size

        # Xavier initialization scale
        scale = np.sqrt(2.0 / (fan_in + fan_out))
        self.kernel = np.random.normal(
            0, scale, (n_kernels, input_depth, kernel_size, kernel_size)
        )
        self.biases = np.zeros((n_kernels, self.output_size, self.output_size))

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.output_depth):
            for j in range(self.input_depth):
                a = self.kernel[i, j]
                b = input[j, :, :]
                self.output[i] = valid_cross_corr(a, b)
        return self.output

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        kernel_gradient = np.zeros(self.kernel.shape)
        input_gradient = np.zeros((self.input_depth, self.input_size, self.input_size))
        for i in range(self.output_depth):
            for j in range(self.input_depth):
                og_layer = output_gradient[i, :, :]
                in_layer = self.input[j, :, :]
                k_layer = self.kernel[i, j]

                kernel_gradient[i, j] = valid_cross_corr(og_layer, in_layer)
                input_gradient[j, :, :] = full_convolution(og_layer, k_layer)

        self.kernel -= learning_rate * kernel_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient
