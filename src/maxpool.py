from numpy import ndarray, unravel_index, max, zeros
from src.layer import Layer

# assume pool windows is always square
# assume input dimmensions are size x size x depth and is always square


class MaxPool(Layer):
    def __init__(self, pool_size=2) -> None:
        self.pool_size = pool_size
        self.positions = []

    def forward(self, input: ndarray) -> ndarray:
        self.input_size = input.shape[0]
        self.input_depth = input.shape[2]
        output_size = input.shape[0] - self.pool_size + 1
        self.output = zeros((output_size, output_size, input.shape[2]))
        for i in range(input.shape[2]):  # for all layers of depth
            for j in range(output_size):
                for k in range(output_size):
                    a = input[j : j + self.pool_size, k : k + self.pool_size, i]
                    self.output[j, k, i] = max(a)
                    max_row, max_col = unravel_index(a.argmax(), a.shape)
                    self.positions.append((max_row + j, max_col + k, i))
        return self.output

    def backward(self, output_gradient: ndarray) -> ndarray:
        depth = output_gradient.shape[2]
        size = output_gradient.shape[0]
        index = 0

        input_gradient = zeros((self.input_size, self.input_size, self.input_depth))
        for i in range(depth):
            for j in range(size):
                for k in range(size):
                    position = self.positions[index]
                    input_gradient[position] = output_gradient[j, k, i]
                    index += 1
        return input_gradient
