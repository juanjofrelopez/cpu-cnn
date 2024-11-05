import numpy as np
from numba import njit


@njit
def matloop(output_size, a, b, c, b_size):
    for i in range(output_size):
        for j in range(output_size):
            a_i = a[i : i + b_size, j : j + b_size]
            a_f = a_i.flatten()
            b_f = b.flatten()
            c[i, j] = np.dot(a_f, b_f)
    return c


def full_convolution(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_size = a.shape[0]
    b_size = b.shape[0]

    a_copy = np.copy(b) if a_size < b_size else np.copy(a)
    b_copy = np.copy(a) if a_size < b_size else np.copy(b)

    a_size = a_copy.shape[0]
    b_size = b_copy.shape[0]

    # rotate b by 180
    b_copy = np.rot90(b_copy, 2)
    pad_size = b_size - 1
    a_padded = np.pad(a_copy, pad_size, mode="constant", constant_values=0)

    # result size: N+2P+1-F
    output_size = a_size + 2 * pad_size + 1 - b_size
    c = np.zeros((output_size, output_size))

    c = matloop(output_size, a_padded, b_copy, c, b_size)
    return c


def valid_cross_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_size = a.shape[0]
    b_size = b.shape[0]

    a_copy = np.copy(b) if a_size < b_size else np.copy(a)
    b_copy = np.copy(a) if a_size < b_size else np.copy(b)

    a_size = a_copy.shape[0]
    b_size = b_copy.shape[0]

    output_size = a_size + 1 - b_size

    # result size: N+1-F
    c = np.zeros((output_size, output_size))
    c = matloop(output_size, a_copy, b_copy, c, b_size)
    return c


def class_to_one_hot(y, n_classes: int):
    a = np.zeros((y.shape[0], n_classes, 1))
    for i in range(y.shape[0]):
        a[i, y[i], :] = 1
    return a
