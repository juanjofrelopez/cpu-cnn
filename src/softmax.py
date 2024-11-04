import numpy as np


def softmax(y_pred, y_true):
    shiftx = y_pred - np.max(y_pred)
    exps = np.exp(shiftx)
    probs = exps / sum(exps)
    epsilon = 1e-15
    safe_probs = np.clip(probs, epsilon, 1 - epsilon)
    log_likelihood = -np.log(safe_probs)
    loss = np.sum(y_true * log_likelihood) / y_pred.shape[0]
    gradient = safe_probs - y_true
    return gradient, loss
