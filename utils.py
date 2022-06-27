import numpy as n


def sigmoid(x: n.ndarray) -> n.ndarray:
    return 1.0 / (1.0 + n.exp(-x))


def sigmoid_prime(x: n.ndarray) -> n.ndarray:
    sigmoid_x = sigmoid(x)
    return sigmoid_x * (1 - sigmoid_x)
