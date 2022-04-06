import numpy
from typing import List


class Sample:
    """A training sample's inputs and expected outputs."""

    def __init__(self, inputs: numpy.ndarray, expected_outputs: numpy.ndarray):
        self.inputs = inputs
        # The annotated outputs for the given inputs.
        self.expected_outputs = expected_outputs


class NetworkGradient:
    """The gradient of the entire network's weights and biases for a given sample or batch."""

    def __init__(self, biases: List[numpy.ndarray], weights: List[numpy.ndarray]):
        self.biases = biases
        self.weights = weights


class LayerGradients:
    """The gradient of a single layer's weights and biases for a given sample or batch."""

    def __init__(self, biases: numpy.ndarray, weights: numpy.ndarray):
        self.biases = biases
        self.weights = weights


class NeuronValues:
    """The weighted inputs and activations of the entire network's neurons for a given sample."""

    def __init__(self, inputs: List[numpy.ndarray], activations: List[numpy.ndarray]):
        self.activations = activations
        self.inputs = inputs
