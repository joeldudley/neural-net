import numpy


class Sample:
    """A training sample's inputs and expected outputs."""

    def __init__(self, inputs: numpy.ndarray, expected_outputs: numpy.ndarray):
        self.inputs = inputs
        # The annotated outputs for the given inputs.
        self.expected_outputs = expected_outputs


class NetworkGradient:
    """The gradient for the network's weights and biases for a given sample or batch."""

    def __init__(self, biases: list[numpy.ndarray], weights: list[numpy.ndarray]):
        self.biases = biases
        self.weights = weights


class LayerGradient:
    """The gradient for a single layer's weights and biases for a given sample or batch."""

    def __init__(self, biases: numpy.ndarray, weights: numpy.ndarray):
        self.biases = biases
        self.weights = weights


class NeuronState:
    """The inputs and outputs of the neurons in a network for a given sample."""

    def __init__(self, inputs: list[numpy.ndarray], outputs: list[numpy.ndarray]):
        self.outputs = outputs
        self.inputs = inputs
