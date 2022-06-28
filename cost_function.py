import numpy as n

from utils import sigmoid_prime


class QuadraticCostFunction:
    """The quadratic cost function, 1/2n * sum(||y(x) - a||^2))."""

    @staticmethod
    def output_layer_bias_gradient(output_layer_activations: n.ndarray, expected_outputs: n.ndarray,
                                   neuron_inputs: n.ndarray) -> n.ndarray:
        """The rate of change in the network's cost for a change in the output layer activations (i.e. the first
        derivative of the network's cost function)."""
        return (output_layer_activations - expected_outputs) * sigmoid_prime(neuron_inputs)


class CrossEntropyCostFunction:
    """The cross-entropy cost function."""

    @staticmethod
    def output_layer_bias_gradient(output_layer_activations: n.ndarray, expected_outputs: n.ndarray,
                                   _: n.ndarray) -> n.ndarray:
        """The rate of change in the network's cost for a change in the output layer activations (i.e. the first
        derivative of the network's cost function)."""
        return output_layer_activations - expected_outputs
