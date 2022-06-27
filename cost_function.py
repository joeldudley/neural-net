import numpy as n


class QuadraticCostFunction:
    """The quadratic cost function, 1/2n * sum(||y(x) - a||^2))."""

    @staticmethod
    def output_layer_cost_gradient(output_layer_activations: n.ndarray, expected_outputs: n.ndarray) -> n.ndarray:
        """The rate of change in the network's cost for a change in the output layer activations (i.e. the first
        derivative of the network's cost function."""
        return output_layer_activations - expected_outputs
