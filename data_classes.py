from dataclasses import dataclass
from typing import List

import numpy


@dataclass
class Sample:
    """A training sample's inputs and expected outputs."""
    inputs: numpy.ndarray
    expected_outputs: numpy.ndarray


@dataclass
class NetworkGradient:
    """The gradient of the entire network's weights and biases for a given sample or batch."""
    biases: List[numpy.ndarray]
    weights: List[numpy.ndarray]


@dataclass
class LayerGradients:
    """The gradient of a single layer's weights and biases for a given sample or batch."""
    biases: numpy.ndarray
    weights: numpy.ndarray


@dataclass
class NeuronValues:
    """The weighted inputs and activations of the entire network's neurons for a given sample."""
    inputs: List[numpy.ndarray]
    activations: List[numpy.ndarray]
