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
class NeuronInputsAndActivations:
    """The inputs and activations to each of the network's neurons for a given sample."""
    inputs: List[numpy.ndarray]
    activations: List[numpy.ndarray]
