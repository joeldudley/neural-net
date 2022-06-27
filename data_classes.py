from dataclasses import dataclass
from typing import List

import numpy as n


@dataclass
class Sample:
    """A training sample's inputs and expected outputs."""
    inputs: n.ndarray
    expected_outputs: n.ndarray


@dataclass
class NetworkGradient:
    """The gradient of the entire network's weights and biases for a given sample or batch."""
    biases: List[n.ndarray]
    weights: List[n.ndarray]
