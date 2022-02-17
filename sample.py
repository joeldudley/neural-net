import numpy

# TODO - Describe class.


class Sample:

    def __init__(self, inputs: numpy.ndarray, outputs: numpy.ndarray):
        self.inputs = inputs
        # The annotated outputs for the given inputs.
        self.outputs = outputs
