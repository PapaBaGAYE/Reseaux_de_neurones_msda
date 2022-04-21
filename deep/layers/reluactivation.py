import numpy as np
from deep.tenseurs.tenseur import Tenseur
from deep.layers.activation import Activation


def relu(x: Tenseur) -> Tenseur:
    """
    Compute the relu of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- relu(x)
    """
    s = np.maximum(0, x)

    return s


def relu_prime(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


class ReluActivation(Activation):
    def __init__(self, relu, relu_prime):
        super().__init__(relu, relu_prime)
