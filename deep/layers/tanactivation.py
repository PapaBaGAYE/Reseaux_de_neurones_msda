import numpy as np
from deep.tenseurs.tenseur import Tenseur
from deep.layers.activation import Activation


def tanh(x: Tenseur) -> Tenseur:
    return np.tanh(x)


def tanh_prime(x: Tenseur) -> Tenseur:
    y = tanh(x)
    return 1 - y**2


class TanActivation(Activation):
    def __init__(self, tanh, tanh_prime):
        super().__init__(tanh, tanh_prime)
