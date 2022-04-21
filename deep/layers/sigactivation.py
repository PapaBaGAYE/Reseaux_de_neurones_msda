import numpy as np
from deep.tenseurs.tenseur import Tenseur
from deep.layers.activation import Activation


def sig(x: Tenseur) -> Tenseur:
    return 1.0 / (1 + np.exp(-x))


def sig_prime(x: Tenseur) -> Tenseur:
    return sig(x) * (1 - sig(x))


class SigActivation(Activation):
    def __init__(self, sig, sig_prime):
        super().__init__(sig, sig_prime)
