from typing import Callable, Dict
from deep.tenseurs.tenseur import Tenseur
from deep.layers.layer import Layer

F = Callable[[Tenseur], Tenseur]


class Activation(Layer):
    def __init__(self, f: F, f_prime: F) -> None:
        self.params: Dict[str, Tenseur] = {}
        self.grads: Dict[str, Tenseur] = {}
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tenseur) -> float:
        self.inputs = inputs
        return self.f(self.inputs)

    def backward(self, grad: Tenseur) -> Tenseur:
        return self.f_prime(self.inputs) * grad
