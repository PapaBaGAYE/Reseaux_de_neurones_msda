import numpy as np
from deep.tenseurs.tenseur import Tenseur
from deep.layers.layer import Layer
from typing import Dict


class Lineaire(Layer):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__(input_size,output_size)
        self.input_size = input_size
        self.output_size = output_size
        #self.params["W"] = np.random.randn(input_size, output_size) * 10
        self.params["W"] = np.zeros((input_size, output_size))
        self.params["b"] = np.zeros(output_size)
        

    def forward(self, inputs: Tenseur) -> Tenseur:
        self.inputs = inputs
        return np.dot(inputs,self.params["W"] ) + self.params["b"]

    def backward(self, grad: Tenseur) -> Tenseur:
        """

        y = x @ a.T + b
        dy/dx = a
        dy/da = x.T
        dy/db = I
        """
        self.grads["W"] = np.dot(self.inputs.T, grad)
        self.grads["b"] = np.sum(grad, axis=0)
        return np.dot(self.params["W"], grad)
