
from deep.Optimisation.optimize import Optimiseur
from deep.nn.neuralnet import NeuralNet


class SGD(Optimiseur):
    def __init__(self, lr: float=0.001)-> None:
        super().__init__(lr)
        
    def step(self,reseau: NeuralNet):
        for param,grad in reseau.grad_and_param():
            param -= self.lr*grad