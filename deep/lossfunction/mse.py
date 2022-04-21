import numpy as np
from deep.lossfunction.loss import Loss
from deep.tenseurs.tenseur import Tenseur



class MSE(Loss):
    def __init__(self):
        pass

    def loss(self, predicted: Tenseur, actual: Tenseur) -> float:
        return (1 / len(actual)) * np.sum((predicted - actual) ** 2)

    def gradLoss(self, predicted: Tenseur, actual: Tenseur) -> Tenseur:
        return (2 / len(actual)) * (predicted - actual)