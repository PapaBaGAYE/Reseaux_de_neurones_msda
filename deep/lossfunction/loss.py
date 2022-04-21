"""Les fonctions perte permettent de mesurer la qualité des predictions
On s'interesse à deux  fonctions pertes: MSE(), MAE(), LogisticLoss 
"""
import numpy as np
from deep.tenseurs.tenseur import Tenseur


class Loss:
    def __init__(self):
        raise NotImplementedError

    def loss(self, predicted: Tenseur, actual: Tenseur) -> float:
        raise NotImplementedError

    def gradLoss(self, predicted: Tenseur, actual: Tenseur) -> Tenseur:
        raise NotImplementedError



