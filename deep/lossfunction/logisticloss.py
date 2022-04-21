from deep.tenseurs.tenseur import Tenseur, np
from deep.lossfunction.loss import Loss


class LogisticLoss(Loss):
    def __init__(self):
        pass

    def loss(self, predicted: Tenseur, actual: Tenseur) -> float:
        m = len(predicted)
        return (1 / m) * np.nansum(
            -np.multiply(actual, np.log(predicted))
            - np.multiply(1 - actual, np.log(1 - predicted))
        )

    def gradLoss(self, predicted: Tenseur, actual: Tenseur) -> Tenseur:
        m = len(predicted)
        return (1 / m) * (
            -np.divide(actual, predicted) + np.divide((1 - actual), (1 - predicted))
        )
