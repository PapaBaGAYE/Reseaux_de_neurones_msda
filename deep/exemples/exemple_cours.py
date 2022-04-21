import matplotlib.pyplot as plt
import sys, os

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

import numpy as np
from deep.layers.lineaire import Lineaire
from deep.lossfunction.mse import MSE
from deep.layers.sigactivation import SigActivation, sig, sig_prime
from deep.nn.neuralnet import NeuralNet
from deep.donnees.donnees import BatchIterateur
from deep.training.training import Training


from deep.lossfunction.logisticloss import LogisticLoss
from deep.Optimisation.sgd import SGD


# Donnees
inputs = np.array([[1, 5, 6], [3, 2, 1], [0, 5, 11], [3, 4, 1]])
target = np.array([[0], [1], [0], [1]])
# Reseau de neurones
nn = NeuralNet([Lineaire(3, 1), SigActivation(sig, sig_prime)])
# Creer les batches

batch = BatchIterateur(1)

Trainer = Training(lr=0.001)

errors = Trainer.train(inputs, target, batch, nn, loss=LogisticLoss(), optim=SGD())


for x, y in zip(inputs, target):
    print(x, y)
    predicted = nn.forward(x)
    print(x, predicted)

x = [[2, 5, 0], [-1, 6, -3], [4, -10, 12]]
y = [[0], [1], [0]]

for i, j in zip(x, y):
    print(i, j)
    predicted = nn.forward(i)
    print(i, predicted)

plt.plot(errors)
plt.show()
