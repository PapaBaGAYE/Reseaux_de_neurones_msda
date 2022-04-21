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


from deep.layers.tanactivation import TanActivation, tanh, tanh_prime
from deep.lossfunction.logisticloss import LogisticLoss
from deep.Optimisation.sgd import SGD


inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

target = np.array([[1], [0], [0], [1]])

nn = NeuralNet([Lineaire(input_size=2, output_size=1), TanActivation(tanh, tanh_prime)])

batch = BatchIterateur(1, True)


Trainer = Training(lr=0.0001)

errors = Trainer.train(inputs, target, batch, nn, loss=MSE(), optim=SGD())


for x, y in zip(inputs, target):
    print(x, y)
    predicted = nn.forward(x)
    print(x, predicted)


plt.plot(errors)
plt.show()
