import matplotlib.pyplot as plt
from deep.donnees.donnees import BatchIterateur
from deep.nn.neuralnet import NeuralNet
from deep.Optimisation.optimize import Optimiseur
from deep.Optimisation.sgd import SGD
from deep.lossfunction.loss import Loss
from deep.lossfunction.mse import MSE
from deep.tenseurs.tenseur import Tenseur
from typing import List


class Training:
    def __init__(self,lr: float=0.01,epochs: int= 500)->None:
        self.lr: float = lr
        self.epochs: int = epochs
        
        
    def train(self, inputs, target, batch_data, nn, loss, optim):
        errors = []
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch in batch_data(inputs,target):
                predicted = nn.forward(batch.inputs)
                epoch_loss += loss.loss(predicted,batch.target)
                grad = loss.gradLoss(predicted,batch.target)
                #print(grad)
                grad = nn.backward(grad)
                #print(grad)
                optim.step(nn)
            errors.append(epoch_loss)
            
            print(f"Erreur à l'epoch {epoch} est {epoch_loss}")
        return errors

            
    
    