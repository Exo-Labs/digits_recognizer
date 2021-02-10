import numpy as np
from layer_h import Layer_h
from layer_o import Layer_o
from params import Parameters

class NN():
    def __init__(self):
        self.learning_rate = 10e-4
        self.parameters = Parameters().parameters_generation(2)

    def train(self):
        print('training')


net = NN()

print(net.parameters)
