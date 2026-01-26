import numpy as np

class Logistic:
    def __init__(self):
        self.Z=None

    def forward(self, X):
        Z = 1/(1+np.exp(-X))
        self.Z=Z
        return Z

    def backward(self, gradZ):
        gradX = gradZ*self.Z*(1-self.Z)
        return gradX

