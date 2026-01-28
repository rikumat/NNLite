import numpy as np

class Relu:
    def __init__(self, alpha=0.01):
        self.X = None
        self.alpha = alpha

    def forward(self, X):
        self.X = X
        return np.where(X > 0, X, self.alpha * X)
    
    def backward(self, gradZ):
        gradX = gradZ * np.where(self.X > 0, 1, self.alpha)
        return gradX

