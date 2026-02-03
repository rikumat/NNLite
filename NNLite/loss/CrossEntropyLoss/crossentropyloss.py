import numpy as np
epsilon=1e-12

class CrossEntropyLoss:
    def __init__(self):
        self.P=None
        self.Y=None

    def softmax(self, X):
        """
        calculate numerically safe softmax distribution for each row
        """
        X_max = np.max(X, axis=1, keepdims=True)
        X_exp = np.exp(X-X_max)
        return np.clip(X_exp/np.sum(X_exp, axis=1, keepdims=True), epsilon,1-epsilon)
    
    def forward(self, Z, Y):
        """
        calculate the cross-enropy loss based on logits Z.
        """
        self.P=self.softmax(Z)
        self.Y=Y

        L = np.log(self.P)
        E = -np.sum(L*Y)/Z.shape[0]

        return E
    
    def backward(self):
        """
        return gradient dE / dZ
        """
        return (self.P-self.Y)/self.Y.shape[0]
