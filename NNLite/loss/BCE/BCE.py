import numpy as np
epsilon = 1e-12

class BCE:
    def __init__(self):
        pass
        self.Y=None
        self.Z=None

    def forward(self, Z:np.array, Y):
        Z=np.clip(Z, epsilon, 1-epsilon)
        E=-np.mean((Y*np.log(Z)+(1-Y)*np.log(1-Z)))
        self.Y=Y
        self.Z=Z
        return E

    def backward(self):
        gradZ = -1/(self.Z.shape[0]*self.Z.shape[1])*(self.Y/self.Z-(1-self.Y)/(1-self.Z))
        return gradZ



