import numpy as np

class SGD:
    def __init__(self, params, reg=None, lr=0.01):
        self.reg=reg
        self.params=params
        self.lr=lr
    
    def step(self):
        for p in self.params:
            if self.reg!=None:
                p["grad"]+=self.reg.backward(p["param"])
            p["param"]-=self.lr*p["grad"]
