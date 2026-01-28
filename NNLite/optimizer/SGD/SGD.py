class SGD:
    def __init__(self, model, lr=0.01):
        self.model = model
        self.lr=lr
    
    def step(self):
        params = self.model.params()
        for p in params:
            p["param"]-=self.lr*p["grad"]
    