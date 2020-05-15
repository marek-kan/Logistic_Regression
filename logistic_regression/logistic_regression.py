
import numpy as np

class LogisticRegression():
    def __init__(self, max_iter=100, learning_rate=0.1, reg_lambda=0.5, beta=0.9):
        self.iter = max_iter
        self.lr = learning_rate
        self.reg_lambda = reg_lambda
        self.b = beta
        
    def h(x, w):
        z = np.dot(x, w)
        return 1/(1 + np.exp(-z))
    
    def cost(self, x, y, m):
        c = 1/m * (-y * np.log(self.h(x, self.w)) - (1-y)*np.log(1-self.h(x, self.w)))
        regC = c + self.reg_lambda/(2*m) * np.dot(self.w.T, self.w**2)
        return sum(regC)
    
    def update(self, x, y, m):
        pred = self.h(x, self.w)
        grad0 = 1/m * np.dot(self.w[:, 0].T, sum(pred-y))
        grad1 = 1/m * np.dot(self.w[:, 1:].T, sum(pred-y))
        
        self.v[0] = self.b*self.v[0] + (1-self.b)*grad0
        self.v[1:] = self.b*self.v[1:] + (1-self.b)*grad1
        self.w[0] = self.w[0] - self.lr*self.v[0]
        self.w[1:] = self.w[1:] * (1 - self.lr*self.reg_lambda/m) - self.lr*self.v[1:]
        
        