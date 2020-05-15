
import numpy as np

class LogisticRegression():
    def __init__(self, max_iter=100, learning_rate=0.1, reg_lambda=0.5, beta=0.9):
        self.iter = max_iter
        self.lr = learning_rate
        self.reg_lambda = reg_lambda
        self.b = beta
        
    def h(z):
        return 1/(1 + np.exp(-z))
    
    def cost(self, x, y, m):
        c = 1/m * (-y * np.log(self.h(self.w * x)) - (1-y)*np.log(1-self.h(self.w * x)))
        regC = c + self.reg_lambda/(2*m) * np.dot(self.w.T, self.w**2)
        return sum(regC)