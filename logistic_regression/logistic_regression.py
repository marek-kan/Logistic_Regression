
import numpy as np

class LogisticRegression():
    def __init__(self, max_iter=100, learning_rate=0.1, reg_lambda=0.5, beta=0.9,
                 return_proba=False):
        self.iter = max_iter
        self.lr = learning_rate
        self.reg_lambda = reg_lambda
        self.b = beta
        self.return_proba = return_proba
        
    def h(self, x):
        z = np.dot(x, self.w)
        return 1/(1 + np.exp(-z))
    
    def cost(self, x, y, m):
        c = 1/m * (-y * np.log(self.h(x)) - (1-y)*np.log(1-self.h(x)))
        regC = c + self.reg_lambda/(2*m) * np.dot(self.w.T, self.w**2)
        return sum(regC)
    
    def update(self, x, y, m):
        pred = self.h(x)
        grad0 = 1/m * np.dot(x[:, 0].T, (pred - y))
        grad1 = 1/m * np.dot(x[:, 1:].T, (pred-y))
        
        self.v[0] = self.b*self.v[0] + (1-self.b)*grad0
        self.v[1:] = self.b*self.v[1:] + (1-self.b)*grad1
        self.w[0] = self.w[0] - self.lr*self.v[0]
        self.w[1:] = self.w[1:] * (1 - self.lr*self.reg_lambda/m) - self.lr*self.v[1:]
    
    def fit(self, x, y):
        m = len(y)
        x = np.array(x)
        x = np.c_[np.ones(shape=(m, 1)), x]
        y = np.array(y).reshape(m, 1)
        
        self.w = np.random.normal(size=(x.shape[1], 1))
        self.v = np.zeros(shape=(x.shape[1], 1))
        self.costs = []
        for i in range(self.iter):
            self.update(x, y, m)
            self.costs.append(self.cost(x, y, m)[0])
        return print(f'Final cost: {self.costs[-1]}')
    
    def fit_sample(self, x, y, iterations=1):
        try:
            m = len(y)
        except:
            m = 1
        x = np.array(x)
        x = np.c_[np.ones(shape=(m, 1)), x]
        y = np.array(y).reshape(m, 1)
        for i in range(iterations):
            self.update(x, y, m)
            self.costs.append(self.cost(x, y, m)[0])
        return print('Last cost was: {self.costs[-2]}; New cost: {self.costs[-1]}')
        
    def predict(self, x):
        x = np.array(x)
        m = x.shape[0]
        x = np.c_[np.ones(shape=(m, 1)), x]
        if not self.return_proba:
            return np.round(self.h(x))
        else:
            return self.h(x)
        
        