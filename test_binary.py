import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from logistic_regression import logistic_regression as lr
np.random.seed(5)

x, y = load_breast_cancer(return_X_y=True)
scaler = StandardScaler()

kf = StratifiedKFold(10, shuffle=True, random_state=0)
acc1 = []
acc2 = []
for train_index, validation_index in kf.split(x, y):
    scaler = StandardScaler()
    y_tr,y_te = y[train_index], y[validation_index]
    x_tr,x_te = x[train_index], x[validation_index]
    x_tr = scaler.fit_transform(x_tr)
    x_te = scaler.transform(x_te)
    
    log = lr.LogisticRegression(max_iter=5000, reg_lambda=1e-4, learning_rate=0.15, beta=0.8)
    log.fit(x_tr, y_tr)
    sk = LR()
    sk.fit(x_tr, y_tr)
    
    temp = accuracy_score(y_te, log.predict(x_te))
    acc1.append(temp)
    temp = accuracy_score(y_te, sk.predict(x_te))
    acc2.append(temp)
    

print(f'Custom LogReg mean acc: {np.mean(acc1)}')
print(f'Sklearn LogReg mean acc: {np.mean(acc2)}')

costs = log.costs
plt.plot(range(len(costs)), costs)
plt.title('Training loss')
plt.show()
plt.close()

# Test "online learning"
new_examples = 20
online_acc = []
for i in range(new_examples):
    log.fit_sample(x_te[i, :].reshape(1, x.shape[1]), y_te[i], iterations=2)
    online_acc.append(accuracy_score(y_te[i+1:], log.predict(x_te[i+1:, :])))
    
plt.plot(range(len(online_acc)), online_acc)
plt.title('MAE after online learning')
plt.show()