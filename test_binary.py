
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from logistic_regression import logistic_regression as lr

x, y = load_breast_cancer(return_X_y=True)
scaler = StandardScaler()

x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.1, random_state=5, shuffle=True)
x_tr = scaler.fit_transform(x_tr)
x_te = scaler.transform(x_te)

log = lr.LogisticRegression(max_iter=5000, reg_lambda=0, learning_rate=0.01)
log.fit(x_tr, y_tr)

pred = log.predict(x_te)
acc1 = accuracy_score(y_te, pred)

sk = LR(tol=1e-12)
sk.fit(x_tr, y_tr)
acc2 = accuracy_score(y_te, sk.predict(x_te))

costs = log.costs
plt.plot(range(len(costs)), costs)
plt.title('Training loss')
plt.show()
plt.close()