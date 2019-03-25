import numpy as np
import pandas as pd
import scipy
from matplotlib import pylab, gridspec, pyplot as plt
plt.style.use('fivethirtyeight')

W = None
b = None


def mse(preds, y):
    return ((preds - y) ** 2).mean()


def grad_descent(X, y, lr, num_iter=100):
    global W, b
    np.random.seed(40)
    W = np.random.rand(X.shape[1])
    b = np.array(np.random.rand(1))

    losses = []

    N = X.shape[0]
    for iter_num in range(num_iter):
        preds = predict(X)
        losses.append(mse(preds, y))

        w_grad = np.zeros_like(W)
        b_grad = 0
        for sample, prediction, label in zip(X, preds, y):
            w_grad += 2 * (prediction - label) * sample
            b_grad += 2 * (prediction - label)

        W -= lr * w_grad
        b -= lr * b_grad
    return losses


def predict(X):
    global W, b
    return np.squeeze(X @ W + b.reshape(-1, 1))


np.random.seed(40)
func = lambda x, y: (0.43*x + 0.5*y + 0.67 + np.random.normal(0, 7, size=x.shape))

X = np.random.sample(size=(30)) * 10
Y = np.random.sample(size=(30)) * 150
result_train = [func(x, y) for x, y in zip(X, Y)]
data_train = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)

pd.DataFrame({'x': X, 'y': Y, 'res': result_train}).head()

losses = grad_descent(data_train, result_train, 1e-2, 5)

from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
#data_train - data_train.mean(axis=0))/ np.sqrt(data_train.std(axis=0)
data_train_normalized_1 = normalize(data_train)
r1 = scaler.fit(data_train)
data_train_normalized_2 = scaler.transform(data_train)

pass