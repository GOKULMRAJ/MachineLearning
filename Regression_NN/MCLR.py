import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

mat = loadmat("Digits.mat")

X = mat["X"]
y = mat["y"]

fig, axis = plt.subplots(10, 10, figsize=(8, 8))
for i in range(10):
    for j in range(10):
        axis[i, j].imshow(X[np.random.randint(0, 5000), :].reshape(20, 20, order="F"),
                          cmap="hot")
        axis[i, j].axis("off")
plt.show()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def costFunction(theta, X, y, Lambda):
    m = len(y)
    P = sigmoid(np.dot(X, theta))
    cost = sum(-y * np.log(P)) - ((1 - y) * np.log(1 - P)) / m + Lambda / (2 * m) * sum(theta[1:] ** 2)
    grad = (np.dot(X.transpose(), (P - y))) / m + (Lambda / m) * np.vstack([[0], theta[1:]])
    return cost[0], grad


def gradientDescent(X, y, theta, alpha, num_iters, Lambda):
    J_history = []

    for i in range(num_iters):
        cost, grad = costFunction(theta, X, y, Lambda)
        theta = theta - (alpha * grad)
        J_history.append(cost)

    return theta, J_history


def MCLR(X, y, num_labels, Lambda):
    m, n = X.shape[0], X.shape[1]
    initial_theta = np.zeros((n + 1, 1))
    all_theta = []
    all_J = []

    X = np.hstack((np.ones((m, 1)), X))

    for i in range(1, num_labels + 1):
        theta, J_history = gradientDescent(X, np.where(y == i, 1, 0), initial_theta, 1, 200, Lambda)
        all_theta.extend(theta)
        all_J.extend(J_history)
    return np.array(all_theta).reshape(num_labels, n + 1), all_J


all_theta, all_J = MCLR(X, y, 10, 0.1)

plt.plot(all_J[0:200])
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")
plt.show()


def predictMC(all_theta, X):
    m = X.shape[0]
    X = np.hstack((np.ones((m, 1)), X))

    predictions = X @ all_theta.T
    return np.argmax(predictions, axis=1) + 1


p = predictMC(all_theta, X)
print("Training Set Accuracy:", sum(p[:, np.newaxis] == y)[0] / len(y) * 100, "%")
