import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("ex2data1.txt", header=None)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print(df.head())

pos, neg = (y == 1).reshape(100, 1), (y == 0).reshape(100, 1)
plt.scatter(X[pos[:, 0], 0], X[pos[:, 0], 1], c="r", marker="+")
plt.scatter(X[neg[:, 0], 0], X[neg[:, 0], 1], marker="o", s=10)
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend(["Admitted", "Not admitted"], loc=0)
plt.show()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def costFunction(theta, X, y):
    m = len(y)
    P = sigmoid(np.dot(X, theta))
    cost = sum(-y * np.log(P)) - ((1 - y) * np.log(1 - P)) / m
    grad = (np.dot(X.transpose(), (P - y))) / m

    return cost[0], grad


def featureNormalization(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    X_norm = (X - mean) / std

    return X_norm, mean, std


m, n = X.shape[0], X.shape[1]
X, X_mean, X_std = featureNormalization(X)
X = np.append(np.ones((m, 1)), X, axis=1)
y = y.reshape(m, 1)
initial_theta = np.zeros((n + 1, 1))
cost, grad = costFunction(initial_theta, X, y)
print("Cost of initial theta is", cost)
print("Gradient at initial theta (zeros):", grad)


def gradientDescent(X, y, theta, alpha, num_iters):
    J_history = []

    for i in range(num_iters):
        cost, grad = costFunction(theta, X, y)
        theta = theta - (alpha * grad)
        J_history.append(cost)

    return theta, J_history


theta, J_history = gradientDescent(X, y, initial_theta, 1, 400)
print("Theta optimized by gradient descent:", theta)
print("The cost of the optimized theta:", J_history[-1])

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")
plt.show()

plt.scatter(X[pos[:, 0], 1], X[pos[:, 0], 2], c="r", marker="+", label="Admitted")
plt.scatter(X[neg[:, 0], 1], X[neg[:, 0], 2], c="b", marker="x", label="Not admitted")
x_value = np.array([np.min(X[:, 1]), np.max(X[:, 1])])
y_value = -(theta[0] + theta[1] * x_value) / theta[2]
plt.plot(x_value, y_value, "g")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend(loc=0)
plt.show()


def Predict(theta, X):
    p = X.dot(theta)

    return p > 0


p = Predict(theta, X)
print("Train Accuracy:", sum(p == y)[0], "%")
