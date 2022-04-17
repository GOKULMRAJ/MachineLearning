import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("ex2data2.txt", header=None)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print(df.head())

pos, neg = (y == 1).reshape(118, 1), (y == 0).reshape(118, 1)
plt.scatter(X[pos[:, 0], 0], X[pos[:, 0], 1], c="r", marker="+")
plt.scatter(X[neg[:, 0], 0], X[neg[:, 0], 1], marker="o", s=10)
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend(["Admitted", "Not admitted"], loc=0)
plt.show()


def mapFeature(x1, x2, degree):
    out = np.ones(len(x1)).reshape(len(x1), 1)
    for i in range(1, degree + 1):
        for k in range(i + 1):
            TEMP = (x1 ** (i - k) * x2 ** k).reshape(len(x1), 1)
            out = np.hstack((out, TEMP))
    return out


X = mapFeature(X[:, 0], X[:, 1], 4)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def costFunction(theta, X, y, Lambda):
    m = len(y)
    P = sigmoid(np.dot(X, theta))
    cost = sum(-y * np.log(P)) - ((1 - y) * np.log(1 - P)) / m + Lambda / (2 * m) * sum(theta ** 2)
    grad = (np.dot(X.transpose(), (P - y))) / m + (Lambda / m) * np.vstack([[0], theta[1:]])
    return cost[0], grad


m, n = X.shape[0], X.shape[1]
y = y.reshape(m, 1)
initial_theta = np.zeros((n, 1))
Lambda = 1
cost, grad = costFunction(initial_theta, X, y, Lambda)
print("Cost of initial theta is", cost)
print("Gradient at initial theta (zeros):", grad)


def gradientDescent(X, y, theta, alpha, num_iters, Lambda):
    m = len(y)
    J_history = []

    for i in range(num_iters):
        cost, grad = costFunction(theta, X, y, Lambda)
        theta = theta - (alpha * grad)
        J_history.append(cost)

    return theta, J_history


theta, J_history = gradientDescent(X, y, initial_theta, 1, 400, 0.2)
print("Theta optimized by gradient descent:", theta)
print("The cost of the optimized theta:", J_history[-1])

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")
plt.show()


def mapFeaturePlot(x1, x2, degree):
    out = np.ones(1)
    for i in range(1, degree + 1):
        for k in range(i + 1):
            T = (x1 ** (i - k) * x2 ** k)
            out = np.hstack((out, T))
    return out


plt.scatter(X[pos[:, 0], 1], X[pos[:, 0], 2], c="r", marker="+", label="Admitted")
plt.scatter(X[neg[:, 0], 1], X[neg[:, 0], 2], c="b", marker="x", label="Not admitted")

u_vals = np.linspace(-1, 1.5, 50)
v_vals = np.linspace(-1, 1.5, 50)
z = np.zeros((len(u_vals), len(v_vals)))
for i in range(len(u_vals)):
    for j in range(len(v_vals)):
        z[i, j] = mapFeaturePlot(u_vals[i], v_vals[j], 4) @ theta

plt.contour(u_vals, v_vals, z.T, 0)
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend(loc=0)
plt.show()


def Predict(theta, X):
    p = X.dot(theta)

    return p > 0


p = Predict(theta, X)
print("Train Accuracy:", (sum(p == y) / len(y) * 100)[0], "%")
