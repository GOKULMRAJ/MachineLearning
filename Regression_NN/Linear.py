import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("Uni_linear.txt", header=None)


# print(data.head())
# print(data.describe())


def computeCost(X, y, theta):
    m = len(y)

    return 1 / (2 * m) * np.dot((X.dot(theta) - y).transpose(), X.dot(theta) - y)[0]


D = data.values
m = D[:, 0].size
X = np.append(np.ones((m, 1)), D[:, 0].reshape(m, 1), axis=1)
y = D[:, 1].reshape(m, 1)
theta = np.zeros((2, 1))


def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []

    for i in range(num_iters):
        predictions = X.dot(theta)
        error = np.dot(X.transpose(), (predictions - y))
        descent = alpha * 1 / m * error
        theta -= descent
        J_history.append(computeCost(X, y, theta))

    return theta, J_history


theta, J_history = gradientDescent(X, y, theta, 0.01, 1500)
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.show()

plt.scatter(data[0], data[1])
x_value = [x for x in range(25)]
y_value = [k * theta[1] + theta[0] for k in x_value]
plt.plot(x_value, y_value, color="r")
plt.xticks(np.arange(5, 30, step=5))
plt.yticks(np.arange(-5, 30, step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")
plt.show()


def predict(x, theta):
    predictions = x.dot(theta)

    return predictions[0]


predict1 = predict(np.array([1, 4]), theta) * 10000
print("For population = 40,000, we predict a profit of $" + str(round(predict1, 0)))

data2 = pd.read_csv("Multi_linear.txt", header=None)
data2.head()
data2.describe()


def featureNormalization(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    X_norm = (X - mean) / std

    return X_norm, mean, std


data_n2 = data2.values
m2 = len(data_n2[:, -1])
X2 = data_n2[:, 0:2].reshape(m2, 2)
X2, mean_X2, std_X2 = featureNormalization(X2)
X2 = np.append(np.ones((m2, 1)), X2, axis=1)
y2 = data_n2[:, -1].reshape(m2, 1)
theta2 = np.zeros((3, 1))

theta2, J_history2 = gradientDescent(X2, y2, theta2, 0.1, 400)

plt.plot(J_history2)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.show()

x_sample = featureNormalization(np.array([1650, 3]))[0]
x_sample = np.append(np.ones(1), x_sample)
predict3 = predict(x_sample, theta2)
print("For size of house = 1650, Number of bedroom = 3, we predict a house value of $" + str(round(predict3, 0)))
