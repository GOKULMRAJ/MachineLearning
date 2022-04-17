import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

mat = loadmat("ex3data1.mat")

X = mat["X"]
y = mat["y"]

fig, axis = plt.subplots(10, 10, figsize=(8, 8))
for i in range(10):
    for j in range(10):
        axis[i, j].imshow(X[np.random.randint(0, 5001), :].reshape(20, 20, order="F"),
                          cmap="hot")
        axis[i, j].axis("off")
plt.show()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


mat2 = loadmat("ex3weights.mat")
Theta1 = mat2["Theta1"]
Theta2 = mat2["Theta2"]


def predict(Theta1, Theta2, X):
    m = X.shape[0]
    X = np.hstack((np.ones((m, 1)), X))

    a1 = sigmoid(X @ Theta1.T)
    a1 = np.hstack((np.ones((m, 1)), a1))
    a2 = sigmoid(a1 @ Theta2.T)

    return np.argmax(a2, axis=1) + 1


p2 = predict(Theta1, Theta2, X)
print("Training Set Accuracy:", sum(p2[:, np.newaxis] == y)[0] / len(y) * 100, "%")


def sigmoidGradient(z):
    sigmoid = 1 / (1 + np.exp(-z))

    return sigmoid * (1 - sigmoid)


def NNCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda):
    Theta1 = nn_params[:((input_layer_size + 1) * hidden_layer_size)].reshape(hidden_layer_size, input_layer_size + 1)
    Theta2 = nn_params[((input_layer_size + 1) * hidden_layer_size):].reshape(num_labels, hidden_layer_size + 1)

    m = X.shape[0]
    J = 0
    X = np.hstack((np.ones((m, 1)), X))
    y10 = np.zeros((m, num_labels))

    a1 = sigmoid(X @ Theta1.T)
    a1 = np.hstack((np.ones((m, 1)), a1))
    a2 = sigmoid(a1 @ Theta2.T)

    for i in range(1, num_labels + 1):
        y10[:, i - 1][:, np.newaxis] = np.where(y == i, 1, 0)
    for j in range(num_labels):
        J = J + sum(-y10[:, j] * np.log(a2[:, j]) - (1 - y10[:, j]) * np.log(1 - a2[:, j]))

    cost = 1 / m * J
    reg_J = cost + Lambda / (2 * m) * (np.sum(Theta1[:, 1:] ** 2) + np.sum(Theta2[:, 1:] ** 2))

    grad1 = np.zeros(Theta1.shape)
    grad2 = np.zeros(Theta2.shape)

    for i in range(m):
        xi = X[i, :]
        a1i = a1[i, :]
        a2i = a2[i, :]
        error = a2i - y10[i, :]
        d2 = Theta2.T @ error.T * sigmoidGradient(np.hstack((1, xi @ Theta1.T)))
        grad1 = grad1 + d2[1:][:, np.newaxis] @ xi[:, np.newaxis].T
        grad2 = grad2 + error.T[:, np.newaxis] @ a1i[:, np.newaxis].T

    grad1 = 1 / m * grad1
    grad2 = 1 / m * grad2

    grad1_reg = grad1 + (Lambda / m) * np.hstack((np.zeros((Theta1.shape[0], 1)), Theta1[:, 1:]))
    grad2_reg = grad2 + (Lambda / m) * np.hstack((np.zeros((Theta2.shape[0], 1)), Theta2[:, 1:]))

    return cost, grad1, grad2, reg_J, grad1_reg, grad2_reg


input_layer_size = 400
hidden_layer_size = 2
num_labels = 10
# nn_params = np.append(Theta1.flatten(), Theta2.flatten())
# J, reg_J = NNCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, 1)[0:4:3]
# print("Cost at parameters (non-regularized):", J, "\nCost at parameters (Regularized):", reg_J)


def randInitializeWeights(Lin, Lout):
    epi = (6 ** 1 / 2) / (Lin + Lout) ** 1 / 2

    W = np.random.rand(Lout, Lin + 1) * (2 * epi) - epi

    return W


initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)
initial_nn_params = np.append(initial_Theta1.flatten(), initial_Theta2.flatten())


def gradientDescentNN(X, y, initial_nn_params, alpha, num_iters, Lambda, input_layer_size,
                      hidden_layer_size, num_labels):
    Theta1 = initial_nn_params[:((input_layer_size + 1) * hidden_layer_size)].reshape(hidden_layer_size,
                                                                                      input_layer_size + 1)
    Theta2 = initial_nn_params[((input_layer_size + 1) * hidden_layer_size):].reshape(num_labels,
                                                                                      hidden_layer_size + 1)

    m = len(y)
    J_history = []

    for i in range(num_iters):
        nn_params = np.append(Theta1.flatten(), Theta2.flatten())
        cost, grad1, grad2 = NNCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)[
                             3:]
        Theta1 = Theta1 - (alpha * grad1)
        Theta2 = Theta2 - (alpha * grad2)
        J_history.append(cost)

    nn_paramsFinal = np.append(Theta1.flatten(), Theta2.flatten())
    return nn_paramsFinal, J_history


NNTheta, NNJ_history = gradientDescentNN(X, y, initial_nn_params, 0.8, 400, 1, input_layer_size, hidden_layer_size,
                                         num_labels)
Theta1 = NNTheta[:((input_layer_size + 1) * hidden_layer_size)].reshape(hidden_layer_size, input_layer_size + 1)
Theta2 = NNTheta[((input_layer_size + 1) * hidden_layer_size):].reshape(num_labels, hidden_layer_size + 1)

p3 = predict(Theta1, Theta2, X)
print("Training Set Accuracy:", sum(p3[:, np.newaxis] == y)[0] / len(y) * 100, "%")
