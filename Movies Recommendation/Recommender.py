import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

mat3 = loadmat("ex8_movies.mat")
mat4 = loadmat("ex8_movieParams.mat")
Y = mat3["Y"]
R = mat3["R"]
X = mat4["X"]
Theta = mat4["Theta"]

plt.figure(figsize=(8, 16))
plt.imshow(Y)
plt.xlabel("Users")
plt.ylabel("Movies")
plt.show()


def CostFunc(params, Y, R, num_users, num_movies, num_features, Lambda):
    X = params[:num_movies * num_features].reshape(num_movies, num_features)
    Theta = params[num_movies * num_features:].reshape(num_users, num_features)

    predictions = X @ Theta.T
    err = (predictions - Y)
    J = 1 / 2 * np.sum((err ** 2) * R)

    reg_X = Lambda / 2 * np.sum(Theta ** 2)
    reg_Theta = Lambda / 2 * np.sum(X ** 2)
    reg_J = J + reg_X + reg_Theta

    X_grad = err * R @ Theta
    Theta_grad = (err * R).T @ X
    grad = np.append(X_grad.flatten(), Theta_grad.flatten())

    reg_X_grad = X_grad + Lambda * X
    reg_Theta_grad = Theta_grad + Lambda * Theta
    reg_grad = np.append(reg_X_grad.flatten(), reg_Theta_grad.flatten())

    return J, grad, reg_J, reg_grad


num_users, num_movies, num_features = 4, 5, 3
X_test = X[:num_movies, :num_features]
Theta_test = Theta[:num_users, :num_features]
Y_test = Y[:num_movies, :num_users]
R_test = R[:num_movies, :num_users]
params = np.append(X_test.flatten(), Theta_test.flatten())

J, grad = CostFunc(params, Y_test, R_test, num_users, num_movies, num_features, 0)[:2]
print("Cost at loaded parameters:", J)

J2, grad2 = CostFunc(params, Y_test, R_test, num_users, num_movies, num_features, 2)[2:]
print("Cost at loaded parameters (lambda = 2):", J2)

my_ratings = np.zeros((1682, 1))

my_ratings[0] = 4
my_ratings[97] = 2
my_ratings[6] = 3
my_ratings[11] = 5
my_ratings[53] = 4
my_ratings[63] = 5
my_ratings[65] = 3
my_ratings[68] = 5
my_ratings[82] = 4
my_ratings[225] = 5
my_ratings[354] = 5

movieList = open("movie_ids.txt", "r").read().split("\n")[:-1]

print("New user ratings:\n")
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print("Rated", int(my_ratings[i]), "for index", movieList[i])


def normalizeRatings(Y, R):
    m, n = Y.shape[0], Y.shape[1]
    Ymean = np.zeros((m, 1))
    Ynorm = np.zeros((m, n))

    for i in range(m):
        Ymean[i] = np.sum(Y[i, :]) / np.count_nonzero(R[i, :])
        Ynorm[i, R[i, :] == 1] = Y[i, R[i, :] == 1] - Ymean[i]

    return Ynorm, Ymean


def gradientDescent(initial_parameters, Y, R, num_users, num_movies, num_features, alpha,
                    num_iters, Lambda):
    X = initial_parameters[:num_movies * num_features].reshape(num_movies, num_features)
    Theta = initial_parameters[num_movies * num_features:].reshape(num_users, num_features)

    J_history = []

    for i in range(num_iters):
        params = np.append(X.flatten(), Theta.flatten())
        cost, grad = CostFunc(params, Y, R, num_users, num_movies, num_features, Lambda)[2:]

        X_grad = grad[:num_movies * num_features].reshape(num_movies, num_features)
        Theta_grad = grad[num_movies * num_features:].reshape(num_users, num_features)
        X = X - (alpha * X_grad)
        Theta = Theta - (alpha * Theta_grad)
        J_history.append(cost)

    paramsFinal = np.append(X.flatten(), Theta.flatten())
    return paramsFinal, J_history


Y = np.hstack((my_ratings, Y))
R = np.hstack((my_ratings != 0, R))

Ynorm, Ymean = normalizeRatings(Y, R)

num_users = Y.shape[1]
num_movies = Y.shape[0]
num_features = 10

X = np.random.randn(num_movies, num_features)
Theta = np.random.randn(num_users, num_features)
initial_parameters = np.append(X.flatten(), Theta.flatten())
Lambda = 10

paramsFinal, J_history = gradientDescent(initial_parameters, Y, R, num_users, num_movies, num_features,
                                         0.001, 400, Lambda)

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")
plt.show()

X = paramsFinal[:num_movies * num_features].reshape(num_movies, num_features)
Theta = paramsFinal[num_movies * num_features:].reshape(num_users, num_features)

p = X @ Theta.T + Ymean
my_predictions = p[:, 4][:, np.newaxis]

import pandas as pd

df = pd.DataFrame(np.hstack((my_predictions, np.array(movieList)[:, np.newaxis])))
df.sort_values(by=[0], ascending=False, inplace=True)
df.reset_index(drop=True, inplace=True)

print("Top recommendations for 5th user:\n")
for i in range(10):
    print("Predicting rating", round(float(df[0][i]), 1), " for index", df[1][i])
