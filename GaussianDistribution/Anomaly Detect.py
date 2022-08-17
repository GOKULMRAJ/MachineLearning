import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

mat = loadmat("Computer_data.mat")
X = mat["X"]
Xval = mat["Xval"]
yval = mat["yval"]

plt.scatter(X[:, 0], X[:, 1], marker="x")
plt.xlim(0, 30)
plt.ylim(0, 30)
plt.xlabel("Latency (ms)")
plt.ylabel("Throughput (mb/s)")
plt.show()


def Gaussian(X):
    m = X.shape[0]

    sum = np.sum(X, axis=0)
    mu = 1 / m * sum

    var = 1 / m * np.sum((X - mu) ** 2, axis=0)

    return mu, var


mu, sigma2 = Gaussian(X)


def multivariateGaussian(X, mu, sigma2):
    k = len(mu)

    sigma2 = np.diag(sigma2)
    X = X - mu.T
    p = 1 / ((2 * np.pi) ** (k / 2) * (np.linalg.det(sigma2) ** 0.5)) * np.exp(
        -0.5 * np.sum(X @ np.linalg.pinv(sigma2) * X, axis=1))
    return p


p = multivariateGaussian(X, mu, sigma2)
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], marker="x")
X1, X2 = np.meshgrid(np.linspace(0, 35, num=70), np.linspace(0, 35, num=70))
p2 = multivariateGaussian(np.hstack((X1.flatten()[:, np.newaxis], X2.flatten()[:, np.newaxis])), mu, sigma2)
contour_level = 10 ** np.array([np.arange(-20, 0, 4, dtype=float)])[0]
plt.contour(X1, X2, p2[:, np.newaxis].reshape(X1.shape), contour_level)
plt.xlim(0, 35)
plt.ylim(0, 35)
plt.xlabel("Latency (ms)")
plt.ylabel("Throughput (mb/s)")
plt.show()


def selectThreshold(yval, pval):
    best_epi = 0
    best_F1 = 0

    stepsize = (max(pval) - min(pval)) / 1000
    epi_range = np.arange(pval.min(), pval.max(), stepsize)
    for epi in epi_range:
        predictions = (pval < epi)[:, np.newaxis]
        tp = np.sum(predictions[yval == 1] == 1)
        fp = np.sum(predictions[yval == 0] == 1)
        fn = np.sum(predictions[yval == 1] == 0)

        prec = tp / (tp + fp)
        rec = tp / (tp + fn)

        F1 = (2 * prec * rec) / (prec + rec)

        if F1 > best_F1:
            best_F1 = F1
            best_epi = epi

    return best_epi, best_F1


pval = multivariateGaussian(Xval, mu, sigma2)
epsilon, F1 = selectThreshold(yval, pval)
print("Best epsilon found using cross-validation:", epsilon)
print("Best F1 on Cross Validation Set:", F1)

plt.figure(figsize=(8, 6))

plt.scatter(X[:, 0], X[:, 1], marker="x")

X1, X2 = np.meshgrid(np.linspace(0, 35, num=70), np.linspace(0, 35, num=70))
p2 = multivariateGaussian(np.hstack((X1.flatten()[:, np.newaxis], X2.flatten()[:, np.newaxis])), mu, sigma2)
contour_level = 10 ** np.array([np.arange(-20, 0, 4, dtype=float)])[0]
plt.contour(X1, X2, p2[:, np.newaxis].reshape(X1.shape), contour_level)

outliers = np.nonzero(p < epsilon)[0]
plt.scatter(X[outliers, 0], X[outliers, 1], marker="o", facecolor="none", edgecolor="r", s=70)

plt.xlim(0, 35)
plt.ylim(0, 35)
plt.xlabel("Latency (ms)")
plt.ylabel("Throughput (mb/s)")
plt.show()
