import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

mat = loadmat("ex6data1.mat")
X = mat["X"]
y = mat["y"]

m, n = X.shape[0], X.shape[1]
pos, neg = (y == 1).reshape(m, 1), (y == 0).reshape(m, 1)
plt.scatter(X[pos[:, 0], 0], X[pos[:, 0], 1], c="r", marker="+", s=50)
plt.scatter(X[neg[:, 0], 0], X[neg[:, 0], 1], c="y", marker="o", s=50)

from sklearn.svm import SVC

classifier = SVC(kernel="linear")
classifier.fit(X, np.ravel(y))

plt.figure(figsize=(8, 6))
plt.scatter(X[pos[:, 0], 0], X[pos[:, 0], 1], c="r", marker="+", s=50)
plt.scatter(X[neg[:, 0], 0], X[neg[:, 0], 1], c="y", marker="o", s=50)

X_1, X_2 = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 1].max(), num=100),
                       np.linspace(X[:, 1].min(), X[:, 1].max(), num=100))
plt.contour(X_1, X_2, classifier.predict(np.array([X_1.ravel(), X_2.ravel()]).T).reshape(X_1.shape), 1, colors="b")
plt.xlim(0, 4.5)
plt.ylim(1.5, 5)

classifier2 = SVC(C=100, kernel="linear")
classifier2.fit(X, np.ravel(y))

plt.figure(figsize=(8, 6))
plt.scatter(X[pos[:, 0], 0], X[pos[:, 0], 1], c="r", marker="+", s=50)
plt.scatter(X[neg[:, 0], 0], X[neg[:, 0], 1], c="y", marker="o", s=50)

X_3, X_4 = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 1].max(), num=100),
                       np.linspace(X[:, 1].min(), X[:, 1].max(), num=100))
plt.contour(X_3, X_4, classifier2.predict(np.array([X_3.ravel(), X_4.ravel()]).T).reshape(X_3.shape), 1, colors="b")
plt.xlim(0, 4.5)
plt.ylim(1.5, 5)

mat2 = loadmat("ex6data2.mat")
X2 = mat2["X"]
y2 = mat2["y"]

m2, n2 = X2.shape[0], X2.shape[1]
pos2, neg2 = (y2 == 1).reshape(m2, 1), (y2 == 0).reshape(m2, 1)
plt.figure(figsize=(8, 6))
plt.scatter(X2[pos2[:, 0], 0], X2[pos2[:, 0], 1], c="r", marker="+")
plt.scatter(X2[neg2[:, 0], 0], X2[neg2[:, 0], 1], c="y", marker="o")
plt.xlim(0, 1)
plt.ylim(0.4, 1)

classifier3 = SVC(kernel="rbf", gamma=30)
classifier3.fit(X2, y2.ravel())

plt.figure(figsize=(8, 6))
plt.scatter(X2[pos2[:, 0], 0], X2[pos2[:, 0], 1], c="r", marker="+")
plt.scatter(X2[neg2[:, 0], 0], X2[neg2[:, 0], 1], c="y", marker="o")

X_5, X_6 = np.meshgrid(np.linspace(X2[:, 0].min(), X2[:, 1].max(), num=100),
                       np.linspace(X2[:, 1].min(), X2[:, 1].max(), num=100))
plt.contour(X_5, X_6, classifier3.predict(np.array([X_5.ravel(), X_6.ravel()]).T).reshape(X_5.shape), 1, colors="b")
plt.xlim(0, 1)
plt.ylim(0.4, 1)

mat3 = loadmat("ex6data3.mat")
X3 = mat3["X"]
y3 = mat3["y"]
Xval = mat3["Xval"]
yval = mat3["yval"]

m3, n3 = X3.shape[0], X3.shape[1]
pos3, neg3 = (y3 == 1).reshape(m3, 1), (y3 == 0).reshape(m3, 1)
plt.figure(figsize=(8, 6))
plt.scatter(X3[pos3[:, 0], 0], X3[pos3[:, 0], 1], c="r", marker="+", s=50)
plt.scatter(X3[neg3[:, 0], 0], X3[neg3[:, 0], 1], c="y", marker="o", s=50)


def Params(X, y, Xval, yval, vals):
    acc = 0
    best_c = 0
    best_gamma = 0
    for i in vals:
        C = i
        for j in vals:
            gamma = 1 / j
            classifier = SVC(C=C, gamma=gamma)
            classifier.fit(X, y)
            prediction = classifier.predict(Xval)
            score = classifier.score(Xval, yval)
            if score > acc:
                acc = score
                best_c = C
                best_gamma = gamma
    return best_c, best_gamma


vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
C, gamma = Params(X3, y3.ravel(), Xval, yval.ravel(), vals)
classifier4 = SVC(C=C, gamma=gamma)
classifier4.fit(X3, y3.ravel())

plt.figure(figsize=(8, 6))
plt.scatter(X3[pos3[:, 0], 0], X3[pos3[:, 0], 1], c="r", marker="+", s=50)
plt.scatter(X3[neg3[:, 0], 0], X3[neg3[:, 0], 1], c="y", marker="o", s=50)

X_7, X_8 = np.meshgrid(np.linspace(X3[:, 0].min(), X3[:, 1].max(), num=100),
                       np.linspace(X3[:, 1].min(), X3[:, 1].max(), num=100))
plt.contour(X_7, X_8, classifier4.predict(np.array([X_7.ravel(), X_8.ravel()]).T).reshape(X_7.shape), 1, colors="b")
plt.xlim(-0.6, 0.3)
plt.ylim(-0.7, 0.5)
plt.show()

import re
from nltk.stem import PorterStemmer

file_contents = open("emailSample1.txt", "r").read()
vocabList = open("vocab.txt", "r").read()

vocabList = vocabList.split("\n")[:-1]

vocabList_D = {}
for NW in vocabList:
    value, key = NW.split("\t")[:]
    vocabList_D[key] = value


def processEmail(email_contents, vocabList_D):
    email_contents = email_contents.lower()

    email_contents = re.sub("[0-9]+", "number", email_contents)

    email_contents = re.sub("[http|https]://[^\s]*", "httpaddr", email_contents)

    email_contents = re.sub("[^\s]+@[^\s]+", "emailaddr", email_contents)

    email_contents = re.sub("[$]+", "dollar", email_contents)

    specialChar = ["<", "[", "^", ">", "+", "?", "!", "'", ".", ",", ":"]
    for char in specialChar:
        email_contents = email_contents.replace(str(char), "")
    email_contents = email_contents.replace("\n", " ")

    ps = PorterStemmer()
    email_contents = [ps.stem(token) for token in email_contents.split(" ")]
    email_contents = " ".join(email_contents)

    word_indices = []

    for char in email_contents.split():
        if len(char) > 1 and char in vocabList_D:
            word_indices.append(int(vocabList_D[char]))

    return word_indices


word_indices = processEmail(file_contents, vocabList_D)


def emailFeatures(word_indices, vocabList_D):
    n = len(vocabList_D)

    features = np.zeros((n, 1))

    for i in word_indices:
        features[i] = 1

    return features


features = emailFeatures(word_indices, vocabList_D)
print("Length of feature vector: ", len(features))
print("Number of non-zero entries: ", np.sum(features))

spam_mat = loadmat("spamTrain.mat")
X_train = spam_mat["X"]
y_train = spam_mat["y"]

C = 0.1
spam_svc = SVC(C=0.1, kernel="linear")
spam_svc.fit(X_train, y_train.ravel())
print("Training Accuracy:", (spam_svc.score(X_train, y_train.ravel())) * 100, "%")

spam_mat_test = loadmat("spamTest.mat")
X_test = spam_mat_test["Xtest"]
y_test = spam_mat_test["ytest"]

spam_svc.predict(X_test)
print("Test Accuracy:", (spam_svc.score(X_test, y_test.ravel())) * 100, "%")

