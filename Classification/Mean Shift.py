import matplotlib.pyplot as plt
import numpy as np

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11],
              [8, 2],
              [10, 2],
              [9, 3]])
plt.scatter(X[:, 0], X[:, 1], s=150)

colors = 10 * ['g', 'r', 'c', 'b', 'k']


class MS:
    def __init__(self, radius=4):
        self.radius = radius

    def fit(self, data):
        centroids = {}

        for i in range(len(data)):
            centroids[i] = data[i]

        while True:
            new = []
            for i in range(len(centroids)):
                Range = []
                centroid = centroids[i]
                for x in data:
                    if np.linalg.norm(x - centroid) < self.radius:
                        Range.append(x)
                new.append(tuple(np.average(Range, axis=0)))

            unique = sorted(list(set(new)))

            pop = []

            for i in unique:
                for k in unique:
                    if i == k:
                        pass
                    elif np.linalg.norm(np.array(i) - np.array(k)) <= self.radius:
                        pop.append(k)
                        break

            for i in pop:
                if i in unique:
                    unique.remove(i)

            prev = dict(centroids)
            centroids = {}
            for i in range(len(unique)):
                centroids[i] = unique[i]

            stable = True
            for i in centroids:
                if not np.array_equal(centroids, prev):
                    stable = False
                if not stable:
                    break
            if stable:
                break
        self.centroids = centroids

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = (distances.index(min(distances)))
        return classification


classifier = MS()
classifier.fit(X)

for c in classifier.centroids:
    plt.scatter(classifier.centroids[c][0], classifier.centroids[c][1], color='k', marker='*', s=150)

plt.show()
