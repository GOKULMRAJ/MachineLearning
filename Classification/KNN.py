import numpy as np
from sklearn import preprocessing, neighbors, model_selection
import pandas as pd
import warnings
import random
from collections import Counter

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], axis=1, inplace=True)

X = np.array(df.drop(['class'], axis=1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

classifier = neighbors.KNeighborsClassifier()
classifier.fit(X_train, y_train)

accuracy = classifier.score(X_test, y_test)
print(accuracy)


def KNN(dataset, prediction, k=5):
    if len(dataset) >= k:
        warnings.warn('k is less than total groups!!')
    distances = []
    for group in dataset:
        for features in dataset[group]:
            ED = np.linalg.norm(np.array(features) - np.array(prediction))
            distances.append([ED, group])

    nearest = [x[1] for x in sorted(distances)[:k]]
    result = Counter(nearest).most_common(1)[0][0]
    confidence = Counter(nearest).most_common(1)[0][1] / k

    return result, confidence


df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], axis=1, inplace=True)
data = df.astype(float).values.tolist()

random.shuffle(data)

test_size = 0.2
train_set = {2: [], 4: []}
test_set = {2: [], 4: []}

train_data = data[:-int(test_size * len(data))]
test_data = data[-int(test_size * len(data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
    for features in test_set[group]:
        vote, confidence = KNN(train_set, features)
        if group == vote:
            correct += 1
        else:
            print(confidence)
        total += 1

print(correct / total)
