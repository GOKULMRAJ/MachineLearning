import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans

df = pd.read_excel('titanic.xls')
df.drop(['body', 'name'], axis=1, inplace=True)
df.convert_dtypes(convert_integer=True)
df.fillna(0, inplace=True)


def convert_non_numeric(df):
    columns = df.columns.values

    for column in columns:
        conversion = {}

        def convert_to_int(val):
            return conversion[val]

        if df[column].dtype != int and df[column].dtype != float:
            unique = set(df[column].values.tolist())
            x = 0
            for k in unique:
                if k not in conversion:
                    conversion[k] = x
                    x += 1
            df[column] = list(map(convert_to_int, df[column]))

    return df


df = convert_non_numeric(df)

X = np.array(df.drop(['survived'], axis=1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

classifier = KMeans(n_clusters=2)
classifier.fit(X)

correct = 0
for i in range(len(X)):
    x = X[i].reshape(-1, len(X[i]))
    prediction = classifier.predict(x)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(X))

