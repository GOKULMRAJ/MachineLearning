import pickle

import math
import numpy as np
import quandl as q
from matplotlib import style
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression

style.use('ggplot')
df = q.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. Close', 'Adj. High', 'Adj. Low', 'Adj. Volume']]
df['HLP'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100
df['OCP'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100
df = df[['Adj. Close', 'HLP', 'OCP', 'Adj. Volume']]
forecast = int(math.ceil(0.01 * len(df)))
df.fillna(-9999, inplace=True)
df['Label'] = df['Adj. Close'].shift(-forecast)

X = np.array(df.drop(['Label'], 1))
X = X[:-forecast]
X_test = X[-forecast:]
df.dropna(inplace=True)
y = np.array(df['Label'])

X = preprocessing.scale(X)

X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X, y, test_size=0.2)

classifier = LinearRegression()
classifier.fit(X_train, y_train)
with open('linebackers.pickle', 'wb') as f:
    pickle.dump(classifier, f)
P = open('linebackers.pickle', 'rb')
classifier = pickle.load(P)

y_test = classifier.predict(X_test)

accuracy = classifier.score(X_test, y_test)
print(accuracy)
