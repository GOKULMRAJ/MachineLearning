import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('diamonds.csv')
# print(data.head())
# print(data.shape)
# print(data.info())
# print(data.isnull().sum().sum())

X = data.iloc[:, 1:]
X = X.drop('price', axis=1)

y = data['price']
# print(y.head())

duplicated = X[data.duplicated(keep='first')]
X = X.drop(duplicated.index, axis=0)
y = y.drop(duplicated.index, axis=0)
# print(data.head())

categories = [col for col in X if X[col].dtype == 'object']
# for i in range(X.shape[1]):
#     for k in range(i + 1, X.shape[1]):
#         if not (X.columns[i] in categories or X.columns[k] in categories):
#             for category in categories:
#                 sns.scatterplot(x=X.columns[i], y=X.columns[k], data=X, hue=category)
#                 plt.show()

from sklearn.preprocessing import OrdinalEncoder

ord_color = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
ord_cut = data['cut'].value_counts().sort_values().index.tolist()
ord_clarity = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

ordered = {
    'color': ord_color,
    'cut': ord_cut,
    'clarity': ord_clarity
}

for category in categories:
    ordinal_encoder = OrdinalEncoder(categories=[ordered[category]])
    X[category] = ordinal_encoder.fit_transform(X[[category]])

X = X.drop('x', axis=1)
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=.40, random_state=42)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV

my_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('polynomial_features', PolynomialFeatures()),
    ('ridge_regression', Ridge())
])

params = {
    'polynomial_features__degree': [1, 2, 3],
    'ridge_regression__alpha': np.geomspace(1e-5, 1e5, num=50)
}

grid = GridSearchCV(my_pipeline, params, cv=10)

grid.fit(X_train, y_train)
prediction = grid.predict(X_valid)
print(mean_absolute_error(prediction, y_valid))
