
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
dataset2 = pd.read_csv('test.csv')
X_train = dataset.iloc[:, 1: - 1].values
y_train = dataset.iloc[:, 25].values
X_test = dataset2.iloc[:, 1:]

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import ExtraTreesRegressor
regressor = ExtraTreesRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X_train, y_train)


# Predicting a new result
y_pred = regressor.predict(X_test)


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10)
accuracies.mean()