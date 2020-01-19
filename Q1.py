# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 13:15:02 2020

@author: IKhan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_startups.csv')
X = dataset.iloc[:, 3:4].values #0=NewYork, 1=california , 2=florida
y = dataset.iloc[:, 4:5].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Country vs Profit (Training set)')
plt.xlabel('Country 0=Newyork 1=California 2=Florida')
plt.ylabel('Profit')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Country vs Profit (Test set)')
plt.xlabel('Country 0=Newyork 1=California 2=Florida')
plt.ylabel('Profit')
plt.show()




