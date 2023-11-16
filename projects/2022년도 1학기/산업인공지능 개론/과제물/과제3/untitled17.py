# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 21:29:03 2022

@author: howat
"""

from sklearn.preprosessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

import numpy as np

np.random.seed(1)
X = np.random.rand(40,1)**2
y = (10-1./(X.ravel()+0.1)) + np.random.randn(40)

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
X_test = np.linspace(0.1, 1.1, 500).reshape(-1, 1)
fig = plt.figure(figsize=(12, 10))
for i, degree in enumerate([1, 3, 5, 10], start=1):
    ax = fig.add_subplot(2,2,i)
    ax.scatter(X.ravel(), y, s=15)
    y_test = make_pipeline(PolynomialFeatures(degree), LinearRegression()).fit(X, y).predict(X_test)
    ax.plot(X_test.ravel(), y_test, label='degree={0}'.format(degree))
    ax.set_xlim(-0.1, 1.0)
    ax.set.ylim(-2, 12)
    ax.legend(loc='best')