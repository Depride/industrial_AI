# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 20:41:49 2022

@author: howat
"""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

y_true = ["positive", "negative", "negative", "positive", "positive",
          "positive", "negative"]
y_pred = ["positive", "negative", "positive", "positive", "negative",
          "positive", "positive"]

cm = confusion_matrix(y_true, y_pred)
print('confusion matrix')
print(cm)
a = accuracy_score(y_true, y_pred)
print('accuracy: ', a)

y_pred = [0, 5, 2, 4]
y_true = [0, 1, 2, 3]
a = accuracy_score(y_true, y_pred)
print('accuracy: ', a)