# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 21:20:29 2022

@author: howat
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

data = np.array([[30, 12], [150, 25], [300, 35], [400, 48], [130, 21],
                 [240, 33], [350, 46], [200, 41], [100, 20], [110, 23],
                 [190, 32], [120, 24], [130, 19], [270, 37], [255, 24]])

plt.scatter(data[:, 0], data[:, 1]) #데이터 위치의 산포도 출력
plt.title("Linear Regression")
plt.xlabel("Delivery Distance")
plt.ylabel("Delievery Time")
plt.axis([0, 420, 0, 50])

x = data[:, 0].reshape(-1, 1) #입력
y = data[:, 1].reshape(-1, 1) #출력

model = LinearRegression()
model.fit(x,y)          #모델 학습

y_pred = model.predict(x)   #예측값 계산
plt.plot(x, y_pred, color='r')
plt.show()