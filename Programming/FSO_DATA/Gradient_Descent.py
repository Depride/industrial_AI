from sklearn.linear_model import LinearRegression
import numpy as np

# 선형 회귀 모델 생성
model = LinearRegression()

# X와 Y 좌표를 특성으로, 감도를 타겟으로 설정
X = combined_df[['X', 'Y']]
y = combined_df['감도 (dBm)']

# 모델 학습
model.fit(X, y)

# 모델의 계수와 절편 출력
a, b = model.coef_
c = model.intercept_

a, b, c
