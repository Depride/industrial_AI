import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# 한글 폰트 설정
font_path = 'C:/Windows/Fonts/malgun.ttf'  # 사용할 한글 폰트의 경로를 지정하세요.
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# 최고 수신 영역의 정보
x0, y0 = -3.241, -4.653  # 최고 수신 영역의 중심 좌표 (mm)
P0 = -2.41  # 최고 수신 영역의 광량 (dBm)

# 최저 수신감도
P_min = -21  # dBm

# 표준 편차 계산 (불규칙성을 조절)
sigma = 3.0  # 표준 편차를 조절하여 데이터에 불규칙성을 주세요.

# 그래프를 그리기 위한 좌표 생성
x = np.linspace(-25, 25, 500)
y = np.linspace(-25, 25, 500)
X, Y = np.meshgrid(x, y)

# 가우시안 함수 계산
P = P0 * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))

# 최저 수신감도 이하의 값은 최저 수신감도로 설정
P[P < P_min] = P_min

# 불규칙성을 추가하여 더미 데이터 생성
random_noise = np.random.normal(0, 1, P.shape)  # 평균 0, 표준 편차 1의 정규 분포에서 추출한 잡음
P_with_noise = P + random_noise

# 2D 플롯 그리기
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, P_with_noise, cmap='viridis', levels=100)
plt.colorbar(label='수신감도 (dBm)')
plt.xlabel('X 좌표 (mm)')
plt.ylabel('Y 좌표 (mm)')
plt.title('25mm 원에 수신되는 광의 수신감도 분포 (불규칙성 포함)')
plt.show()
