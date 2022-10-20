import random
import time
from math import exp, sqrt
import numpy as np

S0 = 100  # 초기 지수 수준
r = 0.05  # 일정한 단기 금리
T = 1.0  # 일년이라는 기간을 쪼개는 부분들의 시계
sigma = 0.2  # 상수 변동성 계수
values = []
for _ in range(1000000):
    ST = S0 * exp((r - 0.5 * sigma ** 2) * T + sigma * random.gauss(0, 1) * sqrt(T))
    values.append(ST)

# 넘파이 버전
ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T +
                 sigma * np.random.standard_normal(1000000) * np.sqrt(T))
