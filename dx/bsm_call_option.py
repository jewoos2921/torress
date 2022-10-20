# 블랙-숄즈-머튼 모형에 의한 유러피안 콜 옵션의 가치 산정 베가와 내재 변동성 계산 포함
from math import log, sqrt, exp
from scipy import stats


class bsm_call_option(object):
    """
    속성
    =========
    S0: float
        초기 주가/지수 수준
    K: float
        행사가
    T: float
        만기 (연수로 계산)
    r: float
        고정 단기 무위험 이자율
    sigma: float
        변동성

    메서드
    ======
    value: float
        유러피안 콜 옵션의 현재 가치
    vega: float
        콜 옵션의 베가 반환
    imp_vol: float
        주어진 가격에 대한 내재 변동성 반환
    """

    def __init__(self, S0, K, T, r, sigma):
        self.S0 = float(S0)
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    def value(self):
        """옵션 가치 반환"""
        d1 = ((log(self.S0 / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * sqrt(self.T)))
        d2 = ((log(self.S0 / self.K) + (self.r - 0.5 * self.sigma ** 2) * self.T) / (self.sigma * sqrt(self.T)))
        value = (self.S0 * stats.norm.cdf(d1, 0.0, 1.0) - self.K * exp(-self.r * self.T) * stats.norm.cdf(d2, 0.0, 1.0))
        return value

    def vega(self):
        d1 = ((log(self.S0 / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * sqrt(self.T)))
        vega = self.S0 * stats.norm.pdf(d1, 0.0, 1.0) * sqrt(self.T)
        return vega

    def imp_vol(self, C0, sigma_est=0.3, it=100):
        option = bsm_call_option(self.S0, self.K, self.T, self.r, sigma_est)
        for i in range(it):
            option.sigma -= (option.value() - C0) / option.value()
        return option.sigma

