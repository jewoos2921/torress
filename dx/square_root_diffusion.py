# 시뮬레이션 클래스 - 제곱근 확산 모형

import numpy as np

from sn_random_numbers import sn_random_numbers
from simulation_class import simulation_class


class square_root_diffusion(simulation_class):
    """
    Coc-Ingersoll-Ross 제곱근 확산 모형에 기반한 시뮬레이션 경로를 생성하는 클래스

    속성
    ==========
    name: string
        객체 이름
    mar_env: instance of market_environment
        시뮬레이션을 위한 시장 환경
    corr: boolean
        다른 시뮬레이션 모델 객체와 상관이 있으면 True
    메서드
    ==========
    update:
        인숫값 갱신
    generate_paths:
        시장 환경이 주어지면 몬테카를로 경로를 반환
    """

    def __init__(self, name, mar_env, corr=False):
        super(square_root_diffusion, self).__init__(name, mar_env, corr)
        self.kappa = mar_env.get_constant("kappa")  # 평균 회귀 계수
        self.theta = mar_env.get_constant("theta")  # 확률과정의 장기 평균

    def update(self, initial_value=None, volatility=None, kappa=None,
               theta=None, final_date=None):
        if initial_value is not None:
            self.initial_value = initial_value
        if volatility is not None:
            self.volatility = volatility
        if kappa is not None:
            self.kappa = kappa
        if theta is not None:
            self.theta = theta
        if final_date is not None:
            self.final_date = final_date
        self.instrument_values = None

    def generate_paths(self, fixed_seed=False, day_count=365.):
        if self.time_grid is None:
            # 제너릭 시뮬레이션 클래스 메서드
            self.generate_time_grid()

        # 시간 그리드의 날짜 개수
        M = len(self.time_grid)
        # 경로 개수
        I = self.paths
        # 경로 시뮬레이션을 위해 ndarray 초기화
        paths = np.zeros((M, I))
        paths_ = np.zeros_like(paths)

        paths_[0] = self.initial_value
        paths[0] = self.initial_value
        # 상관성이없는 경우 난수 생성
        if not self.correlated:
            rnd = sn_random_numbers((1, M, I), fixed_seed=fixed_seed)
        else:
            # 상관성이 있는 경우 주어진 시장 환경의 난수 객체를 사용
            rnd = self.random_numbers

        # 확률 과정의 단기 이자율 사용
        for t in range(1, len(self.time_grid)):
            dt = (self.time_grid[t] - self.time_grid[t - 1]).days / day_count
            # 관련된 난수 집합에서 올바른 시간 구간을 선택
            if not self.correlated:
                ran = rnd[t]
            else:
                # 포트폴리오에 상관관계가 있는 경우
                ran = np.dot(self.cholesky_matrix, rnd[:, t, :])
                ran = ran[self.rn_set]

            # 풀 트렁케이션 오일러 이산화
            paths[t] = (paths[t - 1] + self.kappa *
                        (self.theta - np.maximum(0, paths_[t - 1, :])) * dt +
                        np.sqrt(np.maximum(0, paths_[t - 1, :])) * self.volatility * np.sqrt(dt) * ran)

            # 해당 날짜의 시뮬레이션 값 생성
            self.instrument_values = paths
