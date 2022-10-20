# 시뮬레이션 클레스 - 베이스 클레스

import numpy as np
import pandas as pd


class simulation_class(object):
    """
    시뮬레이션 클래스의 기본 메서드

    속성
    ==========
    name: str
        객체의 이름
    mar_env: instance of market_environment
        시뮬레이션을 위한 시장 환경 데이터
    corr: bool
        다른 모델 객체와 상관성이 있으면 True

    메서드
    ==========
    generate_time_grid:
        시뮬레이션을 위한 시간 그리드 반환
    get_instrument_values:
        현재 상품 가치 (배열)을 반환
    """

    def __init__(self, name, mar_env, corr):
        self.name = name
        self.pricing_date = mar_env.pricing_date
        self.initial_value = mar_env.get_constant("initial_value")
        self.volatility = mar_env.get_constant("volatility")
        self.final_date = mar_env.get_constant("final_date")
        self.currency = mar_env.get_constant("currency")
        self.frequency = mar_env.get_constant("frequency")
        self.paths = mar_env.get_constant("paths")
        self.discount_curve = mar_env.get_curve("discount_curve")
        try:  # 만약 mar_env가 time_grid 객체를 가지고 있는 경우
            self.time_grid = mar_env.get_list("time_grid")
        except:
            self.time_grid = None
        try:  # 만약 특별한 날짜가 있는 경우 추가
            self.special_dates = mar_env.get_list("special_dates")
        except:
            self.special_dates = []
        self.instrument_values = None
        self.correlated = corr
        if corr is True:
            # 위험 요소가 서로 상관 관계가 있는 경우
            self.cholesky_matrix = mar_env.get_list("cholesky_matrix")
            self.rn_set = mar_env.get_list("rn_set")[self.name]
            self.random_numbers = mar_env.get_list("random_numbers")

    def generate_time_grid(self):
        start = self.pricing_date
        end = self.final_date
        # pandas date_range 함수
        # 주기가 평일인 경우 'B' , 주간인 경우 'W', 월간인 경우 'M'
        time_grid = pd.date_range(start=start, end=end,
                                  freq=self.frequency).to_pydatetime()
        time_grid = list(time_grid)
        # time_grid에 시작일, 종료일, 특별한 날짜 추가
        if start not in time_grid:
            time_grid.insert(0, start)
        # 리스트에 시작 일이 없으면 추가
        if end not in time_grid:
            time_grid.append(end)

        # 시작일에 종료일이 없으면 추가
        if len(self.special_dates) > 0:
            # 모든 특별한 날짜 추가
            time_grid.extend(self.special_dates)
            # 중복 제거
            time_grid = list(set(time_grid))
            time_grid.sort()
        self.time_grid = np.array(time_grid)

    def get_instrument_values(self, fixed_seed=True):
        if self.instrument_values is None:
            # 상품 가치가 없는 경우에 초기 시뮬레이션 시작
            self.generate_paths(fixed_seed=fixed_seed, day_count=365.)
        elif fixed_seed is False:
            self.generate_paths(fixed_seed=fixed_seed, day_count=365.)
        return self.instrument_values
