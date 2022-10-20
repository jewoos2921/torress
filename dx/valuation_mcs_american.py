# 가치 평가 - 아메리칸 행사방식 클래스

import numpy as np
from valuation_class import valuation_class


class valuation_mcs_american(valuation_class):
    """
    단일 요인 몬테카를로 시뮬레이션을 사용한
    임의의 페이오프를 가진 아메리카 옵션의 가치 평가 클래스
     메서드
    ============
    generate_payoff:
        경로와 페이오프 함수가 주어지면 페이오프 계산
    present_value:
        롱스태프-슈바르츠 방법에 따른 현재 가치
    """

    def generate_payoff(self, fixed_seed=False):
        try:
            # 행사가는 반드시 필요하지는 않다.
            strike = self.strike  # 옵션 행사가
        except:
            pass
        paths = self.underlying.get_instrument_values(fixed_seed=fixed_seed)
        time_grid = self.underlying.time_grid
        time_index_start = int(np.where(time_grid == self.pricing_date)[0])
        time_index_end = int(np.where(time_grid == self.maturity)[0])
        instrument_values = paths[time_index_start:time_index_end + 1]
        payoff = eval(self.payoff_func)
        return instrument_values, payoff, time_index_start, time_index_end

    def present_value(self, accuracy=6,
                      fixed_seed=False, bf=5, full=False):
        """

        :param accuracy: int
            반환 결과의 자릿수
        :param fixed_seed: bool
            가치 평가에 고정된 시드값을 사용
        :param bf: int
            회귀용 기저 함수의 개수
        :param full: bool
            현재 가치에 대한 전체 1차원 배열 반환
        :return:
        """
        instrument_values, inner_values, time_index_start, time_index_end = self.generate_payoff(fixed_seed=fixed_seed)
        time_list = self.underlying.time_grid[time_index_start: time_index_end + 1]
        discount_factors = self.discount_curve.get_discount_factors(time_list, dtobjects=True)
        V = inner_values[-1]
        for t in range(len(time_list) - 2, 0, -1):
            # 주어진 시간 구간의 할인율 유도
            df = discount_factors[t, 1] / discount_factors[t + 1, 1]
            # 회귀분석 단계
            rg = np.polyfit(instrument_values[t], V * bf, bf)
            # 경로별 보유가치 계산
            C = np.polyval(rg, instrument_values[t])
            # 추가적인 의사결정 단계
            # 조건을 만족하는 경우 (내재 > 회귀보유)
            # 이경우에 내재가치 선택
            V = np.where(inner_values[t] > C, inner_values[t], V * df)
            df = discount_factors[0, 1] / discount_factors[1, 1]
            result = df * np.sum(V) / len(V)
            if full:
                return round(result, accuracy), df * V
            else:
                return round(result, accuracy)
