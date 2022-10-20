# 가치 평가 - 유러피한 행사 클래스

import numpy as np
from valuation_class import valuation_class


class valuation_mcs_european(valuation_class):
    """
    단일 요인 몬테카를로 시뮬레이션을 사용한 임의의 페이오프를 가진 유러피안 옵션 가치 평가 클래스

    메서드
    ============
    generate_payoff:
        경로와 페이오프 함수가 주어지면 페이오프 계산
    present_value:
        몬테카를로 추정기를 사용한 현재 가치 계산
    """

    def generate_payoff(self, fixed_seed=False):
        """

        :param fixed_seed: bool
            가치 평가에 고정된 시드값
        :return:
        """
        try:
            # 행사가는 반드시 필요하지는 않다.
            strike = self.strike  # 옵션 행사가
        except:
            pass
        paths = self.underlying.get_instrument_values(fixed_seed=fixed_seed)
        time_grid = self.underlying.time_grid

        try:
            time_index = np.where(time_grid == self.maturity)[0]
            time_index = int(time_index)
        except:
            print("Maturity date not in time grid of underlying")
        maturity_value = paths[time_index]
        # 전체 경로의 평균값
        mean_value = np.mean(paths[:time_index], axis=1)
        # 전체 경로의 최대값
        max_value = np.amax(paths[:time_index], axis=1)[-1]
        # 전체 경로의 최솟값
        min_value = np.amin(paths[:time_index], axis=1)[-1]
        try:
            payoff = eval(self.payoff_func)
            return payoff
        except:
            print("Error evaluating payoff function")

    def present_value(self, accuracy=6,
                      fixed_seed=False, full=False):
        """

        :param accuracy: int
            반환 결과의 자릿수
        :param fixed_seed: bool
            가치 평가에 고정된 시드값을 사용
        :param full: bool
            현재 가치에 대한 전체 1차원 배열 반환
        :return:
        """
        cash_flow = self.generate_payoff(fixed_seed=fixed_seed)
        discount_factor = self.discount_curve.get_discount_factors((self.pricing_date, self.maturity))[0, 1]
        result = discount_factor * np.sum(cash_flow) / len(cash_flow)
        if full:
            return round(result, accuracy), discount_factor * cash_flow
        else:
            return round(result, accuracy)

