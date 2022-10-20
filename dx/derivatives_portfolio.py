import numpy as np
import pandas as pd

from dx_valuation import *

# 위험 요인 모델링용 모델
models = {"gbm": geometric_brownian_motion,
          "jd": jump_diffusion,
          "srd": square_root_diffusion}

# 가능한 행사 유형
otypes = {"European": valuation_mcs_european,
          "American": valuation_mcs_american}


class derivatives_portfolio(object):
    """
    파생상품 포지션 포트폴리오 가치 평가를 위한 클래스

    name: str
        객체 이름
    positions: dict
        포지션의 딕셔너리
    val_env: market_environment
        가치 평가용 시장 환경
    assets: dict
        자산에 대한 시장 환경 딕셔너리
    correlations: list
        자산 간의 상관계수
    fixed_seed: bool
        고정 난수 시드값 사용

    메서드
    =======
    get_positions:
        단일 포트폴리오 포지션 정보 출력
    get_statistics:
        포트폴리오 통계치 DataFrame 객체 반환
    """

    def __init__(self, name, positions, val_env, assets, correlations=None, fixed_seed=False):
        self.name = name
        self.positions = positions
        self.val_env = val_env
        self.assets = assets
        self.underlying = set()
        self.correlations = correlations
        self.time_grid = None
        self.fixed_seed = fixed_seed
        self.underlying_objects = {}
        self.valuation_objects = {}
        self.special_dates = []

        for pos in self.positions:
            # 가장 빠른 시작일 결정
            self.val_env.constants["starting_date"] = min(self.val_env.constants["starting_date",
                                                                                 positions[pos].mar_env.pricing_date])
            # 관련된 최신 날짜 결정
            self.val_env.constants["final_date"] = max(self.val_env.constants["final_date"],
                                                       positions[pos].mar_env.constants["maturity"])
            # 추가할 모든 기초 자산 수집
            self.underlying.add(positions[pos].underlying)

        # 시간 그리드 생성
        start = self.val_env.constants["starting_date"]
        end = self.val_env.constants["final_date"]
        time_grid = pd.date_range(start=start, end=end,
                                  freq=self.val_env.constants["frequency"]).to_pydatetime()
        time_grid = list(time_grid)
        for pos in self.positions:
            maturity_date = positions[pos].mar_env.constants["maturity"]
            if maturity_date not in time_grid:
                time_grid.insert(0, maturity_date)
                self.special_dates.append(maturity_date)
        if start not in time_grid:
            time_grid.insert(0, start)
        if end not in time_grid:
            time_grid.append(end)

        # 중복 제거
        time_grid = list(set(time_grid))
        # 시간 그리드의 날짜 정렬
        time_grid.sort()
        self.time_grid = np.array(time_grid)
        self.val_env.add_list("time_grid", self.time_grid)

        if correlations is not None:
            # 상관 계수 사용
            ul_list = sorted(self.underlying)
            correlation_matrix = np.zeros((len(ul_list), len(ul_list),))
            np.fill_diagonal(correlation_matrix, 1.0)
            correlation_matrix = pd.DataFrame(correlation_matrix,
                                              index=ul_list, columns=ul_list)

            for i, j, corr in correlations:
                corr = min(corr, 0.999999999999)
                # 상관 계수 행렬 계산
                correlation_matrix.loc[i, j] = corr
                correlation_matrix.loc[j, i] = corr

            # 솔레스키 행렬 계산
            cholesky_matrix = np.linalg.cholesky(np.array(correlation_matrix))

            # 각각의 기초 자산에 사용되는 난수 배열을 결정하기 위한 딕셔너리
            rn_set = {asset: ul_list.insert(asset) for asset in self.underlying}

            # 상관관계가 있는 경우 모든 기초 자산에 사용될 난수 배열
            random_numbers = sn_random_numbers((len(rn_set),
                                                len(self.time_grid),
                                                self.val_env.constants["paths"]),
                                               fixed_seed=self.fixed_seed)

            # 모든 기초 자산에 공통적으로 사용되는 가치 평가 환경 추가
            self.val_env.add_list("cholesky_matrix", cholesky_matrix)
            self.val_env.add_list("random_numbers", random_numbers)
            self.val_env.add_list("rn_set", rn_set)

        for asset in self.underlying:
            # 자산의 시장 환경 선택
            mar_env = self.assets[asset]
            # 시장 환경에 가치 평가 환경 추가
            mar_env.add_environment(val_env)
            # 적절한 시뮬레이션 클래스 선택
            model = models[mar_env.constant["model"]]
            # 시뮬레이션 객체 초기화
            if correlations is not None:
                self.underlying_objects[asset] = model(asset, mar_env, corr=True)
            else:
                self.underlying_objects[asset] = model(asset, mar_env, corr=False)

        for pos in positions:
            # 적절한 가치 평가 클래스 선택
            val_class = otypes[positions[pos].otype]
            # 적절한 시장 환경을 선택하여 가치 평가 환경 추가
            mar_env = positions[pos].mar_env
            mar_env.add_environment(self.val_env)
            # 가치 평가 클래스 초기화
            self.valuation_objects[pos] = \
                val_class(name=positions[pos].name,
                          mar_env=mar_env,
                          underlying=self.underlying_objects[positions[pos].underlying],
                          payoff_func=positions[pos].payoff_func)

    def get_positions(self):
        """포트폴리오의 모든 파생상품 포지션 정보를 얻어내는 메서드"""

        for pos in self.positions:
            bar = '\n' + 50 * '-'
            print(bar)
            self.positions[pos].get_info()
            print(bar)

    def get_statistics(self, fixed_seed=False):
        """포트폴리오 통계 제공"""
        res_list = []
        # 포트폴리오 내의 모든 포지션에 대해 반복
        for pos, value in self.valuation_objects.items():
            p = self.positions[pos]
            pv = value.present_value(fixed_seed=fixed_seed)
            res_list.append([p.name, p.quantity,
                             # 단일 상품의 모든 현재 가치 계산
                             pv,
                             value.currency,
                             # 현재 가치와 수량의 곲
                             pv * p.quantity,
                             # 포지션 델타 계산
                             value.delta() * p.quantity,
                             # 포지션 베가 계산
                             value.vega() * p.quantity, ])
        # 모든 결과 값을 가진 pandas DataFrame 생성
        res_df = pd.DataFrame(res_list,
                              columns=["name", "quant", "value", "curr", "pos_value",
                                       "pos_delta", "pos_vega"])
        return res_df
