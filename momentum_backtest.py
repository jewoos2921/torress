# 모멘텀 기반 전략들을 대상으로 벡터화 백테스트를 하는 데 쓸 클래스
import pandas as pd
import numpy as np


class MomVectorBacktester(object):
    """
    모멘텀 기반 전략들을 대상으로 벡터화 백테스트를 하는 데 쓸 클래스

    속성
    symbol: str
        작업에 쓸 RIC(금융 수단)
    start: str
        데이터 선택한 시작 부분에 해당하는 날짜
    end: str
        데이터 선택한 끝 부분에 해당하는 날짜
    amount: int, float
        처음에 투자할 금액
    tc: float
        거래당 비례 거래비용 (예: 0.5% = 0.005)

    메서드
    =======
    get_data:
        기본 데이터 집합을 검색해 준비해 둔다
    run_strategy:
        모멘텀기반 전략에 대한 백테스트를 실행
    plot_results:
        종목코드와 비교되는 전략의 성과를 그려낸다.
    """

    def __init__(self, symbol, start, end, amount, tc):
        self.symbol = symbol
        self.amount = amount
        self.tc = tc
        self.start = start
        self.end = end
        self.results = None
        self.get_data()

    def get_data(self):
        raw = pd.read_csv("http://hilpisch.com/pyalgo_eikon_eod_data.csv",
                          index_col=0, parse_dates=True).dropna()
        raw = pd.DataFrame(raw[self.symbol])
        raw = raw.loc[self.start: self.end]
        raw.rename(columns={self.symbol: 'price'}, inplace=True)
        raw["return"] = np.log(raw / raw.shift(1))
        self.data = raw

    def run_strategy(self, momentum=1):
        self.momentum = momentum
        data = self.data.copy().dropna()
        data["position"] = np.sign(data["return"].rolling(momentum).mean())
        data["strategy"] = data["position"].shift(1) * data['return']
        # 거래 성사 시기를 결정한다.
        data.dropna(inplace=True)
        trades = data["position"].diff().fillna(0) != 0
        # 거래가 성사시 수익에서 거래 비용을 뺀다.
        data["strategy"][trades] -= self.tc
        data["creturns"] = self.amount * data["return"].cumsum().apply(np.exp)
        data["cstrategy"] = self.amount * data['strategy'].cumsum().apply(np.exp)
        self.results = data
        # 전략의 절대 성과
        aperf = data["cstrategy"].iloc[-1]
        # 전략의 초과성과/미달성과
        operf = aperf - data['creturns'].iloc[-1]
        return round(aperf, 2), round(operf, 2),

    def plot_results(self):
        if self.results is None:
            print("No results to plot yet. Run a strategy.")
        title = '%s | TC=%.4f' % (self.symbol,
                                  self.tc)
        self.results[['creturns', "cstrategy"]].plot(title=title,
                                                     figsize=(10, 6))


# 평균 회귀 전략들을 대상으로 벡터화 백테스트를 실행
class MRVectorBacktester(MomVectorBacktester):
    """
    평균 회귀 기반 거래 전략
     속성
    symbol: str
        작업에 쓸 RIC(금융 수단)
    start: str
        데이터 선택한 시작 부분에 해당하는 날짜
    end: str
        데이터 선택한 끝 부분에 해당하는 날짜
    amount: int, float
        처음에 투자할 금액
    tc: float
        거래당 비례 거래비용 (예: 0.5% = 0.005)

    메서드
    =======
    get_data:
        기본 데이터 집합을 검색해 준비해 둔다
    run_strategy:
        평균 회귀 기반 전략에 대한 백테스트를 실행
    plot_results:
        종목코드와 비교되는 전략의 성과를 그려낸다.
    """

    def run_strategy(self, SMA, threshold):
        data = self.data.copy().dropna()
        data["sma"] = data["price"].rolling(SMA).mean()
        data["distance"] = data["price"] - data["sma"]
        data.dropna(inplace=True)

        # 매수 신호들
        data["position"] = np.where(data['distance'] > threshold, -1, np.nan)

        # 매도 신호들
        data["position"] = np.where(data['distance'] < -threshold, 1, data["position"])

        # 현재 가격과 SMA의 교차 (즉, 거리가 0임)
        data["position"] = np.where(data["distance"] * data['distance'].shift(1) < 0, 0, data["position"])
        data["position"] = data["position"].ffill().fillna(0)
        data["strategy"] = data["position"].shift(1) * data["return"]

        # 거래 성사 시기를 결정
        trades = data["position"].diff().fillna(0) != 0

        # 거래가 성사 되었을 때 수익에서 거래 비용을 뺀다.
        data["strategy"][trades] -= self.tc
        data["creturns"] = self.amount * data["return"].cumsum().apply(np.exp)
        data["cstrategy"] = self.amount * data['strategy'].cumsum().apply(np.exp)
        self.results = data
        # 전략의 절대 성과
        aperf = data["cstrategy"].iloc[-1]
        # 전략의 초과성과/미달성과
        operf = aperf - data['creturns'].iloc[-1]
        return round(aperf, 2), round(operf, 2),
