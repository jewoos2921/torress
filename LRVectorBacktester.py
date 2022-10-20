# 선형회귀 기반 전력의 벡터화 백테스트를 위한 클래스
import numpy as np
import pandas as pd


class LRVectorBacktester(object):
    """
       선형 회귀 기반 거래 전략
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
       select_data:
            데이터의 부분집합 한 개를 선택한다.
       prepare_lags:
            회귀용으로 시차 처리한 데이터를 준비한다.
       fit_model:
            회귀 단계를 구현한다.
       run_strategy:
           회귀 기반 전략에 대한 백테스트를 실행
       plot_results:
           종목코드와 비교되는 전략의 성과를 그려낸다.
       """

    def __init__(self, symbol, start, end, amount, tc):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.amount = amount
        self.tc = tc
        self.results = None
        self.get_data()

    def get_data(self):
        raw = pd.read_csv("http://hilpisch.com/pyalgo_eikon_eod_data.csv",
                          index_col=0, parse_dates=True).dropna()
        raw = pd.DataFrame(raw[self.symbol])
        raw = raw.loc[self.start: self.end]
        raw.rename(columns={self.symbol: 'price'}, inplace=True)
        raw["return"] = np.log(raw / raw.shift(1))
        self.data = raw.dropna()

    def select_data(self, start, end):
        data = self.data[(self.data.index >= start) & (self.data.index <= end)].copy()
        return data

    def prepare_lags(self, start, end):
        """회귀 및 예측 단계에 대해 시차 처리한 데이터를 준비한다."""
        data = self.select_data(start, end)
        self.cols = []
        for lag in range(1, self.lags + 1):
            col = f'lag_{lag}'
            data[col] = data["return"].shift(lag)
            self.cols.append(col)
        data.dropna(inplace=True)
        self.lagged_data = data

    def fit_model(self, start, end):
        self.prepare_lags(start, end)
        reg = np.linalg.lstsq(self.lagged_data[self.cols],
                              np.sign(self.lagged_data["return"]), rcond=None)[0]
        self.reg = reg

    def run_strategy(self, start_in, end_in, start_out, end_out, lags=3):
        self.lags = lags
        self.fit_model(start_in, end_in)
        self.results = self.select_data(start_out, end_out).iloc[lags:]

        self.prepare_lags(start_out, end_out)

        prediction = np.sign(np.dot(self.lagged_data[self.cols], self.reg))
        self.results["prediction"] = prediction
        self.results["strategy"] = self.results["prediction"] * self.results["return"]

        # 거래 성사 시기를 결정한다.

        trades = self.results["prediction"].diff().fillna(0) != 0
        # 거래가 성사시 수익에서 거래 비용을 뺀다.
        self.results["strategy"][trades] -= self.tc
        self.results["creturns"] = self.amount * self.results["return"].cumsum().apply(np.exp)
        self.results["cstrategy"] = self.amount * self.results['strategy'].cumsum().apply(np.exp)


        # 전략의 절대 성과
        aperf = self.results["cstrategy"].iloc[-1]
        # 전략의 초과성과/미달성과
        operf = aperf - self.results['creturns'].iloc[-1]
        return round(aperf, 2), round(operf, 2),

    def plot_results(self):
        if self.results is None:
            print("No results to plot yet. Run a strategy.")
        title = '%s | TC=%.4f' % (self.symbol,
                                  self.tc)
        self.results[['creturns', "cstrategy"]].plot(title=title,
                                                     figsize=(10, 6))
