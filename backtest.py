# SMA 기반 전략들을 대상으로 벡터화 백테스트를 하는 데 쓸 클래스
import pandas as pd
import numpy as np
from scipy.optimize import brute


class SMAVectorBacktester(object):
    """
    symbol: str
        SMA1 작업용 RIC 종목 코드
    SMA1: int
        상대적 단기 SMA를 위한 일별 시간 창
    SMA2: int
        상대적 장기 SMA를 위한 일별 시간 창
    start: str
        데이터 검샏 대상 중 시작 부분에 해당하는 날짜
    end: str
        데이터 검샏 대상 중 끝 부분에 해당하는 날짜


    # 메서드
    =========
    get_data:
        기본 데이터셋을 검색해 준비한다.
    set_parameter:
        한 두개의 SMA 파라미터들을 구성한다.
    run_strategy:
        SMA 기반 전략에 대한 벡테스트를 실행한다.
    plot_results:
        종목코드와 비교한 전략의 성과를 그려 낸다.
    update_and_run:
        SMA 파라미터들을 갱신하고 (부정적인) 절대 성과를 반환한다.
    optimize_parameters:
        두 가지 SMA 파라미터들에 대한 전수 대입(brute-force) 최적화를 구현한다.
    """

    def __init__(self, symbol, SMA1, SMA2, start, end):
        self.symbol = symbol
        self.SMA1 = SMA1
        self.SMA2 = SMA2
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
        raw["SMA1"] = raw['price'].rolling(self.SMA1).mean()
        raw["SMA2"] = raw['price'].rolling(self.SMA2).mean()
        self.data = raw

    def set_parameter(self, SMA1=None, SMA2=None):
        if SMA1 is not None:
            self.SMA1 = SMA1
            self.data["SMA1"] = self.data["price"].rolling(self.SMA1).mean()
        if SMA2 is not None:
            self.SMA2 = SMA2
            self.data["SMA2"] = self.data["price"].rolling(self.SMA2).mean()

    def run_strategy(self):
        data = self.data.copy().dropna()
        data["position"] = np.where(data["SMA1"] > data["SMA2"], 1, -1)
        data["strategy"] = data["position"].shift(1) * data["return"]
        data.dropna(inplace=True)
        data["creturns"] = data["return"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        # 전략의 총 성과
        aperf = data["cstrategy"].iloc[-1]
        # 전략의 초과성과/미달성과
        operf = aperf - data['creturns'].iloc[-1]
        return round(aperf, 2), round(operf, 2),

    def plot_results(self):
        if self.results is None:
            print("No results to plot yet. Run a strategy.")
        title = '%s | SMA1=%d, SMA2=%d' % (self.symbol,
                                           self.SMA1,
                                           self.SMA2)
        self.results[['creturns', "cstrategy"]].plot(title=title,
                                                     figsize=(10, 6))

    def update_and_run(self, SMA):
        """
        
        :param SMA: tuple
            SMA 파라미터 튜플
        :return: 
        """
        self.set_parameter(int(SMA[0]), int(SMA[1]))
        return -self.run_strategy()[0]

    def optimize_parameters(self, SMA1_range, SMA2_range):
        """
        주어진 SMA 파라미터 범위 내의 전역 최대 (global maximum, 최댓값)를 찾는다.
        :param SMA1_range: tuple
        :param SMA2_range: tuple
        (시작, 종료, 단계 크기) 형식으로 된 튜플
        :return:
        """
        opt = brute(self.update_and_run, (SMA1_range, SMA2_range), finish=None)
        return opt, -self.update_and_run(opt),
