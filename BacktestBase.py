# 이벤트 기반 백테스트를 위한 기저 클래스
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn")
plt.rcParams["font.family"] = 'serif'


class BacktestBase(object):
    """
    속성
       symbol: str
           작업에 쓸 TR RIC(금융 수단)
       start: str
           데이터 선택한 시작 부분에 해당하는 날짜
       end: str
           데이터 선택한 끝 부분에 해당하는 날짜
       amount: float
            한 번 투자하거나 거래당 투자할 금액
       ftc: float
           거래당 고정 거래 비용 
       ptc: float 
            거래당 비례 거래비용

       =======
       get_data:
           기본 데이터 집합을 검색해 준비해 둔다
       plot_data:
            종목 코드에 대한 종가를 그린다.
       get_data_price:
            주어진 봉에 대한 일자와 가격을 반환한다.
       print_balance:
            현재 잔고를 프린트한다.
       print_net_wealth:
            현재 순 자산을 프린트 한다.
       place_buy_order:
            매수 주문을 넣는다.
       place_sell_order:
            매도 주문을 넣는다.
       close_out:
            롱 포지셔이나 숏 포지션을 닫는다.
    """

    def __init__(self, symbol, start, end, amount, ftc=0., ptc=0.0, verbose=True):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.initial_amount = amount
        self.amount = amount
        self.ftc = ftc
        self.ptc = ptc
        self.units = 0  # 초기 포트폴리오의 수단으로 삼은것의 단위 (주식수)
        self.position = 0
        self.trades = 0
        self.verbose = verbose
        self.get_data()

    def get_data(self):
        raw = pd.read_csv("http://hilpisch.com/pyalgo_eikon_eod_data.csv",
                          index_col=0, parse_dates=True).dropna()
        raw = pd.DataFrame(raw[self.symbol])
        raw = raw.loc[self.start: self.end]
        raw.rename(columns={self.symbol: 'price'}, inplace=True)
        raw["return"] = np.log(raw / raw.shift(1))
        self.data = raw.dropna()

    def plot_data(self, cols=None):
        if cols is None:
            cols = ["price"]
        self.data["price"].plot(figsize=(10, 6), title=self.symbol)

    def get_data_price(self, bar):
        date = str(self.data.index[bar])[:10]
        price = self.data.price.iloc[bar]
        return date, price

    def print_balance(self, bar):
        date, price = self.get_data_price(bar)
        print(f"{date} | current balance {self.amount:.2f}")

    def print_net_wealth(self, bar):
        date, price = self.get_data_price(bar)
        net_wealth = self.units * price + self.amount
        print(f"{date} | current net wealth {net_wealth:.2f}")

    def place_buy_order(self, bar, units=None, amount=None):
        date, price = self.get_data_price(bar)
        if units is None:
            units = int(amount / price)
        self.amount -= (units * price) * (1 + self.ptc) + self.ftc
        self.units += units
        self.trades += 1
        if self.verbose:
            print(f"{date} | selling {units} units at {price:.2f}")
            self.print_balance(bar)
            self.print_net_wealth(bar)

    def place_sell_order(self, bar, units=None, amount=None):
        date, price = self.get_data_price(bar)
        if units is None:
            units = int(amount / price)
        self.amount += (units * price) * (1 - self.ptc) - self.ftc
        self.units -= units
        self.trades += 1
        if self.verbose:
            print(f"{date} | selling {units} units at {price:.2f}")
            self.print_balance(bar)
            self.print_net_wealth(bar)

    def close_out(self, bar):
        date, price = self.get_data_price(bar)
        self.amount += self.units * price
        self.units = 0
        self.trades += 1
        if self.verbose:
            print(f"{date} | inventory {self.units} units at {price:.2f}")
            print("=" * 55)
        print("Final balance [$] {:.2f}".format(self.amount))
        perf = ((self.amount - self.initial_amount) / self.initial_amount * 100)
        print("Net Performance [%] {:.2f}".format(perf))
        print("Trades Executed [#] {:.2f}".format(self.trades))
        print("=" * 55)


# 이벤트 기반 백테스트를 위한 롱 전용 클래스
class BacktestLongOnly(BacktestBase):

    def run_sma_strategy(self, SMA1, SMA2):
        """
        SMA 기반 전략 백테스트
        :param SMA1:
        :param SMA2: int
        롱과 숏의 더 긴 기간에 대한 단순 이동평균 (일)
        :return:
        """
        msg = f'\n\nRunning SMA strategy | SMA1={SMA1} & SMA2={SMA2}'
        msg += f'\nfixed costs {self.ftc} | '
        msg += f'proportional costs {self.ptc}'
        print(msg)
        print("=" * 55)
        self.position = 0  # 초기 뉴트럴 포지션
        self.trades = 0  # 아직 거래 없음
        self.amount = self.initial_amount  # 초기 금액을 재설정
        self.data["SMA1"] = self.data["price"].rolling(SMA1).mean()
        self.data["SMA2"] = self.data["price"].rolling(SMA2).mean()

        for bar in range(SMA2, len(self.data)):
            if self.position == 0:
                if self.data["SMA1"].iloc[bar] > self.data["SMA2"].iloc[bar]:
                    self.place_buy_order(bar, amount=self.amount)
                    self.position = 1  # 롱 포지션
            elif self.position == 1:
                if self.data["SMA1"].iloc[bar] < self.data["SMA2"].iloc[bar]:
                    self.place_sell_order(bar, amount=self.amount)
                    self.position = 0  # 중립 포지션
            self.close_out(bar)

    def run_momentum_strategy(self, momentum):
        """
        모멘텀 기반 전략 백테스트
        :param momentum: int
        평균 수익 계산 일수
        :return:
        """
        msg = f'\n\nRunning momentum strategy | {momentum} days'
        msg += f'\nfixed costs {self.ftc} | '
        msg += f'proportional costs {self.ptc}'
        print(msg)
        print("=" * 55)
        self.position = 0  # 초기 뉴트럴 포지션
        self.trades = 0  # 아직 거래 없음
        self.amount = self.initial_amount  # 초기 금액을 재설정
        self.data["momentum"] = self.data["return"].rolling(momentum).mean()
        for bar in range(momentum, len(self.data)):
            if self.position == 0:
                if self.data['momentum'].iloc[bar] > 0:
                    self.place_buy_order(bar, amount=self.amount)
                    self.position = 1
            elif self.position == 1:
                if self.data['momentum'].iloc[bar] < 0:
                    self.place_sell_order(bar, amount=self.amount)
                    self.position = 0
            self.close_out(bar)

    def run_mean_reversion_strategy(self, SMA, threshold):
        """
        평균 회귀 기반 전략 백테스트
        :param SMA: int
            일 단위 단순 이동 평균
        :param threshold: float
            SMA에 대한 편차 기반 신호의 절대 값
        :return:
        """
        msg = f'\n\nRunning mean reversion strategy | SMA={SMA} & thr={threshold}'
        msg += f'\nfixed costs {self.ftc} | '
        msg += f'proportional costs {self.ptc}'
        print(msg)
        print("=" * 55)
        self.position = 0  # 초기 뉴트럴 포지션
        self.trades = 0  # 아직 거래 없음
        self.amount = self.initial_amount  # 초기 금액을 재설정
        self.data["SMA"] = self.data["price"].rolling(SMA).mean()

        for bar in range(SMA, len(self.data)):
            if self.position == 0:
                if self.data['price'].iloc[bar] < self.data["SMA"].iloc[bar] - threshold:
                    self.place_buy_order(bar, amount=self.amount)
                    self.position = 1
            elif self.position == 1:
                if self.data['price'].iloc[bar] >= self.data["SMA"].iloc[bar]:
                    self.place_sell_order(bar, amount=self.amount)
                    self.position = 0
            self.close_out(bar)


class BacktestLongShort(BacktestBase):

    def go_long(self, bar, units=None, amount=None):
        if self.position == 1:
            self.place_buy_order(bar, units=-self.units)
        if units:
            self.place_buy_order(bar, units=units)
        elif amount:
            if amount == 'all':
                amount = self.amount
            self.place_buy_order(bar, amount=amount)

    def go_short(self, bar, units=None, amount=None):
        if self.position == 1:
            self.place_sell_order(bar, units=-self.units)
        if units:
            self.place_sell_order(bar, units=units)
        elif amount:
            if amount == 'all':
                amount = self.amount
            self.place_sell_order(bar, amount=amount)

    def run_sma_strategy(self, SMA1, SMA2):
        """
        SMA 기반 전략 백테스트
        :param SMA1:
        :param SMA2: int
        롱과 숏의 더 긴 기간에 대한 단순 이동평균 (일)
        :return:
        """
        msg = f'\n\nRunning SMA strategy | SMA1={SMA1} & SMA2={SMA2}'
        msg += f'\nfixed costs {self.ftc} | '
        msg += f'proportional costs {self.ptc}'
        print(msg)
        print("=" * 55)
        self.position = 0  # 초기 뉴트럴 포지션
        self.trades = 0  # 아직 거래 없음
        self.amount = self.initial_amount  # 초기 금액을 재설정
        self.data["SMA1"] = self.data["price"].rolling(SMA1).mean()
        self.data["SMA2"] = self.data["price"].rolling(SMA2).mean()

        for bar in range(SMA2, len(self.data)):
            if self.position in [0, -1]:
                if self.data["SMA1"].iloc[bar] > self.data["SMA2"].iloc[bar]:
                    self.go_long(bar, amount='all')
                    self.position = 1  # 롱 포지션

            elif self.position in [0, 1]:
                if self.data["SMA1"].iloc[bar] < self.data["SMA2"].iloc[bar]:
                    self.go_short(bar, amount='all')
                    self.position = -1  # 숏 포지션
            self.close_out(bar)

    def run_momentum_strategy(self, momentum):
        """
        모멘텀 기반 전략 백테스트
        :param momentum: int
        평균 수익 계산 일수
        :return:
        """
        msg = f'\n\nRunning momentum strategy | {momentum} days'
        msg += f'\nfixed costs {self.ftc} | '
        msg += f'proportional costs {self.ptc}'
        print(msg)
        print("=" * 55)

        self.position = 0  # 초기 뉴트럴 포지션
        self.trades = 0  # 아직 거래 없음
        self.amount = self.initial_amount  # 초기 금액을 재설정
        self.data["momentum"] = self.data["return"].rolling(momentum).mean()

        for bar in range(momentum, len(self.data)):
            if self.position in [0, -1]:
                if self.data['momentum'].iloc[bar] > 0:
                    self.go_long(bar, amount="all")
                    self.position = 1

            elif self.position in [0, 1]:
                if self.data['momentum'].iloc[bar] <= 0:
                    self.go_short(bar, amount='all')
                    self.position = -1
            self.close_out(bar)

    def run_mean_reversion_strategy(self, SMA, threshold):
        """
        평균 회귀 기반 전략 백테스트
        :param SMA: int
            일 단위 단순 이동 평균
        :param threshold: float
            SMA에 대한 편차 기반 신호의 절대 값
        :return:
        """
        msg = f'\n\nRunning mean reversion strategy | SMA={SMA} & thr={threshold}'
        msg += f'\nfixed costs {self.ftc} | '
        msg += f'proportional costs {self.ptc}'
        print(msg)
        print("=" * 55)
        self.position = 0  # 초기 뉴트럴 포지션
        self.trades = 0  # 아직 거래 없음
        self.amount = self.initial_amount  # 초기 금액을 재설정

        self.data["SMA"] = self.data["price"].rolling(SMA).mean()

        for bar in range(SMA, len(self.data)):
            if self.position == 0:
                if self.data['price'].iloc[bar] < self.data["SMA"].iloc[bar] - threshold:
                    self.go_long(bar, amount=self.initial_amount)
                    self.position = 1

                elif self.data["price"].iloc[bar] > self.data["SMA"].iloc[bar] + threshold:
                    self.go_short(bar, amount=self.initial_amount)
                    self.position = -1

            elif self.position == 1:
                if self.data['price'].iloc[bar] >= self.data["SMA"].iloc[bar]:
                    self.place_sell_order(bar, units=self.units)
                    self.position = 0

            elif self.position == -1:
                if self.data["price"].iloc[bar] <= self.data["SMA"].iloc[bar]:
                    self.place_buy_order(bar, units=-self.units)
                    self.position = 0

            self.close_out(bar)
