# Event Based Backtesting
# --Base class (1)

class BacktestBase:
    def __init__(self, env, model, amount, ftc, ptc, verbose=False):
        self.env = env
        self.model = model
        self.initial_amount = amount
        self.current_balance = amount
        self.ftc = ftc
        self.ptc = ptc
        self.verbose = verbose
        self.units = 0  # 초기 포트폴리오의 수단으로 삼은것의 단위 (주식수)
        self.trades = 0

    def get_date_price(self, bar):
        """Returns date and price for a given bar."""
        date = str(self.env.data.index[bar])[:10]
        price = self.env.data[self.env.symbol].iloc[bar]
        return date, price

    def print_balance(self, bar):
        """Prints the current cash balance for a given bar."""
        date, price = self.get_date_price(bar)
        print(f"{date} | current balance {self.current_balance:.2f}")

    def calculate_net_wealth(self, price):
        return self.current_balance + self.units * price

    def print_net_wealth(self, bar):
        """Prints the net wealth for a given bar (cash + position)"""
        date, price = self.get_date_price(bar)
        net_wealth = self.calculate_net_wealth(price)
        print(f"{date} | current net wealth {net_wealth:.2f}")

    def place_buy_order(self, bar, amount=None, units=None):
        """Places a buy order for a given bar and for a given amount or
        number of units."""
        date, price = self.get_date_price(bar)
        if units is None:
            # units = amount / price # alternative handling
            units = int(amount / price)
        self.current_balance -= (1 + self.ptc) * units * price + self.ftc
        self.units += units
        self.trades += 1
        if self.verbose:
            print(f"{date} | buy {units} units at {price:.4f}")
            self.print_balance(bar)
            self.print_net_wealth(bar)

    def place_sell_order(self, bar, amount=None, units=None):
        """Places a sell order for a given bar and for a given amount or
              number of units."""
        date, price = self.get_date_price(bar)
        if units is None:
            units = int(amount / price)
        self.current_balance += (1 - self.ptc) * units * price - self.ftc
        self.units -= units
        self.trades += 1
        if self.verbose:
            print(f"{date} | sell {units} units at {price:.2f}")
            self.print_balance(bar)
            self.print_net_wealth(bar)

    def close_out(self, bar):
        """Closes out any open position at a given bar."""
        date, price = self.get_date_price(bar)
        print("=" * 50)
        print(f'{date} | *** CLOSING OUT ***')
        if self.units < 0:
            self.place_buy_order(bar, units=-self.units)
        else:
            self.place_sell_order(bar, units=self.units)

        self.current_balance += self.units * price
        self.units = 0
        self.trades += 1
        if not self.verbose:
            print(f"{date} | current balance = {self.current_balance:.2f}")

        perf = (self.current_balance / self.initial_amount - 1) * 100
        print(f"{date} | Net Performance [%] {perf:.4f}")
        print(f"{date} Number of trades [#] {self.trades}")
        print("=" * 50)


# Event Based Backtesting
# --Base class (2)
#

# 리스크 과리 백테스팅
class BacktestBaseRM(BacktestBase):

    def set_price(self, price):
        """Sets prices for tracking of performancce.
        To test for e.g. trailing stop loss hit.
        """
        self.entry_price = price  # 진입 가격 설정
        self.max_price = price  # 최대 가격 설정
        self.min_price = price  # 최소 가격 설정

    def place_buy_order(self, bar, amount=None, units=None, gprice=None):
        """places a buy order for a given bar and for a given amount or number of units"""
        date, price = self.get_date_price(bar)
        if gprice is not None:
            price = gprice
        if units is None:
            units = int(amount / price)

        self.current_balance -= (1 - self.ptc) * units * price + self.ftc
        self.units += units
        self.trades += 1
        self.set_price(price)  # 매매가 실행된 후 관련 가격 설정
        if self.verbose:
            print(f"{date} | buy {units} units for {price:.4f}")
            self.print_balance(bar)

    def place_sell_order(self, bar, amount=None, units=None, gprice=None):
        """places a sell order for a given bar and for a given amount or number of units"""
        date, price = self.get_date_price(bar)
        if gprice is not None:
            price = gprice
        if units is None:
            units = int(amount / price)
        self.current_balance += (1 - self.ptc) * units * price - self.ftc
        self.units -= units
        self.trades += 1
        self.set_price(price)  # 매매가 실행된 후 관련 가격 설정
        if self.verbose:
            print(f"{date} | sell {units} units for {price:.4f}")
            self.print_balance(bar)
