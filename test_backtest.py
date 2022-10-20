import unittest


class MyTestCase(unittest.TestCase):
    def test_back1(self):
        from backtest import SMAVectorBacktester
        smabt = SMAVectorBacktester("EUR=", 42, 252,
                                    "2010-1-1", "2020-12-31")
        print(smabt.run_strategy())
        smabt.set_parameter(SMA1=20, SMA2=100)
        print(smabt.run_strategy())
        print(smabt.optimize_parameters((30, 56, 4), (200, 300, 4)))

    def test_back2(self):
        from momentum_backtest import MomVectorBacktester
        mombt = MomVectorBacktester("XAU=", "2010-1-1", "2020-12-31",
                                    10000, 0.0)

        print(mombt.run_strategy())
        print(mombt.run_strategy(2))
        mombt = MomVectorBacktester("XAU=", "2010-1-1", "2020-12-31",
                                    10000, 0.001)
        print(mombt.run_strategy(2))

    def test_back3(self):
        from momentum_backtest import MRVectorBacktester
        mombt = MRVectorBacktester("GDX", "2010-1-1", "2020-12-31",
                                   10000, 0.0)

        print(mombt.run_strategy(SMA=25, threshold=5))
        mombt = MRVectorBacktester("GDX", "2010-1-1", "2020-12-31",
                                   10000, 0.001)

        print(mombt.run_strategy(SMA=25, threshold=5))
        mombt = MRVectorBacktester("GDX", "2010-1-1", "2020-12-31",
                                   10000, 0.001)

        print(mombt.run_strategy(SMA=42, threshold=7.5))

    def test_back4(self):
        from LRVectorBacktester import LRVectorBacktester
        lrbt = LRVectorBacktester(".SPX", '2010-1-1', '2018-06-29', 10000, 0.0)
        print(lrbt.run_strategy("2010-1-1", '2019-12-31', '2010-1-1', '2019-12-31'))
        print(lrbt.run_strategy("2010-1-1", '2015-12-31', '2016-1-1', '2019-12-31'))

        lrbt = LRVectorBacktester(".SPX", '2010-1-1', '2018-06-29', 10000, 0.001)
        print(lrbt.run_strategy("2010-1-1", '2019-12-31', '2010-1-1', '2019-12-31', lags=5))
        print(lrbt.run_strategy("2010-1-1", '2016-12-31', '2017-1-1', '2019-12-31', lags=5))

    def test_back5(self):
        from ScikitVectorBacktester import ScikitVectorBacktester

        lrbt = ScikitVectorBacktester(".SPX", '2010-1-1', '2018-06-29', 10000, 0.0, 'regression')
        print(lrbt.run_strategy("2010-1-1", '2019-12-31', '2010-1-1', '2019-12-31'))
        print(lrbt.run_strategy("2010-1-1", '2015-12-31', '2016-1-1', '2019-12-31'))

        lrbt = ScikitVectorBacktester(".SPX", '2010-1-1', '2019-12-31', 10000, 0.0, 'logistic')
        print(lrbt.run_strategy("2010-1-1", '2019-12-31', '2010-1-1', '2019-12-31'))
        print(lrbt.run_strategy("2010-1-1", '2016-12-31', '2017-1-1', '2019-12-31'))

        lrbt = ScikitVectorBacktester(".SPX", '2010-1-1', '2019-12-31', 10000, 0.001, 'logistic')
        print(lrbt.run_strategy("2010-1-1", '2019-12-31', '2010-1-1', '2019-12-31', lags=15))
        print(lrbt.run_strategy("2010-1-1", '2016-12-31', '2017-1-1', '2019-12-31', lags=15))

    def test_back6(self):
        from BacktestBase import BacktestBase
        bb = BacktestBase("AAPL.O", "2010-1-1", "2019-12-31", 10000)
        print(bb.data.info())
        print(bb.data.tail())
        bb.plot_data()

    def test_back7(self):
        from BacktestBase import BacktestLongOnly
        lobt = BacktestLongOnly("AAPL.O", "2010-1-1", "2019-12-31", 10000, verbose=False)

        def run_strategies():
            lobt.run_sma_strategy(42, 252)
            lobt.run_momentum_strategy(60)
            lobt.run_mean_reversion_strategy(50, 5)

        run_strategies()
        # 거래비용 10 USD 고정, 1% 변동
        lobt = BacktestLongOnly("AAPL.O", "2010-1-1", "2019-12-31", 10000, 10.0, 0.01, verbose=False)
        run_strategies()

    def test_back8(self):
        from BacktestBase import BacktestLongShort
        lobt = BacktestLongShort("AAPL.O", "2010-1-1", "2019-12-31", 10000, verbose=False)

        def run_strategies():
            lobt.run_sma_strategy(42, 252)
            lobt.run_momentum_strategy(60)
            lobt.run_mean_reversion_strategy(50, 5)

        run_strategies()
        # 거래비용 10 USD 고정, 1% 변동
        lobt = BacktestLongShort("AAPL.O", "2010-1-1", "2019-12-31", 10000, 10.0, 0.01, verbose=False)
        run_strategies()


if __name__ == '__main__':
    unittest.main()
