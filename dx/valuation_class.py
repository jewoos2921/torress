# 가치평가 - 베이스 클래스

class valuation_class(object):
    """
    단일 인수 가치 평가를 위한 기반 클래스
    name: str
        객체 이름
    underlying: instance of simulation class
        단일 위험 요인을 모델링 하는 객체
    mar_env: instance of market_environment
        가치 평가용 시장 환경 데이터
    payoff_func: str
        파이썬 문법으로 된 파생상품 페이오프
        예: np.maximum(maturity_value - 100, 0)
        maturity_value는 기초 자산의 만기 값을 나타내는 NUMPY 벡터
        예: np.maximum(instrument_values - 100, 0)
        instrument_values는 기초 자산의 전체 시간/경로 그리드 값을 나타내는 NUMPY 행렬
        
    메서드 
    update:
        선택된 가치 평가 인수를 갱신
    delta: 
        파생상품의 델타를 반환
    vega: 
        파생상품의 베가를 반환
    """

    def __init__(self, name, underlying, mar_env, payoff_func=""):
        self.name = name
        self.pricing_date = mar_env.pricing_date
        try:
            # 행사가는 반드시 필요하지는 않다.
            self.strike = mar_env.get_constant("strike")
        except:
            pass
        self.maturity = mar_env.get_constant("maturity")
        self.currency = mar_env.get_constant("currency")

        # 시뮬레이션 객체의 할인 커브와 시뮬레이션 인수
        self.frequency = underlying.frequency
        self.paths = underlying.paths
        self.discount_curve = underlying.discount_curve
        self.payoff_func = payoff_func
        self.underlying = underlying
        # 기초 자산의 만기와 가치 평가일 제공
        self.underlying.special_dates.extend([self.pricing_date,
                                              self.maturity])

    def update(self, initial_value=None, volatility=None,
               strike=None, maturity=None):
        if initial_value is not None:
            self.underlying.update(initial_value=initial_value)
        if volatility is not None:
            self.underlying.update(volatility=volatility)
        if strike is not None:
            self.strike = strike
        if maturity is not None:
            self.maturity = maturity
            # 새로운 만기일 추가
            if maturity not in self.underlying.time_grid:
                self.underlying.special_dates.append(maturity)
                self.underlying.instrument_values = None

    def delta(self, interval=None, accuracy=4):
        if interval is None:
            interval = self.underlying.initial_value / 50.
        # 전향 차분 근사화
        # 수치적인 델타의 좌측값 계산
        value_left = self.present_value(fixed_seed=True)
        # 기초 자산의 우측값 계산
        initial_del = self.underlying.initial_value + interval
        self.underlying.update(initial_value=initial_del)
        # 수치적인 델타의 우측값 계산
        value_right = self.present_value(fixed_seed=True)
        # 시뮬레이션 객체의 초깃값 리셋
        self.underlying.update(initial_value=initial_del - interval)
        delta = (value_right - value_left) / interval
        # 수치 오류 정정
        if delta < -1.0:
            return -1.0
        elif delta > 1.0:
            return 1.0
        else:
            return round(delta, accuracy)

    def vega(self, interval=0.01, accuracy=4):
        if interval < self.underlying.volatility / 50.:
            interval = self.underlying.volatility / 50.
        # 전향 차분 베타화
        # 수치적인 베가의 좌측값 계산
        value_left = self.present_value(fixed_seed=True)
        # 변동성의 우측값 계산
        initial_del = self.underlying.initial_value + interval
        self.underlying.update(initial_value=initial_del)
        # 수치적인 베가의 우측값 계산
        value_right = self.present_value(fixed_seed=True)
        # 시뮬레이션 객체의 변동성값 리셋
        self.underlying.update(initial_value=initial_del - interval)
        vega = (value_right - value_left) / interval

        return round(vega, accuracy)
