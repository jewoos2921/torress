# 포트폴리오 - 파생상품 포지션 클래스

class derivatives_position(object):
    """
    파생 상품 포지션 모델링

    속성
    =========
    name: str
        객체 이름
    quantity: float
        포지션을 구성하는 자산/파생상품의 개수
    underlying: str
        파생상품의 자산/위험 요인 이름
    mar_env: instance of market_environment
        valuation_class의 상수, 리스트, 커브 데이터
    otype: str
        사용할 가치 평가 클래스
    payoff_func: str
        파생상품의 페이오프 문자열

    메서드
    ==========
    get_info:
        파생상품 포지션 정보 출력
    """

    def __init__(self, name, quantity, underlying, mar_env, otype, payoff_func):
        self.name = name
        self.quantity = quantity
        self.underlying = underlying
        self.mar_env = mar_env
        self.otype = otype
        self.payoff_func = payoff_func

    def get_info(self):
        print("NAME")
        print(self.name, "\n")
        print("QUANTITY")
        print(self.quantity, "\n")
        print("UNDERLYING")
        print(self.underlying, "\n")
        print("MARKET ENVIRONMENT")
        print("\n**Constants**")
        for key, value in self.mar_env.constants.items():
            print(key, value)
        print("\n**Lists**")
        for key, value in self.mar_env.lists.items():
            print(key, value)
        print("\n**Curves**")
        for key, value in self.mar_env.curves.items():
            print(key, value)
        print("\nOPTION TYPE")
        print(self.otype, "\n")
        print("PAYOFF FUNCTION")
        print(self.payoff_func)
