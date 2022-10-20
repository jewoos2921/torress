# 프레임 - 시장 환경 클래스

class market_environment(object):
    """
    가치 평가에 필요한 시장 환경을 모델링하기 위한 클리스

    속성
    ======
    name: string
        시장 환경의 이름
    pricing_date: datetime object
        시장 환경의 날짜

    메서드
    ======
    add_constant:
        (모델 인수 등의) 상수 추가
    get_constant:
        상숫값 출력
    add_list:
        (기초 상품 등의) 리스트 추가
    get_list:
        리스트 출력
    add_curve:
        (이자율 커브 등의)시장 커브 추가
    get_curve:
        시장 커브 출력
    add_environment:
        상수, 리스트, 커브 등의 전체 시장 환경을 추가하거나 덮어쓰기
    """

    def __init__(self, name, pricing_date):
        self.name = name
        self.pricing_date = pricing_date
        self.constants = {}
        self.lists = {}
        self.curves = {}

    def add_constant(self, key, constant):
        self.constants[key] = constant

    def get_constant(self, key):
        return self.constants[key]

    def add_list(self, key, list_object):
        self.lists[key] = list_object

    def get_list(self, key):
        return self.lists[key]

    def add_curve(self, key, curve):
        self.curves[key] = curve

    def get_curve(self, key):
        return self.curves[key]

    def add_environment(self, env):
        # 값이 존재하는 경우 덮어쓰기
        self.constants.update(env.constants)
        self.lists.update(env.lists)
        self.curves.update(env.curves)
