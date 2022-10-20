import numpy as np
import pandas as pd

r = 0.05  # 슛 레이트 상수
sigma = 0.5  # 변동성 계수


def generate_sample_data(rows, cols, freq='1min'):
    """
    모의 금융 데이터를 생성하기 위한 함수.
    :param rows: int 생성할 행의 개수
    :param cols: int 생성할 열의 개수
    :param freq: str DatetimeIndex에 대한 빈도 문자열
    :return:
    df : DataFrame 모의 데이터가 있는 객체
    """
    rows = int(rows)
    cols = int(cols)
    # 주어진 빈도에 대한 DatetimeIndex 객체
    index = pd.date_range('2021-1-1', periods=rows, freq=freq)
    # 연도 부분들 내의 타임 델타를 정함
    dt = (index[1] - index[0]) / pd.Timedelta(value="365D")
    # 열 이름들을 생성
    columns = ["No%d" % i for i in range(cols)]
    # 기하적 브라운 운동에 대한 단순 경로들을 생성한다.
    raw = np.exp(np.cumsum((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) *
                           np.random.standard_normal((rows, cols)), axis=0))

    # 데이터를 정규화하여 100에서 시작하게 한다.
    raw = raw / raw[0] * 100
    # DataFrame 객체를 생성
    df = pd.DataFrame(raw, index=index, columns=columns)
    return df
