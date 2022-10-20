# 프레임 - 도움말 함수
import numpy as np


def get_year_deltas(date_list, day_count=365.):
    """
    날짜 벡터를 연수(year fraction)으로 확산
    초기 날짜는 0이 된다.
    :param date_list: list or array
        datetime 객체 모음
    :param day_count: float
        1년의 날짜 수
    :return: delta_list: array
        연수
    """
    start = date_list[0]
    delta_list = [(date - start).days / day_count for date in date_list]
    return np.array(delta_list)
