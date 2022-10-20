# 프레임 - 고정 단기 이자 할인 클래스
import numpy as np

from get_year_deltas import get_year_deltas


class constant_short_rate(object):
    """
    고정 단기 이자 할인을 위한 클래스

    속성
    =======
    name: string
        객체의 이름
    short_rate: float (positive)
        고정 할인율

    메서드
    =======
    get_discount_factors:
        연수나 datetime 객체의 list/array가 주어졌을 때 할인율 계산
    """

    def __init__(self, name, shor_rate):
        self.name = name
        self.shor_rate = shor_rate
        if shor_rate < 0:
            raise ValueError("short rate negative")

    def get_discount_factors(self, date_list, dtobjects=True):
        if dtobjects is True:
            dlist = get_year_deltas(date_list)
        else:
            dlist = np.array(date_list)

        dflist = np.exp(self.shor_rate * np.sort(-dlist))
        return np.array((date_list, dflist)).T
