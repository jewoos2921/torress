import numpy as np
from scipy import stats


def print_statistics(a1, a2):
    """
    선택된 통계값을 출력
    :param a1: ndattay 객체
    :param a2:
    :return:
    시뮬레이션에서 나온 결과 객체
    """
    sta1 = stats.describe(a1)
    sta2 = stats.describe(a2)
    print("%14s %14s %14s" % ('statistic', 'data set 1', 'data set 2',))
    print(45 * "-")
    print("%14s %14.3f %14.3f" % ('size', sta1[0], sta2[0]))
    print("%14s %14.3f %14.3f" % ('min', sta1[1][0], sta2[1][0]))
    print("%14s %14.3f %14.3f" % ('max', sta1[1][1], sta2[1][1]))
    print("%14s %14.3f %14.3f" % ('mean', sta1[2], sta2[2]))
    print("%14s %14.3f %14.3f" % ('std', np.sqrt(sta1[3]), np.sqrt(sta2[3])))
    print("%14s %14.3f %14.3f" % ('skew', sta1[4], sta2[4]))
    print("%14s %14.3f %14.3f" % ('kurtosis', sta1[5], sta2[5]))

    