# 프레임 - 난수 생성
import numpy as np


def sn_random_numbers(shape, antithetic=True,
                      moment_matching=True,
                      fixed_seed=False):
    """
    표준 정규 분포 (의사) 난수 형태의 nadrray 객체를 반환한다.

    :param shape:
        (o, n, m) 형태의 배열 생성
    :param antithetic:
        대조변수 생성
    :param moment_matching:
        1차 및 2차 모멘트 정합
    :param fixed_seed:
        seed 값 수정 인수
    :return:
        ran: (o, n, m) array of (pseudo) random numbers
    """
    if fixed_seed:
        np.random.seed(1000)
    if antithetic:
        ran = np.random.standard_normal((shape[0], shape[1], shape[2] // 2))
        ran = np.concatenate((ran, -ran), axis=2)
    else:
        ran = np.random.standard_normal(shape)

    if moment_matching:
        ran = ran - np.mean(ran)
        ran = ran / np.std(ran)
    if shape[0] == 1:
        return ran[0]
    else:
        return ran
