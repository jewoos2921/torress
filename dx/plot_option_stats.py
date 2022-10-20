# 가치 평가 - 플롯 옵션 통계

import matplotlib.pyplot as plt


def plot_option_stats(s_list, p_list, d_list, v_list):
    """
    여러 가지 기초 자산 값에 대한 옵션의 가격, 델타, 베가를 플롯
    :param s_list: array or list
        기초 자산의 초깃값 집합
    :param p_list: 
        현재 가치
    :param d_list: 
        델타값
    :param v_list: 
        베가값
    :return: 
    """
    plt.figure(figsize=(10, 6))
    sub1 = plt.subplot(311)
    plt.plot(s_list, p_list, "ro", label="present value")
    plt.plot(s_list, p_list, "b")
    plt.legend(loc=0)
    plt.setp(sub1.get_xticklabels(), visible=False)
    sub2 = plt.subplot(312)
    plt.plot(s_list, d_list, "go", label="Delta")
    plt.plot(s_list, d_list, "b")
    plt.legend(loc=0)
    plt.ylim(min(d_list) - 0.1, max(d_list) + 0.1)
    plt.setp(sub2.get_xticklabels(), visible=False)
    sub3 = plt.subplot(313)
    plt.plot(s_list, v_list, "yo", label="Vega")
    plt.plot(s_list, v_list, "b")
    plt.xlabel("initial value of underlying")
    plt.legend(loc=0)
    plt.setp(sub3.get_xticklabels(), visible=False)
    plt.show()
