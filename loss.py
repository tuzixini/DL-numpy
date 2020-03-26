# coding=utf-8
# ----tuzixini@gmail.com----
# WIN10 Python3.6.6
# 用途: loss 函数

import numpy as np


def mean_squared_error(y, t):
    # MSE 均方误差
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, y_):
    # 交叉熵损失函数
    if y.ndim == 1:
        y_ = y_.reshape(1, y_.size)
        y = y.reshape(1, y.size)
    if y_.size == y.size:
        y_ = y_.argmax(axis=1)
    BS = y.shape[0]
    return -np.sum(np.log(y[np.arange(BS), y_] + 1e-7)) / BS
