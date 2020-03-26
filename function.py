# coding=utf-8
# ----tuzixini@gmail.com----
# WIN10 Python3.6.6
# 用途: functions
# 提供自己实现的神经网络中用到的一些function
import numpy as np


def step_function(x):
    # 阶跃函数
    y = x > 0
    return y.astype(np.int)


def sigmoid(x):
    # sigmoid 函数
    return 1 / (1 + np.exp(-x))


def relu(x):
    # relu函数
    return np.maximum(0, x)


def softmax(x):
    if x.ndim == 2:
        import pdb
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))
