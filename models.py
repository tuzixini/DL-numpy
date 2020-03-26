# coding=utf-8
# ----tuzixini@gmail.com----
# WIN10 Python3.6.6
# 用途: 保存一些模型定义

import pdb
import layers
import numpy as np
import function
from loss import cross_entropy_error as CEE
from gradient import numerical_gradient
from collections import OrderedDict


class simpleFC():
    def __init__(self, dimin, dimhid, dimout, weight_init_std=0.01):
        self.layers = OrderedDict()
        self.layers['fc1'] = layers.FC(dimin, dimhid)
        self.layers['relu'] = layers.Relu()
        self.layers['fc2'] = layers.FC(dimhid, dimout)
        self.softmax = layers.SoftmaxWithLoss()

    def forward(self, x):
        t = x
        for key in self.layers.keys():
            t = self.layers[key].forward(t)
        return t

    def loss(self, x, y_):
        y = self.forward(x)
        return self.softmax.forward(y, y_)

    def backward(self, x, y_):
        self.loss(x, y_)
        dout = self.softmax.backward()
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            # pdb.set_trace()
            dout = layer.backward(dout)

    def acc(self, x, y_):
        # 计算准确率
        y = np.argmax(self.forward(x), axis=1)
        acc = np.sum(y == y_) / float(x.shape[0])
        return acc


class simpleCNN:
    def __init__(self, dimhid=100, dimout=10, weight_init_std=0.01, imgsize=28):
        strid = 1
        pad = 0
        fitsize = 3
        cout = 30
        cin = 1
        self.layers = OrderedDict()
        self.layers['conv'] = layers.conv(
            cin, cout, fitsize, strid=strid, pading=pad)
        self.layers['relu1'] = layers.Relu()
        self.layers['Pool1'] = layers.Pooling(pool_h=2, pool_w=2, stride=2)
        con_out_size = (imgsize - fitsize + 2 * pad) / strid + 1
        dimin = int(cout*(con_out_size/2)*(con_out_size/2))
        self.layers['fc1'] = layers.FC(dimin, dimhid)
        self.layers['relu2'] = layers.Relu()
        self.layers['fc2'] = layers.FC(dimhid, dimout)
        self.softmax = layers.SoftmaxWithLoss()

    def forward(self, x):
        t = x
        for key in self.layers.keys():
            t = self.layers[key].forward(t)
        return t

    def loss(self, x, y_):
        y = self.forward(x)
        return self.softmax.forward(y, y_)

    def backward(self, x, y_):
        self.loss(x, y_)
        dout = self.softmax.backward()
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            # pdb.set_trace()
            dout = layer.backward(dout)

    def acc(self, x, y_):
        # 计算准确率
        y = np.argmax(self.forward(x), axis=1)
        acc = np.sum(y == y_) / float(x.shape[0])
        return acc
