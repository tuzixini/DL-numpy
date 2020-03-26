# coding=utf-8
# ----tuzixini@gmail.com----
# WIN10 Python3.6.6
# 用途: 自己实现一些神经网络的层

import pdb
import numpy as np
import function
import loss
import utils
from utils import im2col, col2im


class Relu:
    def __init__(self):
        self.par = None
        self.gra = None
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class Sigmoid:
    def __init__(self):
        self.par = None
        self.gra = None
        self.out = None

    def forward(self, x):
        self.out = function.sigmoid(x)
        return self.out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


class FC:
    def __init__(self, dimin, dimout, weight_init_std=0.01):
        self.par = dict()
        self.par['w'] = weight_init_std * np.random.randn(dimin, dimout)
        self.par['b'] = np.zeros(dimout)
        self.gra = dict()
        self.gra['w'] = None
        self.gra['b'] = None
        self.x = None
        self.xshape = None

    def forward(self, x):
        self.xshape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        return np.dot(self.x, self.par['w']) + self.par['b']

    def backward(self, dout):
        dx = np.dot(dout, self.par['w'].T)
        self.gra['w'] = np.dot(self.x.T, dout)
        self.gra['b'] = np.sum(dout, axis=0)
        dx = dx.reshape(*self.xshape)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.par = None
        self.gra = None
        self.loss = None
        self.y = None  # softmax的输出
        self.t = None  # 监督数据

    def forward(self, x, t):
        self.t = t
        self.y = function.softmax(x)
        self.loss = loss.cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 监督数据是one-hot-vector的情况
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx


class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    """

    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.par = None
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None  # Conv层的情况下为4维，全连接层的情况下为2维

        # 测试时使用的平均值和方差
        self.running_mean = running_mean
        self.running_var = running_var

        # backward时使用的中间数据
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)

        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * \
                self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * \
                self.running_var + (1-self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx


class conv:
    def __init__(self, cin, cout, filtersize, strid=1, pading=0, weight_init_std=0.01):
        self.strid = strid
        self.pading = pading
        # 参数
        self.par = dict()
        w = weight_init_std * \
            np.random.randn(cout, cin, filtersize, filtersize)
        self.par['w'] = w
        b = np.zeros(cout)
        self.par['b'] = b
        # 梯度
        self.gra = dict()
        self.gra['w'] = None
        self.gra['b'] = None
        # backward 需要的数据
        self.x = None
        self.col = None
        self.col_w = None

    def forward(self, x):
        FN, C, FH, FW = self.par['w'].shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pading - FH) / self.strid)
        out_w = 1 + int((W + 2*self.pading - FW) / self.strid)

        col = im2col(x, FH, FW, self.strid, self.pading)
        col_W = self.par['w'].reshape(FN, -1).T

        out = np.dot(col, col_W) + self.par['b']
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.par['w'].shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.gra['b'] = np.sum(dout, axis=0)
        self.gra['w'] = np.dot(self.col.T, dout)
        self.gra['w'] = self.gra['w'].transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.strid, self.pading)

        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.par = None
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size),
             self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h,
                    self.pool_w, self.stride, self.pad)

        return dx
