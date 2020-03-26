# coding=utf-8
# ----tuzixini@gmail.com----
# WIN10 Python3.6.6
# 用途: 训练网络
import pdb
from tqdm import tqdm
import numpy as np
import pickle
import optimizer
from dataset.mnist import load_mnist
from models import simpleFC
import matplotlib.pyplot as plt



def get_data(flag="train"):
    (x_train, y_train), (x_test, y_test) = load_mnist(
        normalize=True, flatten=True, one_hot_label=False)
    if flag == "train":
        return x_train, y_train
    elif flag == "test":
        return x_test, y_test
    else:
        print("data type error!!!")


batch_size = 100  # 批数量
lr = 0.1
EPOCH = 100

net = simpleFC(784, 50, 10)
optim = optimizer.SGD(lr=lr)
x_train, y_train = get_data(flag="train")
train_size = x_train.shape[0]
x_test, y_test = get_data(flag="test")

train_loss = []
train_acc = []
test_loss = []
test_acc = []

for epoch in tqdm(range(EPOCH)):
    # train
    for i in range(0, len(x_train), batch_size):
        batch_mask = np.random.choice(train_size, batch_size)
        x = x_train[batch_mask]
        y_ = y_train[batch_mask]
        net.backward(x, y_)
        temp = net.loss(x, y_)
        train_loss.append(temp)
        optim.step(net.layers)
    # test
    tempacc = net.acc(x_train, y_train)
    train_acc.append(tempacc)
    tempt = net.acc(x_test, y_test)
    test_acc.append(tempt)
    print("Train ACC:{}, Test ACC:{}  for EPOCH:{}".format(tempacc, tempt, epoch))

# 绘制图形
markers = {'train': 'o', 'test': 's'}
x = np.arange(EPOCH)
plt.plot(x, train_acc, label='train acc')
plt.plot(x, test_acc, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
