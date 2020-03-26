# 基于Numpy自己的动手实现深度学习(部分)
代码的实现以及原理学习基于以下书籍:

[📗参考书籍《深度学习入门：基于Python的理论与实现》](https://www.ituring.com.cn/book/1921)

## 用到的Python库
- numpy
- os
- pickle (用来加速数据集载入)
- gzip (处理mnist数据集使用)
- tqdm (现实训练进度的进度条,可以删除相关代码)
- matplotlib (训练网络时候画loss变化图,可以删除相关代码)

## 文件目录信息
- dataset
    - mnist.py  # 用来下载,初始化,加载mnist数据集
- function.py  # 提供了 阶跃函数,relu,sigmoid,softmax 等函数的实现
- layers.py  # 提供了 Relu层,FC层,卷积层,BN层,池化层,sigmoid层 等layer的实现
- loss.py  # 提供了 交叉熵损失函数和均方误差函数的实现
- models.py  # 基于前面的layer和loss定义简单的卷积网络和全连接网络
- optimizer.py  # 定义了部分优化方法 SGD,Adam,Momentum SGD
- train_cnn.py  # 完整的卷积网络训练代码
- train_fc.py  # 完整的全连接网络训练代码
- utils.py  # 提供了卷积时候使用的 im2col,col2im两个转换函数