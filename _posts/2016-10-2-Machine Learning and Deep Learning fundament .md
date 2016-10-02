---
layout: post
title: Machine Learning and Deep Learning fundament
---


```python
from IPython.display import display, Math, Latex
```

以下内容是我学习以下几本书的学习笔记
机器学习部分：
1. Andrew Ng的机器学习讲义(网易公开课里边可以打包下载)，重点推荐
2. 机器学习(周志华)  

神经网络部分：
1. Haykin的神经网络与机器学习
2. Deep Learning([book address](http://www.deeplearningbook.org/))

然后我没有做过多的数学公式描述，要看详细公式推导，看原书是最好的。我只是把我自己从中理解到的东西归纳整理出来，方便大家更快的理解，不用一上来就从公式中去理解算法的思路，那样会很痛苦，理解效果不好，学习曲线也很陡峭。

## 机器学习和神经网络的一般流程

> 用一个例子说明：输入为x，输出为y，机器学习的目标就是，在我们不知道x与y的确切关系时，用大量的样本去拟合x与y的关系。下面一一介绍拟合过程中需要解决的问题。  

- 有了输入x和输出y之后，我们必须为x与y之间的关系建模：也就是我们必须确定***模型***。简而言之，模型就是我们通过客观观察和分析，人为的指定的x与y之间的关系。比如：$$y = \theta x$$ 这里我们就为x和y建立了一个线性模型。  
当然我们也可以为x，y建立二次模型，或者对数关系： $$y= x^2$$ $$y=logx$$ 或者更为复杂的模型
- 假设我们已经对x与y进行了建模，他们之间的关系符合： $$ f(x) = \theta x$$ (这里留意一下我的写法，没有用$y=\theta x$) 因为\theta现在处于一个一维参数空间中，所以需要通过训练（其实就是拟合），将$\theta$的值确定下来。那么现在需要解决的问题就是：衡量模型与观测结果之间的差别，也就是y与f(x)之间的相对差别，方法就是 ***损失函数(loss function，cost function)***,最常见的损失函数就是均方损失函数 $$ loss = L(\theta) = (y - f(x)) ^ 2 = (y - \theta x) $$
- 有了损失函数，下一步就是该训练模型了，也就是确定\theta，这里就需要用到各种***优化算法***（神经网络里边叫optimizor），最常见的优化算法就是Gridient Decent，以及由此扩展的Stochastic Gridient Decent(SGD)。关于梯度下降的算法详细介绍，看下边的小节
- 上边就已经差不多是一个简单的机器学习训练过程了，但是实际情况肯定没有如此简单，当输入x不是标量，是向量时，然后我们的模型参数也不只一个参数时，就开始变复杂了。这时会出现一个训练中最常见的情况，***overfitting和underfitting***（过拟合和欠拟合）。很好理解，就是训练出来的模型，过于复杂或简单，以至于泛化能力弱（就是对训练数据以外的数据，预测能力较差）。解决这个问题的办法就是***正则化***（神经网络里也叫weight decay，权值衰减）。正则化的目标就是通过少参数个数（参数值等于0），或者减小参数的值减小模型的复杂度。常见的正则化：L1和L2。
- 总结一下：将模型和正则化结合在一起，可以将要训练的目标函数Object function写成如下形式$$ object = loss + regularization(正则化项) $$ 然后通过优化算法，不断寻找减小loss的参数，最后找到一个能使loss最小的参数，训练过程就结束了。最后必须说明一下，为了不让大家误解，上边的过程是一般情况，部分算法是不符合这个流程的，比如聚类相关算法

### 常见模型
> 先交待一下下面要用到的记号：  
$x_1, x_2 ,... ,x_n也就是\mathbf{x}$ : 训练集的输入input  
$y$  : 训练样本的目标结果  
$f(\mathbf{x})$ :模型的output

#### 线性回归模型
Motivation:线性回归模型是很经典，古老的模型。线性模型很多情况下，具有确定的代数解，也就是熟知的最小二乘法。但是某些情况是不能直接代数求解的，需要用一般的迭代算法，比如下边要说的梯度下降算法。线性模型公式表述：$$y=\theta_1 x_1 + ... + \theta_n x_n = \mathbf{\theta^\mathrm{T} x}$$

#### logistic 回归（用于分类）
Motivation:现在我们有了1000个人的身高($x_1$)，体重($x_2$)，年龄($x_3$)信息，然后要根据这三个信息预测男女性别。这时就不能用简单的线性回归模型，因为线性回归的模型输入是连续值，而这里输出只允许为0或1其中的一个（假设我们将男性编码为0，女性编码为1）。这种输入只允许为某些特定数的问题就是分类问题，分类模型，一般解决思路就是找到几条直线（平面中），或几个超平面（高维空间中）来将不同类别的点分隔开。  
先说上边的区分男女性别的二分类情况。身高($x_1$)，体重($x_2$)，年龄($x_3$),实际就是一个三维空间，然后每个样本（人）的数据，就是这个三维空间中的一个点，我们希望找到这样一个平面，
$$\theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 = 0$$ 
可以将两种类别的样本分开。我们记
$$ f(\mathbf{x}) = \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 $$ 
当一个点位于平面下方时，也就是$f(\mathbf{x}) < 0$时，样本的性别就是男，当一个点位于平面上方时，也就是$f(\mathbf{x}) > 0$时，样本的性别就是女。所以问题就简单了，我们需要找到这样一种函数，当函数自变量大于一个值一丁点儿时就输出1，自变量小于一个值一丁点儿时就输出0，是不是很像逻辑中的0-1 gate（logistic回归的含义）。然后恰好就有这么一个函数符合条件：
$$ y = \frac{1}{(1 + e^{-x})} $$
到此为止我们的模型就确定下来了：
$$f(\mathbf{x}) = \frac{1}{1+e^{-(\theta_1 x_1 + \theta_2 x_2 + \theta3 x_3)}} = \frac{1}{1+e^{-\mathbf{\theta^\mathrm{T} x}}}$$
实际上，logistic函数中的y的含义被重修修改了一下，上边我们说的y的含义直接是指男或女，但是实际中我们是把y=1当做该样本为女性的概率,即$P(女性|\mathbf{x})$，y=0当做时该样本为非女性，也就是男性的概率,$P(男性|\mathbf{x})$。所以最终的logistic数学形式就是：
$$女性的概率=P(y=1|\mathbf{x}) = \frac{1}{1+e^{-\mathbf{\theta^\mathrm{T} x}}}$$

$$男性的概率=P(y=0|\mathbf{x}) = \frac{e^{-\mathbf{\theta^\mathrm{T} x}}}{1+e^{-\mathbf{\theta^\mathrm{T} x}}}$$

#### Softmax
上边讨论了用于二分类的logistic回归，但是很多时候我们需要分辩多种类别，比如根据动物的特征区分是哪种动物。为了引出Softmax函数公式,且为了显示logistic回归是softmax中的一种特殊形式，我们先从logistic出发，将它写成一般形式。  
首先我们将logistic回归用向量的形式写出来：
$$我们令原来的\theta=\theta_1 - \theta_2$$
那么logistic的公式就可以写成：
$$女性的概率=P(y=1|\mathbf{x}) = \frac{1}{1+e^{-\mathbf{(\theta_1 -\theta_2) x}}} = \frac{e^{\mathbf{\theta_1 x}}}{e^{\mathbf{\theta_1 x}}+e^{\mathbf{\theta_2 x}}} = \frac{e^{\mathbf{\theta_1 x}}}{\sum_{i=1}^ {2}e^{\mathbf{\theta_i^\mathrm{T} x}}}$$

$$男性的概率=P(y=0|\mathbf{x}) = \frac{e^{-\mathbf{(\theta_1 -\theta_2) x}}}{1+e^{-\mathbf{(\theta_1 -\theta_2) x}}} = \frac{e^{\mathbf{\theta_2 x}}}{e^{\mathbf{\theta_1 x}}+e^{\mathbf{\theta_2 x}}} = \frac{e^{\mathbf{\theta_2 x}}}{\sum_{i=1}^ {2}e^{\mathbf{\theta_i^\mathrm{T} x}}}$$

是不是已经发现了什么？接下来我们用矩阵的形式写下来：
$$\begin{bmatrix} P(y=1) \\ P(y=2) \\ \vdots \\ p(y=n) \end{bmatrix} 
= \begin{bmatrix} \frac{e^{\mathbf{\theta_1^\mathrm{T} x}}}{\sum_{i=1}^ {n}e^{\mathbf{\theta_i^\mathrm{T} x}}} \\ \frac{e^{\mathbf{\theta_2^\mathrm{T} x}}}{\sum_{i=1}^ {n}e^{\mathbf{\theta_i^\mathrm{T} x}}} \\ \vdots \\ \frac{e^{\mathbf{\theta_n^\mathrm{T} x}}}{\sum_{i=1}^ {n}e^{\mathbf{\theta_i^\mathrm{T} x}}} \end{bmatrix}$$
这里y可以有n种类别，比如有100种动物（猫，狗，牛，鸡...），那么在这个模型中，我们就可以为猫编码为：
$$\begin{bmatrix} P(猫) & P(狗) & \cdots & P(树獭) \end{bmatrix}^\mathrm{T} = \begin{bmatrix} 1 & 0 & \cdots & 0\end{bmatrix}^\mathrm{T}$$

总结一下：我们可以把$e^{\mathbf{\theta_i x}}$理解为第i个类别在所有类别中所占的权重，那么很自然的，第i个类别的概率就可以写成：
$\frac{e^{\mathbf{\theta_i x}}}{\sum_{i=1}^ {n}e^{\mathbf{\theta_i x}}}$ 到此就可以基本理解softmax回归了。  
然后再说点题外话，softmax有个缺点，当类别较多的时候，你需要计算每个类别的概率，然后比较这些概率谁大，对应的类别就是output。比如在自然语言的word embedding中，给每个字词以向量表示，这时直接用softmax是不可取的，训练的计算量太大，这时可以用一种扩展的方法（NCE）来处理，这个在自然语言处理部分再说。


#### Naive Bayes(可用于分类，也可用于回归)
#### Gaussian Discriminant
#### SVM(Support Vector Machines)

### loss function
#### 均方损失函数
#### 0-1损失函数
#### hinge损失函数
#### 交叉熵(cross entropy)
#### EM算法

### 优化方法
#### 梯度下降法与随机梯度下降发
#### 坐标下降法
#### 牛顿法

### 正则化
#### L1
#### L2
#### cross validation(这里主要讲K-fold)
#### dropout(主要用于神经网络)

## 其他算法
### 聚类分析
### 混合高斯模型
### Factor Analysis(FA)
### Principal Component Analysis(PCA，这里顺带讲下矩阵奇异值分解SVD)
### 独立主成分分析
### Reinforcement Learning


```python

```
