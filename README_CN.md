# 给初学者的梯度下降
---

## 目录
1. 问题引入
2. 回归的局限性与挑战
3. 损失曲面与寻找最优解
4. 梯度下降的数学逻辑
5. 反向传播的直观理解
6. 从单变量到多变量的推导
7. 高维特征与优化策略
8. 非线性传递链与激活函数

---

## 01 问题引入
二氧化碳（CO₂）排放是全球变暖的主要原因之一，而车辆是主要的排放源。
- **目标**：构建一个模型，输入车辆特征（发动机排量、燃油消耗量等），输出预测的 CO₂ 排放量。
- **数据集**：FuelConsumptionCo2.csv（来自加拿大政府的公开数据集）。

---

### 互动：需求分析
打开互动页面，探索数据，反思预测的意义与挑战，直观理解数据预测在现实世界中的重要性。

---

## 02 回归的局限性与挑战
### 高中统计学 vs. 测试集验证
使用简单回归拟合数据并观察其在测试集上的表现，可以发现：
- 传统统计方法在处理复杂数据时会遇到瓶颈。
- 计算速度受到严重限制。

---

### 互动：模型假设
打开互动页面，观察使用最小二乘法回归的线性模型的拟合性能：
- 模型受多个特征影响，导致拟合效果不理想。
- 我们需要增加特征以提高模型复杂度，从而提升预测精度。

---

## 03 损失曲面与寻找最优解
### 什么是“损失”？
损失函数用于衡量预测值与真实值之间的差异。损失越小，预测越好。
- **均方误差 (MSE)**: $L=\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$
- **平均绝对误差 (MAE)**: $L=\frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$
- **二元交叉熵**: $L=-\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$
- **分类交叉熵**: $L=-\frac{1}{n} \sum_{i=1}^{n} \sum_{k=1}^{K} y_{i,k} \log(\hat{y}_{i,k})$

---

### 互动 3：可视化理解
打开互动页面，手动调整参数 $k$ 和 $b$ 以找到损失函数的最小值，并观察模型拟合情况。

#### 思考与探索
我们能否设计一种算法来寻找损失函数的最小值以进行拟合，而不是采用暴力求解？

---

## 04 梯度下降的数学逻辑
### 梯度的定义
- **梯度**是一个函数所有一阶偏导数构成的向量。

函数在某一点的导数代表了函数变化的方向和速率。

---

### 互动：参数更新演示
打开互动页面，调整学习率，观察小球在损失曲面上的运动轨迹，直观理解学习率如何影响收敛速度和稳定性。

---

## 05 反向传播的直观理解
### 反向传播的核心定义
反向传播是一种高效计算梯度的方法。

它从输出层到输入层，逐层计算损失函数关于每个参数的偏导数。

它利用**链式法则**分解复杂的导数，从而高效地计算网络参数的梯度。

---

### 下降过程
沿**负梯度方向**迭代更新参数 $x$，直到达到最小点。

$$Loss=\frac{x^2}{2}$$
$$Gradient=\frac{d}{dx}\cdot\frac{x^2}{2}=x$$

```python
lr = 0.1
x = 2
for _ in range(100):
    Loss = (x ** 2) / 2 # 损失函数
    Gradient = x # 梯度
    x = x - lr * Gradient # 更新参数
print(f'x:{x:.2f}; Loss:{Loss:.2f}')
```

---

## 06 从单变量到多变量的推导
1. 标量计算规则的扩展
2. 多变量特征的梯度下降

---

### 标量计算规则的扩展

$$\begin{bmatrix} w_1, w_2, \dots, w_n \end{bmatrix} \cdot \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} + b = C $$

$$\begin{bmatrix} w_{11} & w_{12} & \dots & w_{1n} \\ w_{21} & w_{22} & \dots & w_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ w_{m1} & w_{m2} & \dots & w_{mn} \end{bmatrix} \cdot \begin{bmatrix} x_{11} & x_{12} & \dots & x_{1k} \\ x_{21} & x_{22} & \dots & x_{2k} \\ \vdots & \vdots & \ddots & \vdots \\ x_{n1} & x_{n2} & \dots & x_{nk} \end{bmatrix} + \begin{bmatrix} b_{11} & b_{12} & \dots & b_{1k} \\ b_{21} & b_{22} & \dots & b_{2k} \\ \vdots & \vdots & \ddots & \vdots \\ b_{m1} & b_{m2} & \dots & b_{mk} \end{bmatrix} = \begin{bmatrix} c_{11} & c_{12} & \dots & c_{1k} \\ c_{21} & c_{22} & \dots & c_{2k} \\ \vdots & \vdots & \ddots & \vdots \\ c_{m1} & c_{m2} & \dots & c_{mk} \end{bmatrix} $$

---

### 标量计算规则的扩展

1. **标量展开**
$$WX + b = C$$
$$w_1x_1 + w_2x_2 + \dots + w_nx_n + b = C \\$$

2. **对单个权重的导数**
$$\frac{dC}{dw_n} = \frac{d}{dw_n}(w_1x_1 + w_2x_2 + \dots + w_nx_n + b) = x_n \\$$

3. **合并为矩阵形式**
$$\frac{dC}{dW} = [x_1, x_2, \dots, x_n]^T = X^T \\$$

4. **对偏置的导数**
$$\frac{dC}{db} = \frac{d}{db}(w_1x_1 + w_2x_2 + \dots + w_nx_n + b) = 1$$

---
### 标量计算规则的扩展

$$WX + B = C \\$$

1. **标量展开**

$$c_{ij} = \sum_{t=1}^n w_{it}x_{tj} + b_{ij} \quad (i=1,2,\dots,m; j=1,2,\dots,k) \\$$

2. **对单个值的导数**

$$\frac{dC}{dw_{pq}} = \frac{d}{dw_{pq}}\left(\sum_{t=1}^n w_{it}x_{tj} + b_{ij}\right) = x_{qj}; 

\frac{dC}{db_{pq}} = \frac{d}{db_{pq}}\left(\sum_{t=1}^n w_{it}x_{tj} + b_{ij}\right) = 1 \\$$

3. **合并为矩阵形式**

$$\frac{dC}{dW} = [x_1, x_2, \dots, x_n]^T = X^T ;
\frac{dB}{dW} = [x_1, x_2, \dots, x_n]^T = \textbf{1}_{m\times k} \\$$
*(注：原文此处 $\frac{dB}{dW}$ 似有笔误，根据上下文应为 $\frac{dC}{dB}$ 且结果为全1矩阵)*

---

### 多变量特征的梯度下降

$Y_{pred} = W \cdot X + B$  

$L = \text{MSE} = \frac{1}{2}(Y_{true} - Y_{pred})^2$

$\frac{dL}{dY_{pred}} = \frac{d}{dY_{pred}} \left[\frac{1}{2}(Y_{true} - Y_{pred})^2\right] = -(Y_{true} - Y_{pred})$

$\frac{dY_{pred}}{dW} = \frac{d}{dW}(W \cdot X + B) = X^T$  

$\frac{dY_{pred}}{dB} = \frac{d}{dB}(W \cdot X + B) = 1$

###### 最终梯度（链式法则）

$\frac{dL}{dW} = \frac{dL}{dY_{pred}} \cdot \frac{dY_{pred}}{dW} = -(Y_{true} - Y_{pred}) \cdot X^T = (Y_{pred} - Y_{true}) \cdot X^T$  

$\frac{dL}{dB} = \frac{dL}{dY_{pred}} \cdot \frac{dY_{pred}}{dB} = -(Y_{true} - Y_{pred}) \cdot 1 = Y_{pred} - Y_{true}$

---

### 多变量特征与多样本的梯度下降

$Y_{pred}^{(i)} = W \cdot X^{(i)} + B$  

$L = \text{MSE} = \frac{1}{2n}\sum_{i=1}^{n}(Y_{true}^{(i)} - Y_{pred}^{(i)})^2$

$\frac{dL}{dY_{pred}^{(i)}} = \frac{d}{dY_{pred}^{(i)}} \left[\frac{1}{2n}\sum_{i=1}^{n}(Y_{true}^{(i)} - Y_{pred}^{(i)})^2\right] = -\frac{1}{n}(Y_{true}^{(i)} - Y_{pred}^{(i)})$

$\frac{dY_{pred}^{(i)}}{dW} = \frac{d}{dW}(W \cdot X^{(i)} + B) = (X^{(i)})^T$ 

$\frac{dY_{pred}^{(i)}}{dB} = \frac{d}{dB}(W \cdot X^{(i)} + B) = 1$

###### 最终梯度（链式法则）

$\frac{dL}{dW} = \sum_{i=1}^{n}\left[-\frac{1}{n}(Y_{true}^{(i)} - Y_{pred}^{(i)}) \cdot (X^{(i)})^T\right] = \frac{1}{n}\sum_{i=1}^{n}\left[(Y_{pred}^{(i)} - Y_{true}^{(i)}) \cdot (X^{(i)})^T\right]$  

$\frac{dL}{dB} = \sum_{i=1}^{n}\left[-\frac{1}{n}(Y_{true}^{(i)} - Y_{pred}^{(i)}) \cdot 1\right] = \frac{1}{n}\sum_{i=1}^{n}\left(Y_{pred}^{(i)} - Y_{true}^{(i)}\right)$

---

## 07 高维特征与优化策略
在现实中，车辆排放受多种特征影响：燃油消耗量、发动机排量、气缸数、燃油类型、重量等。

当特征数量超过 2 个时，我们就进入了高维空间。

#### 高维空间中的挑战
1. **参数数量爆炸** → 计算和内存成本更高，收敛更慢。
2. **损失曲面复杂** → 存在鞍点、平坦区域、局部最优解，容易陷入其中。

---

### SGD – 随机梯度下降 (Stochastic Gradient Descent)
- **核心**：使用单个样本或小批量样本（mini-batch）计算梯度，而非整个数据集。
- **优点**：训练速度快，内存占用低，适用于大数据/高维场景；噪声有助于跳出局部最优解。
- **缺点**：梯度估计有噪声，更新不稳定，收敛过程震荡。

---

### Adam – 自适应矩估计 (Adaptive Moment Estimation)
- 结合了动量（一阶矩）和自适应学习率（二阶矩）。
- 利用梯度及其平方的指数移动平均值，动态调整每个参数的学习率。
- **优点**：收敛快速且稳定；自适应学习率减少了人工调参的需求；对稀疏梯度/噪声数据具有鲁棒性；非常适用于非凸问题和大规模深度学习。

1. $\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \nabla_{\theta} Loss_t$

2. $\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) (\nabla_{\theta} Loss_t)^2$

3. $\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}$

4. $\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}$

---

### PCA – 主成分分析 (Principal Component Analysis)
一种经典的无监督降维方法，旨在尽可能保留信息的同时将高维数据压缩到低维。
- 寻找方差最大的方向作为主成分，去除冗余和高度相关的维度。
- 缓解维数灾难，减少计算量，消除特征相关性，平滑损失曲面，并实现高维数据的可视化。

---

## 08 非线性传递链与激活函数
##### 线性模型的局限性
仅使用线性变换（$y = wx + b$），无论堆叠多少层，整个模型仍然是输入的线性组合。
它无法模拟大多数现实世界的非线性问题：图像边缘检测、语音语调变化、金融数据波动等。

##### 激活函数的作用
激活函数是应用于神经元输出的**非线性映射**，打破了堆叠线性变换的局限性。
应用 Sigmoid 和 ReLU 等激活函数后，输出不再是输入的线性函数。
多层网络因此获得了拟合任意复杂非线性函数的能力。

---

### Sigmoid 函数

$$\sigma(z) = \frac{1}{1+e^{-z}}$$

$$\sigma'(z)= \frac{d}{dz}\left(\frac{1}{1+e^{-z}}\right)= \frac{e^{-z}}{(1+e^{-z})^2}$$

$$\sigma'(z)= \frac{1}{1+e^{-z}} \cdot \frac{e^{-z}}{1+e^{-z}}= \sigma(z) \cdot \big(1-\sigma(z)\big)$$

$$\sigma'(z) = \sigma(z)\big(1-\sigma(z)\big)$$

---

### ReLU 函数

$$\text{ReLU}(z) = \max(0, z)$$

$$\text{ReLU}'(z)= \frac{d}{dz}\left(\max(0, z)\right)= \begin{cases} 1, & z > 0 \\ 0, & z < 0 \\ \text{未定义}, & z = 0 \end{cases}$$

$$\text{ReLU}'(z)= \begin{cases} 1, & z > 0 \\ 0, & z \leq 0 \end{cases}$$

---

### Tanh 函数

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

$$\tanh'(z)= \frac{d}{dz}\left(\frac{e^z - e^{-z}}{e^z + e^{-z}}\right)= \frac{(e^z + e^{-z})^2 - (e^z - e^{-z})^2}{(e^z + e^{-z})^2}$$

$$\tanh'(z)= \frac{4}{(e^z + e^{-z})^2}= 1 - \left(\frac{e^z - e^{-z}}{e^z + e^{-z}}\right)^2$$

$$\tanh'(z) = 1 - \tanh^2(z)$$

---

### Softmax 函数

$$\sigma_i(\boldsymbol{z}) = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}} \quad (i=1,2,...,K)$$

$$\frac{d \sigma_i}{d z_k} = \frac{d}{d z_k}\left(\frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}\right) = \begin{cases} \frac{e^{z_i}\left(\sum_{j=1}^K e^{z_j} - e^{z_i}\right)}{\left(\sum_{j=1}^K e^{z_j}\right)^2}, & i=k \\ -\frac{e^{z_i}e^{z_k}}{\left(\sum_{j=1}^K e^{z_j}\right)^2}, & i≠k \end{cases}$$

$$\frac{d \sigma_i}{d z_k} = \begin{cases} \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}} \cdot \left(1 - \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}\right), & i=k \\ -\frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}} \cdot \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}, & i≠k \end{cases}$$

$$\frac{d \sigma_i}{d z_k} = \begin{cases} \sigma_i(\boldsymbol{z}) \cdot \big(1 - \sigma_i(\boldsymbol{z})\big), & i=k \\ -\sigma_i(\boldsymbol{z}) \cdot \sigma_k(\boldsymbol{z}), & i≠k \end{cases}$$

---

### DG 任务 1

$$Z_1 = XW_1 + b_1,\quad A_1 = \sigma(Z_1)$$
$$Z_2 = A_1W_2 + b_2,\quad A_2 = \sigma(Z_2)$$
$$Z_3 = A_2W_3 + b_3,\quad \hat{y} = \sigma(Z_3)$$
$$Loss = \frac{1}{m}\sum_{i=1}^{m} (\hat{y}_i - y_i)^2$$

---

### DG 任务 1 解答

#### 1. 首先推导各层之间的误差项
- **输出层误差项**: $\delta_3 = (\hat{y} - y) \cdot \sigma'(Z_3) = (A_3 - y) \cdot A_3(1-A_3)$
 ```python
 a2_ = Loss_*a[2]*(1-a[2])
 ```
*(注：代码变量名 `a2_` 对应 $\delta_3$，`Loss_` 在此处似乎指代 $(\hat{y}-y)$ 部分)*

- **隐藏层 2 误差项**: $\delta_2 = \delta_3 \cdot W_2^T \cdot \sigma'(Z_2) = \delta_3 \cdot W_2^T \cdot A_2(1-A_2)$
 ```python
 a1_ = np.dot(a2_, w[2].T)*a[1]*(1-a[1])
 ```

- **隐藏层 1 误差项**: $\delta_1 = \delta_2 \cdot W_1^T \cdot \sigma'(Z_1) = \delta_2 \cdot W_1^T \cdot A_1(1-A_1)$
 ```python
 a0_ = np.dot(a1_, w[1].T)*a[0]*(1-a[0])
 ```

---

### DG 任务 1 解答
#### 2. 各参数的梯度公式（核心）
###### (1) $W_2$ (w[2]) 的梯度: $\nabla_{W_2} Loss = \frac{1}{m} A_2^T \cdot \delta_3$

代码中的权重更新: $W_2 = W_2 - \eta \cdot \frac{1}{m} A_2^T \cdot \delta_3$

 ```python
 w[2] = w[2] - lr*np.dot(a[1].T, a2_)/l
 ```

###### (2) $b_2$ (b[2]) 的梯度: $\nabla_{b_2} Loss = \frac{1}{m} \sum_{i=1}^m \delta_{3(i)}$ (按行求和)
代码中的偏置更新: $b_2 = b_2 - \eta \cdot \frac{1}{m} \sum \delta_3$

 ```python
 b[2] = b[2] - lr*np.sum(a2_, axis=0, keepdims=True)/l
 ```

---

### DG 任务 1 解答
#### 2. 各参数的梯度公式（核心）
###### (3) $W_1$ (w[1]) 的梯度: $\nabla_{W_1} Loss = \frac{1}{m} A_1^T \cdot \delta_2$
代码中的权重更新: $W_1 = W_1 - \eta \cdot \frac{1}{m} A_1^T \cdot \delta_2$

 ```python
 w[1] = w[1] - lr*np.dot(a[0].T, a1_)/l
 ```
###### (4) $b_1$ (b[1]) 的梯度: $\nabla_{b_1} Loss = \frac{1}{m} \sum_{i=1}^m \delta_{2(i)}$
代码中的偏置更新: $b_1 = b_1 - \eta \cdot \frac{1}{m} \sum \delta_2$

 ```python
 b[1] = b[1] - lr*np.sum(a1_, axis=0, keepdims=True)/l
 ```

---

### DG 任务 1 解答
#### 2. 各参数的梯度公式（核心）
###### (5) $W_0$ (w[0]) 的梯度: $\nabla_{W_0} Loss = \frac{1}{m} X^T \cdot \delta_1$
代码中的权重更新: $W_0 = W_0 - \eta \cdot \frac{1}{m} X^T \cdot \delta_1$

 ```python
 w[0] = w[0] - lr*np.dot(data.T, a0_)/l
 ```

###### (6) $b_0$ (b[0]) 的梯度: $\nabla_{b_0} Loss = \frac{1}{m} \sum_{i=1}^m \delta_{1(i)}$
代码中的偏置更新: $b_0 = b_0 - \eta \cdot \frac{1}{m} \sum \delta_1$

 ```python
 b[0] = b[0] - lr*np.sum(a0_, axis=0, keepdims=True)/l
 ```

---

### DG 任务 2

$$Z_0 = XW_0 + b_0,\quad A_0 = \tanh(Z_0)$$
$$Z_1 = A_0W_1 + b_1,\quad A_1 = \text{ReLU}(Z_1)$$
$$Z_2 = A_1W_2 + b_2,\quad \hat{y} = \text{Softmax}(Z_2)$$
$$Loss = -\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{3} y_{i,k} \log(\hat{y}_{i,k} + \epsilon)$$

---

### DG 任务 2 解答
#### 1. 首先推导各层之间的误差项
- **输出层误差项** (Softmax + 交叉熵): $\delta_2 = \hat{y} - y = A_2 - y$
 ```python
 d_2 = a[2] - res
 ```
- **隐藏层 2 误差项** (ReLU): $\delta_1 = \delta_2 \cdot W_2^T \cdot \sigma'(Z_1) = \delta_2 \cdot W_2^T \cdot \mathbb{I}(Z_1 > 0)$
 ```python
 relu_d = np.where(z[1] > 0, 1, 0)
 d_1 = np.dot(d_2, w[2].T) * relu_d
 ```
- **隐藏层 1 误差项** (Tanh): $\delta_0 = \delta_1 \cdot W_1^T \cdot \sigma'(Z_0) = \delta_1 \cdot W_1^T \cdot (1 - A_0^2)$
 ```python
 tanh_d = 1 - np.square(a[0])
 d_0 = np.dot(d_1, w[1].T) * tanh_d
 ```

---

### DG 任务 2 解答
#### 2. 各参数的梯度公式（核心）
###### (1) $W_2$ (w[2]) 的梯度: $\nabla_{W_2} Loss = \frac{1}{m} A_1^T \cdot \delta_2$

代码中的权重更新: $W_2 = W_2 - \eta \cdot \frac{1}{m} A_1^T \cdot \delta_2$

 ```python
 w[2] -= lr * np.dot(a[1].T, d_2) / l
 ```

###### (2) $b_2$ (b[2]) 的梯度: $\nabla_{b_2} Loss = \frac{1}{m} \sum_{i=1}^m \delta_{2(i)}$ (按行求和)
代码中的偏置更新: $b_2 = b_2 - \eta \cdot \frac{1}{m} \sum \delta_2$

 ```python
 b[2] -= lr * np.sum(d_2, axis=0, keepdims=True)/l
 ```

---

### DG 任务 2 解答
#### 2. 各参数的梯度公式（核心）
###### (3) $W_1$ (w[1]) 的梯度: $\nabla_{W_1} Loss = \frac{1}{m} A_0^T \cdot \delta_1$
代码中的权重更新: $W_1 = W_1 - \eta \cdot \frac{1}{m} A_0^T \cdot \delta_1$

 ```python
 w[1] -= lr * np.dot(a[0].T, d_1) / l
 ```
###### (4) $b_1$ (b[1]) 的梯度: $\nabla_{b_1} Loss = \frac{1}{m} \sum_{i=1}^m \delta_{1(i)}$
代码中的偏置更新: $b_1 = b_1 - \eta \cdot \frac{1}{m} \sum \delta_1$

 ```python
 b[1] -= lr * np.sum(d_1, axis=0, keepdims=True)/l
 ```

---

### DG 任务 2 解答
#### 2. 各参数的梯度公式（核心）
###### (5) $W_0$ (w[0]) 的梯度: $\nabla_{W_0} Loss = \frac{1}{m} X^T \cdot \delta_0$
代码中的权重更新: $W_0 = W_0 - \eta \cdot \frac{1}{m} X^T \cdot \delta_0$

 ```python
 w[0] -= lr * np.dot(data.T, d_0) / l
 ```

###### (6) $b_0$ (b[0]) 的梯度: $\nabla_{b_0} Loss = \frac{1}{m} \sum_{i=1}^m \delta_{0(i)}$
代码中的偏置更新: $b_0 = b_0 - \eta \cdot \frac{1}{m} \sum \delta_0$

 ```python
 b[0] -= lr * np.sum(d_0, axis=0, keepdims=True)/l
 ```

---

## 08 神经网络超参数的互动探索
##### 在 TensorFlow Playground 上自由探索
- TensorFlow Playground 提供了一个可视化的互动环境，用于实验核心的神经网络参数，架起了理论与实践之间的桥梁。

- 通过自由修改激活函数、调整学习率数值，并实时观察梯度下降的轨迹，你可以直接见证每一个选择如何改变收敛速度、损失降低情况以及模型的稳定性。

- 这种动手探索能够建立起对超参数相互作用的直观理解，这是静态学习无法复制的。