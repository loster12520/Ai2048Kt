# 数学公式与说明

本项目涉及的主要数学公式如下，涵盖损失函数、激活函数、优化器、初始化器、学习率调度器等。

---
## 损失函数 Loss

### 均方误差 MSE
公式：
$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$

### 平均绝对误差 MAE
公式：
$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|
$$

### Huber Loss
公式：
$$
L_\delta(y, \hat{y}) = \begin{cases}
    \frac{1}{2}(y - \hat{y})^2, & \text{if } |y - \hat{y}| \leq \delta \\
    \delta(|y - \hat{y}| - \frac{1}{2}\delta), & \text{otherwise}
\end{cases}
$$

---
## 激活函数 Activation

### ReLU
公式：
$$
\text{ReLU}(x) = \max(0, x)
$$

### Sigmoid
公式：
$$
\text{Sigmoid}(x) = \frac{1}{1 + e^{-x/\text{zoom}}}
$$

### LeakyReLU
公式：
$$
\text{LeakyReLU}(x) = \begin{cases}
    x, & x > 0 \\
    \alpha x, & x \leq 0
\end{cases}
$$

### SoftPlus
公式：
$$
\text{SoftPlus}(x) = \log_{b}(1 + b^x)
$$

---
## 优化器 Optimizer

### 梯度下降 Gradient Descent
公式：
$$
\theta = \theta - \eta \nabla_\theta J(\theta)
$$

### Momentum
公式：
$$
\begin{align*}
    v &= \beta v + (1-\beta) \nabla_\theta J(\theta) \\
    \theta &= \theta - \eta v
\end{align*}
$$

### Adam
公式：
$$
\begin{align*}
    v_t &= \beta_1 v_{t-1} + (1-\beta_1) g_t \\
    s_t &= \beta_2 s_{t-1} + (1-\beta_2) g_t^2 \\
    \hat{v}_t &= \frac{v_t}{1-\beta_1^t} \\
    \hat{s}_t &= \frac{s_t}{1-\beta_2^t} \\
    \theta_t &= \theta_{t-1} - \eta \frac{\hat{v}_t}{\sqrt{\hat{s}_t} + \epsilon}
\end{align*}
$$

---
## 初始化器 Initialize

### Xavier Uniform
公式：
$$
W \sim U\left(-\sqrt{\frac{6}{fan_{in} + fan_{out}}}, \sqrt{\frac{6}{fan_{in} + fan_{out}}}\right)
$$

### Xavier Normal
公式：
$$
W \sim N\left(0, \sqrt{\frac{2}{fan_{in} + fan_{out}}}\right)
$$

### He Uniform
公式：
$$
W \sim U\left(-\sqrt{\frac{6}{fan_{in}}}, \sqrt{\frac{6}{fan_{in}}}\right)
$$

### He Normal
公式：
$$
W \sim N\left(0, \sqrt{\frac{2}{fan_{in}}}\right)
$$

---
## 学习率调度器 Scheduler

### Step Decay
公式：
$$
\eta = \frac{\eta_0}{1 + epoch \times dropRate}
$$

### Exponential Decay
公式：
$$
\eta = \eta_0 \times (dropRate)^{epoch}
$$

---

如需补充其它公式或说明，请告知。

