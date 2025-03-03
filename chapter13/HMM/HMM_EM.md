## **E步**

的核心任务是计算每个数据点属于每个隐状态的后验概率（即责任度）。在此代码中，这部分是通过调用 hmmSmoother 函数实现的。具体而言，hmmSmoother 使用前向后向算法来计算这些责任度。

#### 前向算法（Alpha）

前向算法计算的是每个时刻 $$t$$ 的状态 $$z_t = k$$ 的前向概率 $$\alpha_k(t)$$，表示在给定观测数据 $$x_1, x_2, \dots, x_t$$ 后，系统处于隐状态 $$k$$ 的概率。

具体来说，前向概率递推公式为：

$$\alpha_k(t) = P(x_1, x_2, \dots, x_t, z_t = k \mid \theta) = \left( \sum_{i=1}^K \alpha_i(t-1) A_{ik} \right) E_{k}(x_t)$$

其中：

- $$\alpha_k(t)$$ 是状态 $$k$$ 在时刻 $$t$$ 的前向概率。
- $$A_{ik}$$是从状态 $$i$$ 到状态 $$k$$ 的转移概率。
- $$E_k(x_t)$$ 是在状态 $$k$$ 时观测到 $$x_t$$ 的概率。

在矩阵乘法 $$M = E * X$$ 中，每个元素 $$M(k, i)$$是通过计算矩阵 $$E$$ 的第 $$k$$ 行与矩阵 $$X$$ 的第 $$i$$ 列的点积来得到的。即：

$$M(k, i) = \sum_{j=1}^{d} E(k, j) \cdot X(j, i)$$这意味着，对于每个时间步$$ i$$，我们根据当前隐状态$$k$$ 和观测符号的概率分布，计算生成该观测的概率。

```matlab
alpha = zeros(K,T);
[alpha(:,1), c(1)] = normalize(s .* M(:,1), 1);
for t = 2:T
    [alpha(:,t), c(t)] = normalize((At * alpha(:,t-1)) .* M(:,t), 1);  % 13.59
end
```



#### 后向算法（Beta）

后向算法计算的是每个时刻 $$t$$ 的状态 $$z_t = k$$的后向概率$$\beta_k(t)$$，表示在给定观测数据$$x_{t+1}, x_{t+2}, \dots, x_T$$后，系统处于隐状态 $$k$$ 的概率。

后向概率的递推公式为：

$$\beta_k(t) = P(x_{t+1}, x_{t+2}, \dots, x_T \mid z_t = k, \theta) = \sum_{i=1}^K A_{ki} E_i(x_{t+1}) \beta_i(t+1)$$

其中：

- $$\beta_k(t)$$ 是状态 $$k$$ 在时刻 $$t$$ 的后向概率。

```matlab
beta = ones(K,T);
for t = T-1:-1:1
    beta(:,t) = A * (beta(:,t+1) .* M(:,t+1)) / c(t+1);   % 13.62
end

```

#### 计算责任度（Gamma）

责任度 $$\gamma(z_t = k \mid x)$$ 表示在给定观测数据的情况下，时刻 $$t$$ 时系统处于状态 $$k$$的后验概率。责任度由前向后向算法的结果 $$\alpha$$ 和 $$\beta$$ 计算得到，公式如下：

$$\gamma_k(t) = P(z_t = k \mid x) = \frac{\alpha_k(t) \beta_k(t)}{\sum_{i=1}^K \alpha_i(t) \beta_i(t)}$$

```matlab
gamma = alpha .* beta;  % 13.64
```

## M步（最大化步骤）

**M步**的任务是更新模型的参数，最大化期望步骤中计算出的责任度矩阵 $$\gamma$$。

#### 更新初始状态概率 $$s$$

初始状态概率 $$s_k$$ 更新公式为：

$$s_k = \gamma_k(1)$$

```matlab
s = gamma(:,1);  % 13.18
```

#### **更新状态转移矩阵 $$A$$**：

$$A_{ik} = \frac{\sum_{t=1}^{T-1} \gamma_i(t) A_{ik} \beta_k(t+1) M_{k,t+1}}{\sum_{t=1}^{T-1} \gamma_i(t)}$$

其中，$$M_{k,t+1}$$ 是给定时间点 $$t+1$$的观测概率。

```matlab
A = normalize(A .* (alpha(:,1:n-1) * (beta(:,2:n) .* M(:,2:n) ./ c(2:n))'), 2);  % 13.19 13.43 13.65

```

#### **更新观测矩阵 $$E$$**：

$$E_k = \frac{\sum_{t=1}^{T} \gamma_k(t) x_t}{\sum_{t=1}^{T} \gamma_k(t)}$$