# Score Matching

## Energy based model （能量模型）
我们从一个典型的未归一化的的概率模型讲起：能量模型是一种使用能量函数描述样本合乎真实分布的程度，用 $E:\mathbb{R}^{d} \to \mathbb{R}$ 表示，一般来说，能量越低，样本越合乎真实分布。能量函数可以看作是一个未归一化的概率，样本的能量越低，似然越高。具体来说，我们可以用以下公式转换为概率密度：
$$
p(x) = \frac{\exp(-E(x))}{Z},Z=\int_x \exp(-E(x)) dx\,. 
$$

一般来说，我们会用一个网络去建模能量函数，也就是 $E(x;\theta)$，这时候我们有：
$$
p(x;\theta) = \frac{\exp(-E(x;\theta))}{Z(\theta)},Z(\theta)=\int_x \exp(-E(x;\theta)) dx\,. 
$$

那训练自然离不开极大似然估计：
$$
\mathcal{L}_{nll} = \sum_{i} - \log p(x_i;\theta) = \sum_{i} \left(\log \sum_{j} \exp(-E(x_j;\theta)) + E(x_i; \theta) \right)
$$

我们可以看到，其中的第一项需要对整个数据集进行计算，并且为了估计足够准确，采样数量需要足够，这就会使得其计算更加困难。

简单来说，对于一个未归一化的概率模型 $q(x;\theta)$ 来说，我们可以通过归一化得到真正的概率密度：
$$
p(x;\theta) = \frac{1}{Z(\theta)}q(x;\theta), Z(\theta) = \int_x q(x;\theta) dx\,.
$$

而这时想要通过极大似然估计去估计 $\theta$ 将会面临着 $Z(\theta)$ 难以计算的问题。而 score matching 则是想要从另一个角度解决这个问题。

## Score Matching
所谓的 Score Function (记为：$s(x)$ ) 实际上是对数似然对样本的梯度，即
$$
s(x) = \triangledown_x \log p(x)
$$

那么为什么要 score function，一方面大名鼎鼎的 stochastic gradient langevin dynamic (SGLD) 可以通过 score function 从噪声生成真实样本：
$$
x_{t+1} = x_t + \frac{\epsilon}{2}\triangledown_x \log p(x) + \sqrt{\epsilon} z,z\sim \mathcal{N}(0, \mathbf{I})
$$

另一方面，当我们去建模 score function 的时候，即 $s(x;\theta)$，我们可以发现：
$$
s(x;\theta) = \triangledown_x \log p(x;\theta) = \triangledown_x \log q(x;\theta) - \triangledown_x Z(\theta) = \triangledown_x \log q(x;\theta)
$$
这直接避开了难以计算的 $Z(\theta)$ 这一项。

### Explicit Score Matching （显式）
最简单直接明了的做法是直接用网络拟合对数似然的梯度：
$$
J_{ESM}(\theta) =  \mathbb{E}_p \left[\frac{1}{2}\|s(x;\theta) - s(x)\|^2 \right] = \mathbb{E}_p \left[\frac{1}{2}\|s(x;\theta) - \triangledown_x \log p(x)\|^2 \right]
$$
虽然 $s(x;\theta)$ 去除了难以计算的 $Z(\theta)$。但是显然，在没有 $p(x)$ 的解析式前提下，$J_{ESM}(\theta)$ 也是无法计算的，我们仍然无法优化网络结构。

虽然目前不知道如何优化网络，不过我们可以知道的是，假设网络 $s(x;\theta)$ 能力足够，那么其最优解：
$$
\theta^* = \argmax_\theta J_{ESM} (\theta)
$$
应该对于 $x\in\mathbb{R}^d$，几乎处处满足 $s(x;\theta^*)=\triangledown_x\log p(x)$。（对于平方和中的每一项，想要取得最小值0，只有每一项都取0，由于是积分，所以可以存在一些点不满足）

### Implicit Score Matching （隐式）
实际上隐式损失函数是对显示损失函数的一个变换之后的形式，具体推导过程如下：
$$
\begin{align}
J_{ESM}(\theta) &= \mathbb{E}_p \left[\frac{1}{2}\|s(x;\theta) - \triangledown_x \log p(x)\|^2 \right] \nonumber \\
&= \int_x p(x) \left[\frac{1}{2}\|s(x;\theta) - \triangledown_x \log p(x)\|^2 \right] dx \nonumber \\
&= \int_x p(x) \left[\frac{1}{2}(s(x;\theta))^2 + \frac{1}{2}(\triangledown_x \log p(x))^2 - s(x;\theta)^\top\triangledown_x \log p(x) \right] dx \nonumber \\
&= \int_x p(x) \left[\frac{1}{2}(s(x;\theta))^2 - \frac{1}{p(x)}s(x;\theta)^\top\triangledown_x p(x) \right] dx + C_1(x) \nonumber \\
&= \int_x p(x) \frac{1}{2}(s(x;\theta))^2 dx - \int_x s(x;\theta)^\top\triangledown_x p(x) dx + C_1(x) \nonumber \\
&= \mathbb{E}_p \left[\frac{1}{2}\|s(x;\theta)\|^2\right] - \int_x s(x;\theta)^\top\triangledown_x p(x) dx + C_1(x) \nonumber \\
\end{align}
$$

我们需要重点关注的是第二项，继续变换：
$$
\begin{align}
\int_x s(x;\theta)^\top\triangledown_x p(x) dx &= \int_x \sum_i [s(x;\theta)]_i \frac{\partial p(x)}{\partial x_i} dx \nonumber \\
&=  \sum_i \int_{x_{\sim i}} \int_{x_i} [s(x;\theta)]_i \frac{\partial p(x)}{\partial x_i} dx_i dx_{\sim i} \nonumber \\
&=  \sum_i \int_{x_{\sim i}} \underset{\text{assume to be }0}{\underbrace{[s(x;\theta)]_i p(x)|_{x_i=-\infty}^{\infty}}} - \int_{x_i} p(x) \frac{\partial [s(x;\theta)]_i}{\partial x_i} dx_i dx_{\sim i} \nonumber \\
&=  -  \sum_i \int_{x_{\sim i}} \int_{x_i} p(x) \frac{\partial [s(x;\theta)]_i}{\partial x_i} dx_i dx_{\sim i} \nonumber \\
&=  - \int_{x} p(x) \sum_i\frac{\partial [s(x;\theta)]_i}{\partial x_i} dx \nonumber \\
&=  - \int_{x} p(x) \text{tr}\left[\underset{\text{Hessian}}{\underbrace{\triangledown_x s(x;\theta)}} \right] dx \nonumber \\
&=  - \mathbb{E}_p \left[\text{tr}\left(\underset{\text{Hessian}}{\underbrace{\triangledown_x s(x;\theta)}} \right)\right] \nonumber \\
\end{align}
$$

于是 $J_{ESM}$ 可以转换成：
$$
\begin{align}
J_{ESM}(\theta) &= \mathbb{E}_p \left[\frac{1}{2}\|s(x;\theta)\|^2\right] + \mathbb{E}_p \left[\text{tr}\left({\triangledown_x s(x;\theta)} \right)\right] + C_1(x) \nonumber \\
&= \mathbb{E}_p \left[\text{tr}\left({\triangledown_x s(x;\theta)} \right) + \frac{1}{2}\|s(x;\theta)\|^2\right] + C_1(x) \nonumber \\
\end{align}
$$

于是隐式的 score matching 的目标函数被定义为：
$$
J_{ISM}(\theta) = \mathbb{E}_p \left[\text{tr}\left({\triangledown_x s(x;\theta)} \right) + \frac{1}{2}\|s(x;\theta)\|^2\right]
$$
可以看到 $J_{ESM}(\theta) = J_{ISM}(\theta) + C_1(x)$，也就是说优化 $J_{ESM}$ 和 优化$J_{ISM}$ 是等价的。