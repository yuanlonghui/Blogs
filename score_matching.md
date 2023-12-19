# Score Matching

- [Home](./README.md)

## Generative Models

现存的生成模型组要可以分为两种：

- likelihood-based models：通过极大似然估计直接学习分布的概率密度函数 $p(x)$，包括 自回归模型（autoregressive models），标准化流模型（normalizing flow models），能量模型（energy-based models），变分自编码器（ variational auto-encoders）
- implicit generative models：概率分布被模型采样过程隐式表达，典型例子是 对抗生成网络（GAN）。

上述方法各有优劣，不赘述，今天主要聚焦于从 likelihood-based models 遇到的问题引出 score function，给出几种等价的学习 score function 的方法，并且介绍 DSM 和 DDPM 的联系。

## Likelihood-based Models

从一个问题出发：给定训练集 $\{x_i\}_{i=1}^N$，拟合其数据分布 $p(x)$。
采用极大似然估计的方式，我们大概可以分为以下几个步骤：

- 定义一个模型 $q(\cdot;\theta): \mathbb{R}^d\to\mathbb{R}^{+}$，并且 $q(x;\theta)$ 越大那么 $p(x)$ 越大。为了使得概率密度积分等于 $1$，我们可以定义
$$p(x;\theta)=\frac{1}{Z(\theta)}q(x;\theta), Z(\theta)=\int_x q(x;\theta)dx$$
- 采用极大似然估计的方式优化模型参数。

**步骤一：定义模型**

为了得到 $\mathbb{R}^d\to\mathbb{R}^{+}$，我们可以先用一个网络完成 $\mathbb{R}^{d}\to\mathbb{R}$，再通过 $\exp$ 运算完成 $\mathbb{R}\to\mathbb{R}^{+}$，具体来说，构建模型 $E(\cdot;\theta):\mathbb{R}^{d} \to \mathbb{R}$，定义 $q(x;\theta)=\exp\big(-E(x;\theta)\big)$，根据上面给出公式我们有：
$$
p(x;\theta) = \frac{\exp\big(-E(x;\theta)\big)}{Z(\theta)},Z(\theta)=\int_x \exp\big(-E(x;\theta)\big) dx\,. 
$$

实际上上面的选择就是能量模型的建模过程。$E(\cdot;\theta)$ 也称之为能量函数，其意义是，能量越低，样本越合乎真实分布。

<!-- 
能量模型是一种描述样本符合真实分布的程度的模型，一般来说用 $E(\cdot;\theta):\mathbb{R}^{d} \to \mathbb{R}$ 表示。其意义是，能量越低，样本越合乎真实分布。能量函数可以看作是一个未归一化的概率，样本的能量越低，似然越高。具体来说，我们可以用以下公式转换为概率密度：
$$
p(x) = \frac{\exp(-E(x))}{Z},Z=\int_x \exp(-E(x)) dx\,. 
$$

一般来说，我们会用一个网络去建模能量函数，也就是 $E(x;\theta)$，这时候我们有：
 -->

**步骤二：训练**

采用极大似然估计进行训练，损失函数被定义为：
$$
\mathcal{L}_{nll} = \sum_{i} - \log p(x_i;\theta) = \sum_{i} \left(\log \sum_{j} \exp(-E(x_j;\theta)) + E(x_i; \theta) \right)
$$

我们可以看到，其中的第一项需要对整个数据集进行计算。

*当然采用采样的方式也能够完成这一项的估计，但是为了估计足够准确，采样数量需要足够，计算仍然很困难*

**问题：**

简单来说，对于一个未归一化的概率模型 $q(x;\theta)$ 来说，我们可以通过归一化得到概率密度：
$$
p(x;\theta) = \frac{1}{Z(\theta)}q(x;\theta), Z(\theta) = \int_x q(x;\theta) dx\,.
$$

而这时想要通过极大似然估计去估计 $\theta$ 将会面临着 $Z(\theta)$ 难以计算的问题。

而 score matching 则是想要从另一个角度解决这个问题。

## Score Matching
所谓的 Score Function (记为：$s(x)$ ) 实际上是对数似然对样本的梯度，即
$$
s(x) = \triangledown_x \log p(x)
$$
那 Score Matching 也就是去建模 Score Function.

需要明确的两点是：
- Score Matching 需要解决前面提到的 $Z(\theta)$ 难以计算的问题
- Score Matching 在学习完 Score Function 之后，能够采样生成样本，这是初衷。

**$Z(\theta)$ 难以计算**

在给定 $p(x;\theta)$ 之后，当计算 score function 的时候，即 $s(x;\theta)$，我们可以发现：
$$
s(x;\theta) = \triangledown_x \log p(x;\theta) = \triangledown_x \log q(x;\theta) - \triangledown_x Z(\theta) = \triangledown_x \log q(x;\theta)
$$
这直接避开了难以计算的 $Z(\theta)$ 这一项。

值得指出的是：
- 可以使用一个 $q(x;\theta)$ 网络作为唯一的网络，然后用其对 $x$ 求梯度得到 $s(x;\theta)$，然后采用后面介绍的一些 score matching 的方法去学习 $q(x;\theta)$。
- 也可以直接使用一个 vector-valued 网络直接作为 $s(x;\theta)$。

**采样**

一旦我们训练了好的一个 score function，即 $s(x;\theta)\approx\triangledown_x \log p(x)$，我们可以使用 朗之万动力学（Langevin Dynamics）生成样本。

<!-- 那么为什么要 score function，一方面大名鼎鼎的 stochastic gradient langevin dynamic (SGLD) 可以通过 score function 从噪声生成真实样本： -->
<!-- Langevin Dynamics (Stochastic Gradient Langevin Dynamic (SGLD)) 提供了一个 MCMC （Markov chain Monte Carlo） 过程，其仅仅使用 score function，也就是$\triangledown_x \log p(x)$ 来完成从分布 $p(x)$ 的采样。 -->
Langevin Dynamics 仅仅使用 score function，也就是$\triangledown_x \log p(x)$ 来完成从分布 $p(x)$ 的采样。
具体来说，它首先从任意先验分布初始化 $x_0\sim\pi(x)$，接着通过以下迭代方式生成样本
$$
x_{t+1} = x_t + \frac{\epsilon}{2}\triangledown_x \log p(x) + \sqrt{\epsilon} z,z\sim \mathcal{N}(0, \mathbf{I})
$$
其中 $\epsilon$ 是步长，在 $\epsilon$ 足够小，$t$ 足够大的时候，认为样本 $x_{t}$ 是从 $p(x)$ 中采样得到。

下图展示了二维的 Langevin Dynamics 的可视化：

![ld](./src/score_matching/langevin.gif)

注意到 Langevin Dynamics 只用到了 score function 进行采样。当我们得到 $s(x;\theta)\approx\triangledown_x \log p(x)$ 之后，直接将 $s(x;\theta)$ 代入采样即可。

### Explicit Score Matching [1]
为了学习 score function，最简单的做法是直接拟合对数似然的梯度：
$$
J_{ESM}(\theta) =  \mathbb{E}_{p} \left[\frac{1}{2}\|s(x;\theta) - s(x)\|^2 \right] \\
= \mathbb{E}_{p} \left[\frac{1}{2}\|s(x;\theta) - \triangledown_x \log p(x)\|^2 \right] \\
% =\mathbb{E}_{p(x)} \left[\frac{1}{2}\|\triangledown_x \log q(x;\theta) - \triangledown_x \log p(x)\|^2 \right] \\
$$
虽然 $s(x;\theta)$ 去除了难以计算的 $Z(\theta)$。但是显然，在没有 $p(x)$ 的解析式前提下，$J_{ESM}(\theta)$ 也是无法计算的，我们仍然无法优化网络结构。

虽然目前不知道如何优化网络，不过我们可以知道的是，假设网络 $s(x;\theta)$ 能力足够，那么其最优解：
$$
\theta^* = \argmin_\theta J_{ESM} (\theta)
$$
应该对于 $x\in\mathbb{R}^d$，几乎处处满足 $s(x;\theta^*)=\triangledown_x\log p(x)$。（对于平方和中的每一项，想要取得最小值0，只有每一项都取0，由于是积分，所以可以存在一些点不满足）

### Implicit Score Matching [1]
虽然 $J_{ESM}(\theta)$ 无法直接优化，但是其中的 $\triangledown_x \log p(x)$ 是与 $\theta$ 无关的。通过一定的变换，我们可以将 $J_{ESM}(\theta)$ 转换为可优化的形式，也就是隐式的 score matching 损失函数，具体推导过程如下：
$$
\begin{align}
J_{ESM}(\theta) &= \mathbb{E}_p \left[\frac{1}{2}\|s(x;\theta) - \triangledown_x \log p(x)\|^2 \right] \nonumber \\
&= \int_x p(x) \left[\frac{1}{2}\|s(x;\theta) - \triangledown_x \log p(x)\|^2 \right] dx \nonumber \\
&= \int_x p(x) \left[\frac{1}{2}(s(x;\theta))^2 + \frac{1}{2}(\triangledown_x \log p(x))^2 - s(x;\theta)^\top\triangledown_x \log p(x) \right] dx \nonumber \\
&= \int_x p(x) \left[\frac{1}{2}(s(x;\theta))^2 - \frac{1}{p(x)}s(x;\theta)^\top\triangledown_x p(x) \right] dx + C_1 \nonumber \\
&= \int_x p(x) \frac{1}{2}(s(x;\theta))^2 dx - \int_x s(x;\theta)^\top\triangledown_x p(x) dx + C_1 \nonumber \\
&= \mathbb{E}_p \left[\frac{1}{2}\|s(x;\theta)\|^2\right] - \int_x s(x;\theta)^\top\triangledown_x p(x) dx + C_1 \nonumber \\
\end{align}
$$

我们需要重点关注的是第二项，继续变换：
$$
\begin{align}
\int_x s(x;\theta)^\top\triangledown_x p(x) dx &= \int_x \sum_i [s(x;\theta)]_i \frac{\partial p(x)}{\partial x_i} dx \nonumber \\
&=  \sum_i \int_{x_{\sim i}} \int_{x_i} [s(x;\theta)]_i \frac{\partial p(x)}{\partial x_i} dx_i dx_{\sim i} \nonumber \\
&=  \sum_i \int_{x_{\sim i}} \underset{\text{assumed to be }0}{\underbrace{[s(x;\theta)]_i p(x)|_{x_i=-\infty}^{\infty}}} - \int_{x_i} p(x) \frac{\partial [s(x;\theta)]_i}{\partial x_i} dx_i dx_{\sim i} \nonumber \\
&=  -  \sum_i \int_{x_{\sim i}} \int_{x_i} p(x) \frac{\partial [s(x;\theta)]_i}{\partial x_i} dx_i dx_{\sim i} \nonumber \\
&=  - \int_{x} p(x) \sum_i\frac{\partial [s(x;\theta)]_i}{\partial x_i} dx \nonumber \\
&=  - \int_{x} p(x) \text{tr}\left[\underset{\text{Hessian}}{\underbrace{\triangledown_x s(x;\theta)}} \right] dx \nonumber \\
&=  - \mathbb{E}_p \left[\text{tr}\left(\underset{\text{Hessian}}{\underbrace{\triangledown_x s(x;\theta)}} \right)\right] \nonumber \\
\end{align}
$$

于是 $J_{ESM}$ 可以转换成：
$$
\begin{align}
J_{ESM}(\theta) &= \mathbb{E}_p \left[\frac{1}{2}\|s(x;\theta)\|^2\right] + \mathbb{E}_p \left[\text{tr}\left({\triangledown_x s(x;\theta)} \right)\right] + C_1 \nonumber \\
&= \mathbb{E}_p \left[\text{tr}\left({\triangledown_x s(x;\theta)} \right) + \frac{1}{2}\|s(x;\theta)\|^2\right] + C_1 \nonumber \\
\end{align}
$$

于是隐式的 score matching 的目标函数被定义为：
$$
J_{ISM}(\theta) = \mathbb{E}_p \left[\text{tr}\left({\triangledown_x s(x;\theta)} \right) + \frac{1}{2}\|s(x;\theta)\|^2\right]
$$
可以看到 $J_{ESM}(\theta) = J_{ISM}(\theta) + C_1$，也就是说优化 $J_{ESM}$ 和 优化$J_{ISM}$ 是等价的。

通过 $J_{ISM}$，我们不需要知道 $p(x)$ 的具体是什么分布，我们就可以优化 $s(x;\theta)$。

值得注意的是，前面的推导过程涉及到了以下几个假设：

- $p(x)$，$s(x;\theta)$ 可微
- $\mathbb{E}_p \left[\|\triangledown_x \log p(x)\|^2\right]$ 有界
- $\forall \theta, \mathbb{E}_p \left[\|s(x;\theta)\|^2\right]$ 有界
- $\lim_{\|x\|\to\infty}p(x)s(x;\theta) = 0$ 

$J_{ISM}(\theta)$ 的采样形式可以表达为：
$$
\hat{J}_{ISM}(\theta) = \frac{1}{N}\sum_{i=1}^N \left[\text{tr}\left({\triangledown_x s(x_i;\theta)} \right) + \frac{1}{2}\|s(x_i;\theta)\|^2\right]
$$

### Slice Score Matching [2]

虽然 $J_{ISM}$ 提供了一种可行的优化方式，但是注意到其中有一项是计算 Hessian 矩阵（只计算对角线即可），这涉及到需要多次求梯度。当数据维度很高，例如图像，语音等可能成千上万个维度，则需要成千上万次求梯度，这些显然是不合理的。

于是 Song et al. 提出了 SSM，其损失函数被定义为：
$$
J_{SSM}(\theta) = \mathbb{E}_{p_v} \mathbb{E}_{p_x} \left[v^\top{\triangledown_x s(x;\theta)}v + \frac{1}{2}(v^\top s(x;\theta))^2\right]
$$
这时候，基于现有的自动求导，可快速计算损失函数，具体过程如图：

![SSM](./src/score_matching/SSM.png)

Song et al. 详细证明了在满足一定条件下，$J_{SSM}$ 和 $J_{ISM}$ 是等价的。

### Denoising Score Matching [3]

DSM 的优化从另一个角度出发，它受到 SM 和 denoising auto-encoder 的启发：给定 $x\sim p(x)$（分布未知），在添加特定模式的噪声之后得到 $\tilde{x}$，记 $\tilde{x}$ 的条件概率分布为 $p(\tilde{x}|x)$，$\tilde{x}$ 的真实分布 $p(\tilde{x})=\int_x p(x)p(\tilde{x}|x) dx$ 未知，DSM 的损失函数被定义为：
$$
J_{DSM_{p(\tilde{x})}}(\theta) = \mathbb{E}_{p(x, \tilde{x})}\left[\frac{1}{2}\|s(\tilde{x};\theta) - \triangledown_{\tilde{x}} \log p(\tilde{x}|x) \|^2\right]
$$

下面我们推导 $J_{DSM_{p(\tilde{x})}}$ 等价于 $J_{ESM_{p(\tilde{x})}}$：
$$
\begin{align}
J_{ESM_{p(\tilde{x})}}(\theta) &= \mathbb{E}_{p(\tilde{x})} \left[\frac{1}{2}\|s(\tilde{x},;\theta) - \triangledown_{\tilde{x}} \log p(\tilde{x})\|^2 \right] \nonumber \\
&= \mathbb{E}_{p(\tilde{x})} \left[\frac{1}{2}\|s(\tilde{x};\theta)\|^2 \right] - S(\theta) + C_1 \nonumber \\
S(\theta) &= \mathbb{E}_{p(\tilde{x})}\left[\left<s(\tilde{x};\theta),\triangledown_{\tilde{x}} \log p(\tilde{x})\right>\right] \nonumber \\
&= \int_{\tilde{x}} p(\tilde{x})\left<s(\tilde{x};\theta),\triangledown_{\tilde{x}} \log p(\tilde{x})\right> d\tilde{x} \nonumber \\
&= \int_{\tilde{x}} p(\tilde{x})\left<s(\tilde{x};\theta), \frac{1}{p(\tilde{x})}\triangledown_{\tilde{x}} p(\tilde{x})\right> d\tilde{x} \nonumber \\
&= \int_{\tilde{x}} \left<s(\tilde{x};\theta), \triangledown_{\tilde{x}} p(\tilde{x})\right> d\tilde{x} \nonumber \\
&= \int_{\tilde{x}} \left<s(\tilde{x};\theta), \triangledown_{\tilde{x}} \int_x p(x)p(\tilde{x}|x)dx\right> d\tilde{x} \nonumber \\
&= \int_{\tilde{x}} \left<s(\tilde{x};\theta), \int_x p(x) \triangledown_{\tilde{x}} p(\tilde{x}|x)dx\right> d\tilde{x} \nonumber \\
&= \int_{\tilde{x}} \left<s(\tilde{x};\theta), \int_x p(x) p(\tilde{x}|x) \triangledown_{\tilde{x}} \log p(\tilde{x}|x) dx \right> d\tilde{x} \nonumber \\
&= \int_{\tilde{x}} \int_x p(x) p(\tilde{x}|x) \left<s(\tilde{x};\theta),  \triangledown_{\tilde{x}} \log p(\tilde{x}|x)\right>  dx d\tilde{x} \nonumber \\
&= \mathbb{E}_{p(x, \tilde{x})} \left[\left<s(\tilde{x};\theta),  \triangledown_{\tilde{x}} \log p(\tilde{x}|x)\right>\right] \nonumber \\
\Longleftrightarrow J_{ESM_{p(\tilde{x})}}(\theta) &= \mathbb{E}_{p(\tilde{x})} \left[\frac{1}{2}\|s(\tilde{x};\theta)\|^2 \right] - \mathbb{E}_{p(x, \tilde{x})} \left[\left<s(\tilde{x};\theta),  \triangledown_{\tilde{x}} \log p(\tilde{x}|x)\right>\right]  + C_1 \nonumber \\
J_{DSM_{p(\tilde{x})}}(\theta) &= \mathbb{E}_{p(x, \tilde{x})}\left[\frac{1}{2}\|s(\tilde{x};\theta) - \triangledown_{\tilde{x}} \log p(\tilde{x}|x) \|^2\right] \nonumber \\
&= \mathbb{E}_{p(x, \tilde{x})}\left[\frac{1}{2}\|s(\tilde{x};\theta)\|^2\right] - \mathbb{E}_{p(x, \tilde{x})} \left[\left<s(\tilde{x};\theta),  \triangledown_{\tilde{x}} \log p(\tilde{x}|x)\right>\right] + C_2 \nonumber \\
&= \mathbb{E}_{p(\tilde{x})}\left[\frac{1}{2}\|s(\tilde{x};\theta)\|^2\right] - \mathbb{E}_{p(x, \tilde{x})} \left[\left<s(\tilde{x};\theta),  \triangledown_{\tilde{x}} \log p(\tilde{x}|x)\right>\right] + C_2 \nonumber \\
&= J_{DSM_{p(\tilde{x})}}(\theta) - C_1 + C_2 \nonumber \\
\end{align}
$$
可知，两者等价。也就是说最优解 $\theta^* = \argmin_\theta J_{DSM_{p(\tilde{x})}}(\theta)$，对于 $\tilde{x} \in \mathbb{R}^d$，几乎处处满足 $s(\tilde{x};\theta^*)=\triangledown_{\tilde{x}} \log p(\tilde{x})$。

显然，给与固定模式的噪声，$p(\tilde{x}|x)$ 是可以计算的。通过优化 $J_{DSM_{p(\tilde{x})}}$，在 $p(x)$ 和 $p(\tilde{x})$ 都未知的情况下，我们居然可以估计出 $\triangledown_{\tilde{x}} \log p(\tilde{x})$。(amazing!)

举个栗子：$\tilde{x} = x + \sigma\epsilon, \epsilon \sim \mathcal{N}(0,\mathbf{I})$，显然此时 $p(\tilde{x}|x)=\mathcal{N}(\tilde{x};x,\sigma^2\mathbf{I})$，此时:
$$
J_{DSM}(\theta) = \mathbb{E}_{x,\epsilon} \left[\frac12\|s(x+\sigma\epsilon;\theta) - \triangledown_{\tilde{x}} \log p(\tilde{x}|x)\|^2\right] \\
= \mathbb{E}_{x,\epsilon} \left[\frac12\|s(x+\sigma\epsilon;\theta) - \frac{x - \tilde{x}}{\sigma^2}\|^2\right] \\
= \mathbb{E}_{x,\epsilon} \left[\frac12\|s(x+\sigma\epsilon;\theta) - \frac{-\epsilon}{\sigma}\|^2\right]
$$
可以看到，$\frac{-\epsilon}{\sigma}$ 恰好是去除噪声的方向，这也是为什么称之为 Denoising Score Matching。
优化上述损失，最终得到 $s(\tilde{x};\theta)\approx\triangledown_{\tilde{x}} \log p(\tilde{x})$。（ps：有没有感觉很像 DDPM）

## Noise Conditional Score Network (NCSN) [4]

<!-- 先介绍基于 Score Function 的生成模型。再提 Stochastic Gradient Langevin Dynamic (SGLD)，在 Score Function 已知的情况下，我们可以通过以下的迭代式从一个随机噪声生成样本：
$$
x_{t+1} = x_t + \frac{\epsilon}{2}\triangledown_x \log p(x) + \sqrt{\epsilon} z_t, z_t \sim \mathcal{N}(0, \mathbf{I})
$$ -->

通过前面介绍的估计 Score Matching 方法，我们可以估计 $\triangledown_x \log p(x)$。

但是无论是 ESM，ISM，SSM 都会面领着以下几个问题：

- 流形假设：现实世界中的数据往往集中在嵌入在高维空间（即环境空间）中的低维流形上。
- - $\triangledown_x \log p(x)$ 在低维流形之外是未定义的。
- - Score Matching 目标方程仅当数据分布的支撑是整个空间时才提供一致的Score Function Estimator，当数据驻留在低维流形上时，将不一致。

    **流行假设 分析实验**

    ![toy experiment 1](./src/score_matching/toy_exp_1.png)

    使用 SSM 训练一个 ResNet 作为 $s(x;\theta)$，分别在原始 CIFAR-10 数据集 和 在 CIFAR-10 上加入随机噪声 $\mathcal{N}(0, 0.0001)$ 形成的数据集 进行训练。我们认为加噪之后的分布的概率密度的支撑集充满了整个 $\mathbb{R}^d$。结果如上图所示，直接训练，当数据限制在低维流形上的时候，难以收敛。而当支撑集充满整个空间的时候，SSM 的损失函数最终能够收敛。
- 低密度区域的数据稀缺可能会导致 分数估计、分数匹配 和 使用朗之万动力学采样 困难。
- - Score Matching 在低密度区域不准确。
- - Langevin Dynamics 无法区分分布的混合。

    **低密度区域拟合不准分析实验**

    上面介绍的几种 score matching 的方法的损失函数都是以期望形式出现的，也就是说对于低概率密度区域，其训练权重将会非常低，在真实应用时（网络能力不够）这些区域往往不能很好的拟合真正的 score。

    ![toy experiment 2](./src/score_matching/toy_exp_2.png)

    实验一：给定 $p_{data} = \frac15\mathcal{N}([-5,-5]^\top, \mathbf{I}) + \frac45\mathcal{N}([5,5]^\top, \mathbf{I})$，直接采用 ESM 拟合 score，结果如上图所示。可以看到，除去红框标注的高密度区域，剩余的区域 score 预测的都不准确。

    另一方面：我们假设有一个分布被表示为 $p_{data} = \pi p_1(x) + (1-\pi)p_2(x), \pi \in [0,1]$，并且我们假设 $p_1(x)$ 和 $p_2(x)$ 的支撑集交集为空。这个时候在 $p_1(x)$ 支撑集上，$\triangledown_x \log p_{data}(x) = \triangledown_x \log \pi p_1(x) = \triangledown_x \log p_1(x)$，而在 $p_2(x)$ 的支撑集上时，$\triangledown_x \log p_{data}(x) = \triangledown_x \log (1-\pi)p_2(x) = \triangledown_x \log p_2(x)$。也就是说这时候 $\triangledown_x \log p_{data}(x)$ 根本不取决于 $\pi$ 的取值。这时候采用 Langevin dynamics 采样在理论上就会出现错误。

    上述分析中支撑集交集为空的假设可能过于严格，现实场景中往往是共享同一个支撑集，但是不同分布的高概率密度区域被低概率密度区域分隔。
    例如我们采用前面的例子 $p_{data} = \frac15\mathcal{N}([-5,-5]^\top, \mathbf{I}) + \frac45\mathcal{N}([5,5]^\top, \mathbf{I})$，分别真实 i.i.d. 采样 和 使用真实的 score function 进行 Langevin Dynamic 采样。结果如图下图所示，如何前面的分析。
    ![toy experiment 3](./src/score_matching/toy_exp_3.png)

总的来说，直接使用 Score Matching 然后采用 Langevin Dynamic 进行采样，作为生成模型还是会遇到一些问题。

**观察与分析**

- 由于流行假设，直接在原始数据上做 score matching 不能准确估计真正的 score function。
- 通过加入噪声，可以将真实分布的支撑集从低维流形扩散至整个空间。
- LD 往往对于分布的混合是不敏感的，如果概率分布是类似分析实验中的分布，LD 进行采样将会出错。

那么基于上述几点，NCSN 采用的思想是：通过给原始数据分布加入从小到大的高斯噪声，形成从低维流形到充满整个空间的不同的加噪数据分布。接着采用 Langevin Dynamic 从噪声大的分布慢慢生成噪声小的数据分布。

具体来说，给定一个序列 $\{\sigma_i\}_{i=1}^L$，满足 $\frac{\sigma_1}{\sigma_2}=\frac{\sigma_2}{\sigma_3}=\cdots=\frac{\sigma_{L-1}}{\sigma_L}>1$。

给定 $\sigma$，$\tilde{x} = x + \sigma\epsilon, \epsilon\sim\mathcal{N}(0,\mathbf{I})$。记 $\tilde{x}$ 的数据分布为 $p_{\sigma}(\tilde{x}) = \int_x p(x)p(\tilde{x}|x)dx$，易知 $p(\tilde{x}|x) = \mathcal{N}(\tilde{x};x,\sigma^2\mathbf{I})$。
通过前面介绍的 DSM，我们可以估计到所有的 $\triangledown_{\tilde{x}} \log p_{\sigma_i}(\tilde{x})$。通过 Langevin Dynamic，我们可以生成来自 $p_{\sigma_i}(\tilde{x})$ 的样本。

于是我们可以采取以下策略：

- $\sigma_1$ 足够大，以至于数据分布可以充满整个高维空间，解决前面提到的问题
- $\sigma_L$ 足够小，这样 $p_{\sigma_L}(\tilde{x})$ 可以近似为 $p(x)$。
- 逐步从 $p_{\sigma_1}(\tilde{x})$ 到 $p_{\sigma_L}(\tilde{x})$ 生成样本，相当于采用 $L$ 次 SGLD，最终得到 $p_{\sigma_L}(\tilde{x})$ 的样本，近似真实 $p(x)$ 的样本。

我们采用 $s(x,\sigma;\theta)$ 去估计 $\triangledown_{\tilde{x}} \log p_{\sigma}(\tilde{x})$，训练的目标函数如下：
$$
\ell(\theta;\sigma)=\mathbb{E}_{p(x)}\mathbb{E}_{\tilde{x}\sim\mathcal{N}(x,\sigma^2\mathbf{I})} \left[\frac12\left\|s(\tilde{x},\sigma;\theta)-\triangledown_{\tilde{x}} \log p(\tilde{x}|x)\right\|^2\right] \\
=\mathbb{E}_{p(x)}\mathbb{E}_{\tilde{x}\sim\mathcal{N}(x,\sigma^2\mathbf{I})} \left[\frac12\left\|s(\tilde{x},\sigma;\theta)+\frac{\tilde{x}-x}{\sigma^2}\right\|^2\right]
$$

总的训练损失为：
$$
L(\theta) = \frac1L \sum_{i=1}^L \lambda(\sigma_i)\ell(\theta;\sigma_i)
$$

<!-- 假定 $s(\tilde{x},\sigma;\theta)$ 能力足够。最优解对 $\tilde{x}\in\mathbb{R}^d$ 几乎处处满足 $s(\tilde{x},\sigma;\theta^*) = \triangledown_{\tilde{x}} \log p_{\sigma}(\tilde{x})$。 -->
那么优化上述目标函数，我们可以得到 $s(\tilde{x},\sigma;\theta) \approx \triangledown_{\tilde{x}} \log p_{\sigma}(\tilde{x})$。

优化目标中的 $\lambda(\sigma_i)$ 是希望对于所有的 $\sigma_i$，$\lambda(\sigma_i)\ell(\theta;\sigma_i)$ 都能够在同一数量级，而观察到对于最优解一般 $\|s(\tilde{x},\sigma;\theta)\|_2 \propto \frac1\sigma$，于是我们选择 $\lambda(\sigma_i) = \sigma_i^2$，这时候有 $\lambda(\sigma_i)\ell(\theta;\sigma_i)=\mathbb{E} \left[\frac12\left\|\sigma_i s(\tilde{x},\sigma_i;\theta)+\frac{\tilde{x}-x}{\sigma_i}\right\|^2\right]$，这时候 $\|\sigma_i s(\tilde{x},\sigma_i;\theta)\| \propto 1, \|\frac{\tilde{x}-x}{\sigma_i}\|\propto 1$，对于所有的 $\sigma_i$，$\lambda(\sigma_i)\ell(\theta;\sigma_i)$ 都能够处于同一数量级。

对于采样的过程，按照我们前面提到的方式进行采样：

![Annealed Langevin Dynamic](./src/score_matching/ALD.png)

ALD 的一个二维可视化如下图，（图中 $\sigma$ 关系和上面描述的反过来）：

![ald](./src/score_matching/ald.gif)

### Connection with DDPM

在 DDPM 的前向过程中，我们知道：
$$
x_{t} = \sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon, \epsilon \sim \mathcal{N}(0,\mathbf{I})
$$
我们也可用基于 DSM 的方式去建模 $\triangledown_{x_{t}}\log p(x_{t}|x_0)$，其损失函数为：
$$
\mathcal{L} = \mathbb{E}_{t\sim U(1, T)}\mathbb{E}_{x_0 \sim q_0(x_0)}\mathbb{E}_{x_t\sim q(x_t| x)} \lambda(t) \|s(x_t,t) - \triangledown_{x_{t}}\log p(x_{t}|x_0)\|^2
$$

根据前面的重参数化技巧我们知道：
$$
\triangledown_{x_{t}}\log p(x_{t}|x_0) = - \triangledown_{x_{t}} \frac{(x_t - \sqrt{\bar{\alpha}_t}x_0)^2}{2 (1-\bar{\alpha}_t)} = - \frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}
$$

损失函数变为：
$$
\mathcal{L} = \mathbb{E}_{t\sim U(1, T)}\mathbb{E}_{x_0 \sim q_0(x_0)}\mathbb{E}_{\epsilon\sim \mathcal{N}(0,\mathbf{I})} \lambda(t)\Big\|[-s(\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon,t)] - \frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}\Big\|^2 \\
= \mathbb{E}_{t\sim U(1, T)}\mathbb{E}_{x_0 \sim q_0(x_0)}\mathbb{E}_{\epsilon\sim \mathcal{N}(0,\mathbf{I})} \frac{\lambda(t)}{1-\bar{\alpha}_t} \Big\|[-\sqrt{1-\bar{\alpha}_t}s(\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon,t)] - \epsilon \Big\|^2
$$

取 $\lambda(t) = 1-\bar{\alpha}_t$，有：
$$
\mathcal{L} = \mathbb{E}_{t\sim U(1, T)}\mathbb{E}_{x_0 \sim q_0(x_0)}\mathbb{E}_{\epsilon\sim \mathcal{N}(0,\mathbf{I})} \Big\|[-\sqrt{1-\bar{\alpha}_t}s(\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon,t)] - \epsilon \Big\|^2
$$

观察 DDPM 原始简化训练目标我们有：
$$
s(x,t) = -\frac{\epsilon(x,t)}{\sqrt{1-\bar{\alpha}_t}}
$$

也就是说：DDPM 中学习的噪声网络实际上就是在做 Denoising Score Matching，和前面提到的 NCSN 可以相互转换，通过 ALD 也可以完成采样，不过会更慢。

## 参考文献

[1]: [Estimation of Non-Normalized Statistical Models by Score Matching.](https://www.cs.helsinki.fi/u/ahyvarin/papers/JMLR05.pdf)

[2]: [Sliced Score Matching: A Scalable Approach to Density and Score Estimation](https://arxiv.org/pdf/1905.07088.pdf)

[3]: [A Connection Between Score Matching and Denoising Autoencoders](https://ieeexplore.ieee.org/abstract/document/6795935)

[4]: [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/pdf/1907.05600.pdf)
