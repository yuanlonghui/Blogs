# Diffusion models (扩散模型)

## TO Read
- 随机梯度Langevin动力学
- VAE
- Energy model
- SDE 

## 什么是扩散模型
扩散模型的灵感来自非平衡热力学。他们定义了扩散步骤的马尔可夫链，以缓慢地将随机噪声添加到数据中，然后学习反转扩散过程以从噪声构建所需的数据样本。
![DDPM](./src/diffusion/ddpm_framework.png)

## 前向扩散过程 (Forward diffusion process)
给定一个数据点 $x_0 \sim q(x_0)$ , 前向过程被定义为逐步（总共T步）向样本添加少量高斯噪声，产生一系列的噪声样本 $x_1, x_2, \cdots, x_T$ 。步长由方差控制 $\{\beta_t\in(0,1)\}_{t=1}^T$ ，其中有 $0<\beta_1<\beta_2<\cdots<\beta_T<1$ 。

第 $t$ 时刻的前向过程被定义为：
$$
x_t = \sqrt{1-\beta_t} x_{t-1} + \sqrt{\beta_t}\epsilon_{t-1}, \epsilon_{t-1}\sim\mathcal{N}(0, \mathbf{I}). 
$$

由于 $\epsilon_{t-1}\sim\mathcal{N}(0, \mathbf{I})$ ，所以给定 $x_{t-1}$，$x_{t}$ 的条件分布可以表示为：
$$
q(x_{t}|x_{t-1}) = \mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1}, \beta_t\mathbf{I}).
$$

根据马尔可夫链（Markov chain）的性质，给定 $x_0$ ，$x_{1:T}$ 的分布可以表示为：
$$
q(x_{1:t}|x_0) = \Pi_{t=1}^T q(x_t | x_{t-1}). 
$$

给定任意 $t\ge 1$，$x_t$ 可以通过以下方式计算，（记 $\alpha_t = 1 - \beta_t, \bar{\alpha}_t = \Pi_{i=1}^t \alpha_i$ ）：
$$
\begin{align}
x_t &= \sqrt{1-\beta_t} x_{t-1} + \sqrt{\beta_t}\epsilon \nonumber \\
&=\sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t}\epsilon_{t-1} \nonumber \\
&=\sqrt{\alpha_t} (\sqrt{\alpha_{t-1}} x_{t-2} + \sqrt{1-\alpha_{t-1}}\epsilon_{t-2}) + \sqrt{1-\alpha_t}\epsilon_{t-1} \nonumber \\
&= \sqrt{\alpha_{t}\alpha_{t-1}} x_{t-2} + \sqrt{\alpha_{t}(1-\alpha_{t-1})}\epsilon_{t-2} + \sqrt{1-\alpha_t}\epsilon_{t-1} \nonumber \\
&=\sqrt{\alpha_{t}\alpha_{t-1}} x_{t-2} + \sqrt{1-\alpha_t\alpha_{t-1}}\bar{\epsilon}_{t-2} & \divideontimes \nonumber \\
&=\sqrt{\alpha_{t}\alpha_{t-1}} (\sqrt{\alpha_{t-2}} x_{t-3} + \sqrt{1-\alpha_{t-2}}\epsilon_{t-3}) + \sqrt{1-\alpha_t\alpha_{t-1}}\bar{\epsilon}_{t-2} \nonumber \\
&=\sqrt{\alpha_{t}\alpha_{t-1}\alpha_{t-2}}  x_{t-3} + \sqrt{\alpha_{t}\alpha_{t-1}(1-\alpha_{t-2})}\epsilon_{t-3} + \sqrt{1-\alpha_t\alpha_{t-1}}\bar{\epsilon}_{t-2} \nonumber \\
&=\sqrt{\alpha_{t}\alpha_{t-1}\alpha_{t-2}}  x_{t-3} + \sqrt{1-\alpha_t\alpha_{t-1}\alpha_{t-2}}\bar{\epsilon}_{t-3} & \divideontimes \nonumber \\
& = \cdots \nonumber \\
& = \sqrt{\bar{\alpha_{t}}} x_0 + \sqrt{1-\bar{\alpha_{t}}}\epsilon \nonumber \\
\Longleftrightarrow & \quad p(x_t|x_0) = \mathcal{N}(x_t;\sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)\mathbf{I})\nonumber \\
\end{align}
$$
其中的 $\divideontimes$ 表示的是高斯噪声的合并：
$$\mathcal{N}(0,\sigma_1^2\mathbf{I}) + \mathcal{N}(0,\sigma_2^2\mathbf{I}) = \mathcal{N}(0,(\sigma_1^2 + \sigma_2^2)\mathbf{I}). $$

通常当样本变得更嘈杂时，可以添加的噪声也变得更大，于是有前面提到的 $0<\beta_1<\beta_2<\cdots<\beta_T<1$ ，同时有 $1>\bar{\alpha}_1>\bar{\alpha}_2>\cdots>\bar{\alpha}_T>0$ 。可以看到，只要加噪声步数足够多 $T\to\infty$ ，有 $\bar{\alpha}_T\to0$ ，也就是最终得到的是一个标准高斯噪声。

## 逆扩散过程 （Reverse diffusion process）
如果我们可以将前向过程反转，从 $q(x_{t-1}|x_t)$ 中逐步采样，那么我们就可以将一个给定的高斯噪声 $x_t\in \mathcal{N}(0,\mathbf{I})$ 还原成真实样本。（假设 $\beta_t$ 足够小时， $q(x_{t-1}|x_t)$ 也会是一个高斯分布。）但是估计 $q(x_{t-1}|x_t)$ 需要整个数据集，这是不容易的。于是转而去学习一个模型 $p_\theta$ 能够近似这些条件概率，以便可以完成反向过程：
$$
p_{\theta}(x_{0:T}) = p(x_t)\Pi_{t=1}^T p_{\theta}(x_{t-1}|x_t)
$$

如果将反向过程看作是一个高斯过程，那么有
$$
\begin{align}
p_{\theta}(x_{t-1}|x_t) = \mathcal{N}(x_{t-1};\mu_\theta(x_t, t),\Sigma_\theta(x_t, t)) \nonumber
\end{align}
$$

那么优化目标是什么呢？或者说如何训练呢？这里使用常用的极大似然估计法，也称之为最小化负对数似然（记 $z=x_{1:T},x=x_0$ ）：
$$
\begin{align}
\mathbb{E}_{x\sim q(x)}[-\log p_\theta(x)] &= \int_x q(x)[-\log p_\theta(x)] dx \nonumber \\
&= \int_x \int_z q(x, z) dz [-\log p_\theta(x)] dx \nonumber \\
% &= \int_x q(x) \int_z q(z|x)[-\log p_\theta(x)] dz dx \nonumber \\
&= \mathbb{E}_{x,z\sim q(x,z)}[-\log p_\theta(x)] \nonumber \\
&= \mathbb{E}_{x,z\sim q(x,z)}[-\log \frac{p_\theta(x, z)}{p_\theta(z | x)}] \nonumber \\
&= \mathbb{E}_{x,z\sim q(x,z)}[-\log \frac{p_\theta(x, z)}{q(z | x)}\frac{q(z|x)}{p_\theta(z|x)}] \nonumber \\
&= \mathbb{E}_{x,z\sim q(x,z)}[-\log \frac{p_\theta(x, z)}{q(z | x)}] - \mathbb{E}_{x,z\sim q(x,z)}[\log \frac{q(z|x)}{p_\theta(z|x)}]\nonumber \\
&= \mathbb{E}_{x,z\sim q(x,z)}[-\log \frac{p_\theta(x, z)}{q(z | x)}] - \mathbb{E}_{x\sim q(x)}[KL(q(z|x)||p_\theta(z|x))]\nonumber \\
& \le  \mathbb{E}_{x,z\sim q(x,z)}[-\log \frac{p_\theta(x, z)}{q(z | x)}] = \mathcal{L}_{EVLB}\nonumber
\end{align}
$$

优化右边的上界(从变分法看，应该有什么理论证明)，先变换变换 
$$
\begin{align}
&\mathbb{E}_{x,z\sim q(x,z)}[-\log \frac{p_\theta(x, z)}{q(z | x)}] = \mathbb{E}_{x_{0:T}}[-\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}| x_0)}]\nonumber \\
&=\mathbb{E}_{x_{0:T}}[-\log \frac{p(x_T)\Pi_{t=1}^T p_{\theta}(x_{t-1}|x_t)}{\Pi_{t=1}^T q(x_t | x_{t-1})}]\nonumber \\
&=\mathbb{E}_{x_{0:T}}[-\log p(x_T) -\sum_{t=1}^T \log \frac{p_{\theta}(x_{t-1}|x_t)}{q(x_t | x_{t-1})}]\nonumber \\
&=\mathbb{E}_{x_{0:T}}[-\log p(x_T) -\sum_{t=2}^T \log \frac{p_{\theta}(x_{t-1}|x_t)}{q(x_t | x_{t-1}, x_0)}-\log \frac{p_\theta(x_0|x_1)}{q(x_1|x_0)}]\nonumber \\
&=\mathbb{E}_{x_{0:T}}[-\log p(x_T) -\sum_{t=2}^T \log \frac{p_{\theta}(x_{t-1}|x_t) q(x_{t-1}, x_0)}{q(x_t , x_{t-1}, x_0)}-\log \frac{p_\theta(x_0|x_1)}{q(x_1|x_0)}]\nonumber \\
&=\mathbb{E}_{x_{0:T}}[-\log p(x_T) -\sum_{t=2}^T \log \frac{p_{\theta}(x_{t-1}|x_t) q(x_{t-1}, x_0)}{q(x_{t-1} | x_{t}, x_0) q(x_t, x_0)}-\log \frac{p_\theta(x_0|x_1)}{q(x_1|x_0)}]\nonumber \\
&=\mathbb{E}_{x_{0:T}}[-\log p(x_T) -\sum_{t=2}^T \log \frac{p_{\theta}(x_{t-1}|x_t) q(x_{t-1}|x_0)}{q(x_{t-1} | x_{t}, x_0) q(x_t|x_0)}-\log \frac{p_\theta(x_0|x_1)}{q(x_1|x_0)}]\nonumber \\
&=\mathbb{E}_{x_{0:T}}[-\log p(x_T) -\sum_{t=2}^T \log \frac{p_{\theta}(x_{t-1}|x_t) }{q(x_{t-1} | x_{t}, x_0)}-\log \frac{\bcancel{q(x_{T-1}|x_0)} \bcancel{q(x_{T-2}|x_0)} \cdots \bcancel{q(x_1|x_0)} p_\theta(x_0|x_1)}{q(x_{T}|x_0) \bcancel{q(x_{T-1}|x_0)} \cdots \bcancel{q(x_2|
x_0)} \bcancel{q(x_1|x_0)}}]\nonumber \\
&=\mathbb{E}_{x_{0:T}}[\log \frac{q(x_T|x_0)}{p(x_T)} +\sum_{t=2}^T \log \frac{q(x_{t-1} | x_{t}, x_0)}{p_{\theta}(x_{t-1}|x_t)}-\log p_{\theta}(x_0|x_1)]  \nonumber \\
&=\underset{E_1}{\underbrace{\mathbb{E}_{x_{0:T}}\log \frac{q(x_T|x_0)}{p(x_T)}}} + \underset{E_2}{\underbrace{\sum_{t=2}^T \mathbb{E}_{x_{0:T}} \log \frac{q(x_{t-1} | x_{t}, x_0)}{p_{\theta}(x_{t-1}|x_t)}}} - \mathbb{E}_{x_{0:T}}\log p_{\theta}(x_0|x_1) \nonumber \\
\end{align}
$$

我们先引入几条简单的期望的变换公式：
- 消除不包含的变量 $\mathbb{E}_{x,y\sim p(x,y)}[f(x)] = \mathbb{E}_{x\sim p(x)}[f(x)]$ ，证明：
$$
\begin{align}
\mathbb{E}_{x,y\sim p(x,y)}[f(x)] &= \int_{x,y} p(x, y) f(x) dxdy \nonumber \\
&= \int_x \left[\int_y p(x, y) dy\right] f(x) dx \nonumber \\
&= \int_x p(x)f(x) dx \nonumber \\
&= \mathbb{E}_{x\sim p(x)}[f(x)] \nonumber \\
\end{align}
$$

- 进行条件概率运算变换，即 $\mathbb{E}_{x, y\sim p(x, y)}[f(x, y)] = \mathbb{E}_{x\sim p(x)}\mathbb{E}_{y \sim p(y|x)} [f(x,y)]$ ，证明：
$$
\begin{align}
\mathbb{E}_{x, y\sim p(x, y)}[f(x, y)] &= \int_{x,y} p(x, y) f(x,y) dxdy \nonumber \\
&= \int_x p(x) \left[\int_y p(y|x) f(x, y) dy \right] dx \nonumber \\
&= \int_x p(x) \underset{\text{only related to } x}{\underbrace{\mathbb{E}_{y \sim p(y|x)} [f(x,y)]}} dx \nonumber \\
&= \mathbb{E}_{x\sim p(x)}\mathbb{E}_{y \sim p(y|x)} [f(x,y)]\nonumber \\
\end{align}
$$

- 添加任意无关的变量，即 $\mathbb{E}_{x\sim p(x)}[f(x)] =\mathbb{E}_{x,y\sim p(x,y)}[f(x)]$ ，前提是 $\exist x, y,\,\,p(x,y) > 0$ ，证明：
$$
\begin{align}
\mathbb{E}_{x\sim p(x)}[f(x)] &= \int_x p(x)f(x) dx \nonumber \\
&= \int_x \left[\int_y p(x, y) dy\right] f(x) dx \nonumber \\
&= \int_{x,y} p(x, y) f(x) dxdy \nonumber \\
&= \mathbb{E}_{x,y\sim p(x,y)}[f(x)] \nonumber \\
\end{align}
$$

根据上诉三个公式，变换 $E_1$ ：
$$
\begin{align}
\mathbb{E}_{x_{0:T}}[\log \frac{q(x_T|x_0)}{p(x_T)}]& = \mathbb{E}_{x_{0}, x_{T}}[\log \frac{q(x_T|x_0)}{p(x_T)}] \nonumber \\
&= \mathbb{E}_{x_0} \mathbb{E}_{x_T|x_0}[\log \frac{q(x_T|x_0)}{p(x_T)}] \nonumber \\
&= \mathbb{E}_{x_0} [\underset{\text{only related to }x_0}{\underbrace{KL(q(x_T|x_0)||p(x_T))}}]\nonumber \\
&=\mathbb{E}_{x_{0:T}} [KL(q(x_T|x_0)||p(x_T))]\nonumber \\
\end{align}
% \begin{align}
% \mathbb{E}_{x_{0:T}}[\log \frac{q(x_T|x_0)}{p(x_T)}]& = \int_{x_{0:T}} q(x_{0:T})\log \frac{q(x_T|x_0)}{p(x_T)} dx_{0:T} \nonumber \\
% & = \int_{x_0,x_{T}}\int_{x_{1:T-1}} q(x_0,x_T, x_{1:T-1})dx_{1:T-1}\log \frac{q(x_T|x_0)}{p(x_T)} dx_T dx_0 & (1) \nonumber \\
% &=\int_{x_{0},x_{T}} q(x_0, x_t) \log \frac{q(x_T|x_0)}{p(x_T)} dx_T dx_0 \nonumber \\
% &=\int_{x_{0}} q(x_0) \int_{x_T}q(x_T|x_0) \log \frac{q(x_T|x_0)}{p(x_T)} dx_T dx_0 & (2) \nonumber \\
% &= \int_{x_0} q(x_0) \underset{\text{only related to } x_0}{\underbrace{KL(q(x_T|x_0)||p(x_T))}}dx_0 = \mathbb{E}_{x_0\sim q(x_0)}[KL(q(x_T|x_0)||p(x_T))]\nonumber \\
% &= \int_{x_0}\int_{x_{1:T}} q(x_0, x_{1:T})dx_{1:T} \underset{\text{only related to } x_0}{\underbrace{KL(q(x_T|x_0)||p(x_T))}}dx_0 & (3)\nonumber \\
% &= \int_{x_0}\int_{x_{1:T}} q(x_0, x_{1:T})\underset{\text{only related to } x_0}{\underbrace{KL(q(x_T|x_0)||p(x_T))}}dx_{1:T} dx_0 \nonumber \\
% &= \int_{x_{0:T}} q(x_{0:T})\underset{\text{only related to } x_0}{\underbrace{KL(q(x_T|x_0)||p(x_T))}} dx_{0:T} \nonumber \\
% &= \mathbb{E}_{x_{0:T}\sim q(x_{0:T})}[KL(q(x_T|x_0)||p(x_T))]\nonumber \\
% \end{align}
$$

继续变换 $E_2$：
$$
\begin{align}
\sum_{t=2}^T \mathbb{E}_{x_{0:T}}[ \log \frac{q(x_{t-1} | x_{t}, x_0)}{p_{\theta}(x_{t-1}|x_t)}] &= \sum_{t=2}^T \mathbb{E}_{x_{0}, x_{t-1}, x_{t}}[ \log \frac{q(x_{t-1} | x_{t}, x_0)}{p_{\theta}(x_{t-1}|x_t)}] \nonumber \\
&= \sum_{t=2}^T \mathbb{E}_{x_{0}, x_{t}}\mathbb{E}_{x_{t-1} | x_{0}, x_{t}}[ \log \frac{q(x_{t-1} | x_{t}, x_0)}{p_{\theta}(x_{t-1}|x_t)}] \nonumber \\
&= \sum_{t=2}^T \mathbb{E}_{x_{0}, x_{t}} [\underset{\text{only related to }x_0, x_t}{\underbrace{KL(q(x_{t-1} | x_{t}, x_0)||p_{\theta}(x_{t-1}|x_t))}}] \nonumber \\
&= \sum_{t=2}^T \mathbb{E}_{x_{0:T}} [KL(q(x_{t-1} | x_{t}, x_0)||p_{\theta}(x_{t-1}|x_t))] \nonumber \\
\end{align} 
$$

于是损失函数可以变换为：
$$
\mathcal{L}_{EVLB} = \mathbb{E}_{x_{0:T}}\left[ \underset{\mathcal{L}_T}{\underbrace{KL(q(x_T|x_0)||p(x_T))}} + \underset{\mathcal{L}_{t-1}, t=2, \cdots, T}{\underbrace{\sum_{t=2}^T KL(q(x_{t-1} | x_{t}, x_0)||p_{\theta}(x_{t-1}|x_t))}} \underset{\mathcal{L}_0}{\underbrace{- \log p_{\theta}(x_0|x_1)}} \right]
$$

在上面公式中，可以将损失函数划分为 $\mathcal{L}_{T}$，$\mathcal{L}_{t-1}$ 和 $\mathcal{L}_{0}$ 三个部分，其中 $\mathcal{L}_{T}$ 是两个高斯分布之间的 $KL$ 距离（实际上是希望保证前向过程最终得到的是高斯噪声，而前向过程没有可学参数，而 $x_T$ 在逆过程中本来就是从高斯噪声中采样的，于是这一项是个与 $\theta$ 无关的）。$\mathcal{L}_{t-1}$ 是噪声匹配损失，希望网络能够学到每一个前向过程（给定 $x_0$）的逆过程。 $\mathcal{L}_{0}$ 是最终的重建损失，从噪声样本还原至真实样本。

接下来的问题就是 $q(x_{t-1} | x_{t}, x_0)$ 是什么的问题，通过条件概率公式，我们有
$$
\begin{align}
q(x_{t-1}|x_{t}, x_{0}) &= \frac{q(x_{t-1}, x_{t}, x_{0})}{q(x_{t}, x_{0})} = \frac{q(x_{t}|x_{t-1}, x_{0})q(x_{t-1}, x_{0})}{q(x_{t}, x_{0})} = \frac{q(x_{t}|x_{t-1})q(x_{t-1}| x_{0})}{q(x_{t}| x_{0})} \nonumber \\
&\propto \exp\left[-\frac{1}{2}\left(\frac{(x_{t} - \sqrt{1-\beta_t}x_{t-1})^2}{\beta_t} + \frac{(x_{t-1} - \sqrt{\bar{\alpha}_{t-1}} x_0)^2}{1 - \bar{\alpha}_{t-1}} - \frac{(x_t - \sqrt{\bar{\alpha}_t}^2)}{1 - \bar{\alpha}_t}\right) \right] \nonumber \\
&=  \exp \left[-\frac{1}{2}\left( \frac{x_{t}^2 - 2 \sqrt{\alpha_t}x_{t-1}x_t + \alpha_t x_{t-1}^2}{\beta_t} + \frac{x_{t-1}^2 - 2\sqrt{\bar{\alpha}_{t-1}} x_0 x_{t-1} + \bar{\alpha}_{t-1}x_0^2}{1 - \bar{\alpha}_{t-1}} - \frac{(x_t - \sqrt{\bar{\alpha}_t}^2)}{1 - \bar{\alpha}_t} \right)\right] \nonumber \\
&= \exp \left[-\frac{1}{2}\left( \left(\frac{\alpha_t}{\beta_t} + \frac{1}{1-{\bar{\alpha}_{t-1}}} \right)x_{t-1}^2 - 2\left(\frac{\sqrt{\alpha_t}}{\beta_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}}x_0\right)x_{t-1} + C(x_t, x_0) \right)\right] \nonumber \\
&= \lambda \exp \left[-\frac{1}{2}\left( \left(\frac{\alpha_t}{\beta_t} + \frac{1}{1-{\bar{\alpha}_{t-1}}} \right)x_{t-1}^2 - 2\left(\frac{\sqrt{\alpha_t}}{\beta_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}}x_0\right)x_{t-1}\right)\right] \nonumber \\
&= \lambda \exp \left[-\frac{\left( x_{t-1} - \left(\frac{\sqrt{\alpha_t}}{\beta_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}}x_0\right)/\left(\frac{\alpha_t}{\beta_t} + \frac{1}{1-{\bar{\alpha}_{t-1}}} \right)\right)^2}{2\cdot 1/\left(\frac{\alpha_t}{\beta_t} + \frac{1}{1-{\bar{\alpha}_{t-1}}} \right)}\right] \nonumber \\
\end{align}
$$

可以看到 $q(x_{t-1} | x_{t}, x_0)$ 符合一个整体放缩的高斯分布的概率密度，通过积分等于一可以得到具体的 $\lambda$ 的值，也就是说上述公式完全决定了 $q(x_{t-1} | x_{t}, x_0)$ 就是一个高斯分布，并且其均值和方差都决定了，分别为：
$$
\begin{align}
&q(x_{t-1} | x_{t}, x_0)=\mathcal{N}(x_{t-1};\tilde{\mu}_t(x_t,x_0), \tilde{\beta}_t\mathbf{I}) \nonumber \\
\tilde{\beta}_t &= 1/\left(\frac{\alpha_t}{\beta_t} + \frac{1}{1-{\bar{\alpha}_{t-1}}} \right) = 1/\left(\frac{\alpha_t(1-\bar{\alpha}_{t-1}) + \beta_t}{\beta_t(1-\bar{\alpha}_{t-1})} \right) \nonumber \\
&= 1/\left(\frac{\alpha_t- \alpha_t\bar{\alpha}_{t-1} + \beta_t}{\beta_t(1-\bar{\alpha}_{t-1})} \right) = 1/\left(\frac{1 - \bar{\alpha}_{t}}{\beta_t(1-\bar{\alpha}_{t-1})} \right) = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t}}\beta_t \nonumber \\
\tilde{\mu}_t(x_t,x_0)& = \left(\frac{\sqrt{\alpha_t}}{\beta_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}}x_0\right)/\left(\frac{\alpha_t}{\beta_t} + \frac{1}{1-{\bar{\alpha}_{t-1}}} \right) \nonumber \\
&=\left(\frac{\sqrt{\alpha_t}}{\beta_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}}x_0\right)\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t}}\beta_t \nonumber \\
&= \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})x_t + \beta_t\sqrt{\bar{\alpha}_{t-1}}x_0}{1-\bar{\alpha}_t} \nonumber \\
\end{align}
$$
回顾 $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_{t}}\epsilon$ ，可以有 $x_0 =  \frac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1-\bar{\alpha}_{t}}\epsilon)$ ，于是有
$$
\begin{align}
\tilde{\mu}_t(x_t,x_0)& = \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})x_t + \beta_t\sqrt{\bar{\alpha}_{t-1}}\frac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1-\bar{\alpha}_{t}}\epsilon)}{1-\bar{\alpha}_t} \nonumber \\
& = \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})x_t + \beta_t\frac{1}{\sqrt{{\alpha}_t}}(x_t - \sqrt{1-\bar{\alpha}_{t}}\epsilon)}{1-\bar{\alpha}_t} \nonumber \\
& = \frac{{\alpha_t}(1-\bar{\alpha}_{t-1})x_t + \beta_t(x_t - \sqrt{1-\bar{\alpha}_{t}}\epsilon)}{(1-\bar{\alpha}_t)\sqrt{{\alpha}_t}} \nonumber \\
& = \frac{({\alpha_t}-\bar{\alpha}_{t})x_t + (1 - \alpha_t) x_t - (1 - \alpha_t) \sqrt{1-\bar{\alpha}_{t}}\epsilon}{(1-\bar{\alpha}_t)\sqrt{{\alpha}_t}} \nonumber \\
& = \frac{(1-\bar{\alpha}_{t})x_t - (1 - \alpha_t) \sqrt{1-\bar{\alpha}_{t}}\epsilon}{(1-\bar{\alpha}_t)\sqrt{{\alpha}_t}} \nonumber \\
&= \frac{1}{\sqrt{{\alpha}_t}}\left(x_t -\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_{t}}}\epsilon\right) \nonumber \\
\end{align}
$$

问题转换成如何计算两个高斯分布的KL散度，推到过程如下：
$$
\begin{align}
&KL(\mathcal{N}(\mu_1, \Sigma_1), \mathcal{N}(\mu_2, \Sigma_2)) \nonumber \\
&= \mathbb{E}_{x\sim \mathcal{N}(\mu_1, \Sigma_1)}\Bigg[\log \frac{1}{(2\pi)^{\frac{D}{2}}|\Sigma_1|^{\frac{1}{2}}}\exp\left(-\frac{1}{2}(x-\mu_1)^{\top}\Sigma_1^{-1}(x-\mu_1)\right) \nonumber \\
&- \log \frac{1}{(2\pi)^{\frac{D}{2}}|\Sigma_2|^{\frac{1}{2}}}\exp\left(-\frac{1}{2}(x-\mu_2)^{\top}\Sigma_2^{-1}(x-\mu_2)\right)\Bigg] \nonumber \\
&= \mathbb{E}_{\mathcal{N}_1}\left[\frac{1}{2}\log\frac{|\Sigma_2|}{|\Sigma_1|} - \frac{1}{2}(x-\mu_1)^{\top}\Sigma_1^{-1}(x-\mu_1) + \frac{1}{2}(x-\mu_2)^{\top}\Sigma_2^{-1}(x-\mu_2) \right] \nonumber \\
&= \frac{1}{2}\log\frac{|\Sigma_2|}{|\Sigma_1|} - \frac{1}{2}\mathbb{E}_{\mathcal{N}_1}\left[(x-\mu_1)^{\top}\Sigma_1^{-1}(x-\mu_1)\right] + \frac{1}{2}\mathbb{E}_{\mathcal{N}_1}\left[(x-\mu_2)^{\top}\Sigma_2^{-1}(x-\mu_2)\right] \nonumber \\
&= \frac{1}{2}\log\frac{|\Sigma_2|}{|\Sigma_1|} - \frac{1}{2}\mathbb{E}_{\mathcal{N}_1}\left[\text{tr}\left((x-\mu_1)^{\top}\Sigma_1^{-1}(x-\mu_1)\right)\right] + \frac{1}{2}\mathbb{E}_{\mathcal{N}_1}\left[\text{tr}\left((x-\mu_2)^{\top}\Sigma_2^{-1}(x-\mu_2)\right)\right] \nonumber \\
&= \frac{1}{2}\log\frac{|\Sigma_2|}{|\Sigma_1|} - \frac{1}{2}\mathbb{E}_{\mathcal{N}_1}\left[\text{tr}\left(\Sigma_1^{-1}(x-\mu_1)(x-\mu_1)^{\top}\right)\right] + \frac{1}{2}\mathbb{E}_{\mathcal{N}_1}\left[\text{tr}\left(\Sigma_2^{-1}(x-\mu_2)(x-\mu_2)^{\top}\right)\right] \nonumber \\
&= \frac{1}{2}\log\frac{|\Sigma_2|}{|\Sigma_1|} - \frac{1}{2}\text{tr}\left(\Sigma_1^{-1}\mathbb{E}_{\mathcal{N}_1}\left[(x-\mu_1)(x-\mu_1)^{\top}\right]\right) + \frac{1}{2}\text{tr}\left(\Sigma_2^{-1}\mathbb{E}_{\mathcal{N}_1}\left[(x-\mu_2)(x-\mu_2)^{\top}\right]\right) \nonumber \\
&= \frac{1}{2}\log\frac{|\Sigma_2|}{|\Sigma_1|} - \frac{1}{2}\text{tr}\left(\Sigma_1^{-1}\Sigma\right) + \frac{1}{2}\text{tr}\left(\Sigma_2^{-1}\mathbb{E}_{\mathcal{N}_1}\left[(x-\mu_2)(x-\mu_2)^{\top}\right]\right) \nonumber \\
&= \frac{1}{2}\log\frac{|\Sigma_2|}{|\Sigma_1|} - \frac{D}{2} + \frac{1}{2}\text{tr}\left(\Sigma_2^{-1}\mathbb{E}_{\mathcal{N}_1}\left[xx^\top - \mu_2x^\top - x\mu_2^\top+\mu_2\mu_2^\top\right]\right) \nonumber \\
&= \frac{1}{2}\log\frac{|\Sigma_2|}{|\Sigma_1|} - \frac{D}{2} + \frac{1}{2}\text{tr}\left(\Sigma_2^{-1}\left(\underset{\mathbb{E}_{\mathcal{N}_1}[xx^\top]}{\underbrace{\Sigma_1 + \mu_1\mu_1^\top}} - \mu_2\mu_1^\top - \mu_1\mu_2^\top+\mu_2\mu_2^\top\right)\right) \nonumber \\
&= \frac{1}{2}\log\frac{|\Sigma_2|}{|\Sigma_1|} - \frac{D}{2} + \frac{1}{2}\text{tr}\left(\Sigma_2^{-1}\left(\Sigma_1 + (\mu_1 - \mu_2)(\mu_1-\mu_2)^\top\right)\right) \nonumber \\
&= \frac{1}{2}\log\frac{|\Sigma_2|}{|\Sigma_1|} - \frac{D}{2} + \frac{1}{2}\text{tr}(\Sigma_2^{-1}\Sigma_1) + \frac{1}{2}(\mu_1-\mu_2)^\top\Sigma_2^{-1}(\mu_1 - \mu_2) \nonumber \\
\end{align}
$$
可以看到两个高斯分布的$KL$距离之和均值和协方差矩阵有关，回顾 $q(x_{t-1} | x_{t}, x_0)=\mathcal{N}(x_{t-1};\tilde{\mu}_t(x_t,x_0), \tilde{\beta}_t\mathbf{I})$ 和 $p_{\theta}(x_{t-1}|x_t) = \mathcal{N}(x_{t-1};\mu_\theta(x_t, t),\Sigma_\theta(x_t, t))$，可以将 $\Sigma_\theta(x_t, t)$ 设置为一个只与时间相关的常量 $\sigma_t^2$，这里我们取 $\sigma_t^2=\tilde{\beta}_t \mathbf{I}$ ，于是有
$$
\begin{align}
\mathcal{L}_{t-1} &= KL(q(x_{t-1}|x_t, x_0)||p_\theta(x_{t-1}|x_t)) \nonumber\\
&= KL(\mathcal{N}(x_{t-1};\tilde{\mu}_t(x_t,x_0), \tilde{\beta}_t\mathbf{I})||\mathcal{N}(x_{t-1};\mu_\theta(x_t, t),\tilde{\beta}_t\mathbf{I})) \nonumber\\
&=\frac{1}{2}\log\frac{|\tilde{\beta}_t\mathbf{I}|}{|\tilde{\beta}_t\mathbf{I}|} - \frac{D}{2} + \frac{1}{2}\text{tr}((\tilde{\beta}_t\mathbf{I})^{-1}\tilde{\beta}_t\mathbf{I}) + \frac{1}{2}(\tilde{\mu}_t(x_t,x_0)-\mu_\theta(x_t, t))^\top(\tilde{\beta}_t\mathbf{I})^{-1}(\tilde{\mu}_t(x_t,x_0)-\mu_\theta(x_t, t))\nonumber \\
&= 0 - \frac{D}{2} + \frac{D}{2} + \frac{1}{2\tilde{\beta}_t}(\tilde{\mu}_t(x_t,x_0)-\mu_\theta(x_t, t))^\top(\tilde{\mu}_t(x_t,x_0)-\mu_\theta(x_t, t)) \nonumber \\
&= \frac{1}{2\tilde{\beta}_t}\|\tilde{\mu}_t(x_t,x_0)-\mu_\theta(x_t, t)\|^2 \nonumber \\
\end{align}
$$
前面我们得到 $\tilde{\mu}_t(x_t,x_0)=\frac{1}{\sqrt{{\alpha}_t}}\left(x_t -\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_{t}}}\epsilon\right)$ ，我们可以假定 $\mu_\theta(x_t, t)$ 也具有相同的形式，即
$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{{\alpha}_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t) \right) \nonumber \\
$$

于是可以进一步简化损失，
$$
\begin{align}
\mathcal{L}_{t-1} &= \frac{1}{2\tilde{\beta}_t}\|\tilde{\mu}_t(x_t,x_0)-\mu_\theta(x_t, t)\|^2 \nonumber \\
&= \frac{1}{2\tilde{\beta}_t}\|\frac{1}{\sqrt{{\alpha}_t}}\left(x_t -\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_{t}}}\epsilon\right)-\frac{1}{\sqrt{{\alpha}_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t) \right)\|^2 \nonumber \\
& = \frac{\beta_t^2}{2\tilde{\beta}_t{\alpha}_t(1-\bar{\alpha}_t)} \|\epsilon - \epsilon_\theta(x_t, t)\|^2 \nonumber \\
& = \frac{\beta_t^2}{2\tilde{\beta}_t{\alpha}_t(1-\bar{\alpha}_t)} \|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon, t)\|^2 \nonumber
\end{align}
$$

那么还剩下一项 $\mathcal{L}_0$ 重建损失，前面已经得知 $p(x_{t-1}|x_t) = \mathcal{N}(x_{t-1};\mu_\theta(x_t, t), \tilde{\beta}_t\mathbf{I})$ ，那么直接可得，
$$
\begin{align}
p(x_0|x_1) &= \Pi_{i=1}^D p(x_0^i | x_1 ) \nonumber \\
&= \Pi_{i=1}^D \int_{\delta_{-}(x_0^i)}^{\delta_{+}(x_0^i)} \mathcal{N}(x_{0};\mu_\theta^i(x_1, 1), \tilde{\beta}_1) dx_0\nonumber \\
&= \Pi_{i=1}^D \text{cdf}_{\mathcal{N}(\mu_\theta^i(x_1, 1), \tilde{\beta}_1)}(x) |_{\delta_{-}(x_0^i)}^{\delta_{+}(x_0^i)}\nonumber \\
\delta_{-}(x_0^i) = &
\left\{ 
    \begin{align}
    -\infty, \quad x_0^i = -1 \nonumber \\
    x_0^i - \frac{1}{255}, \quad x_0^i > -1 \nonumber \\
    \end{align} 
\right.,
\delta_{+}(x_0^i) = 
\left\{ 
    \begin{align}
    \infty, \quad x_0^i = 1 \nonumber \\
    x_0^i + \frac{1}{255}, \quad x_0^i < 1 \nonumber \\
    \end{align} 
\right.\nonumber \\
\end{align}
$$
之所以这么计算是因为DDPM认为输入时将图片从[0, 255]压缩为[-1, 1]，由于取值是离散的，于是采用周围区间积分值作为概率值。具体可见下图所示：
![重建似然估计](./src/diffusion/reconstrution.png)

于此同时，DDPM还提出了一种简化训练的方式，即去除所有噪声匹配的权重和最后的重建损失，计算如下：
$$
\begin{align}
\text{Simplified }\mathcal{L}_{t-1} 
& = \|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon, t)\|^2 \nonumber
\end{align}
$$
最终，训练和采样算法如下：
![训练和采样算法](./src/diffusion/training_sampling.png)

值得注意的是，采样过程实际上就是 $p(x_{t-1}| x_t) = \mathcal{N}(x_{t-1};\mu_\theta(x_t, t), \tilde{\beta}_t\mathbf{I})$，也就是
$$
\begin{align}
x_{t-1} &= \mu_\theta(x_t, t)+\sqrt{\tilde{\beta}_t} \epsilon \nonumber \\
&= \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t)\right) + \sqrt{\tilde{\beta}_t} \epsilon \nonumber
\end{align}
$$
图中算法 $2$ 在 $t = 1$ 时，直接采用 $x_0 = \mu_\theta(x_t, t)$ 忽略最后的加噪部分。
值得注意的是，其中的 $\sigma^2_t$ 可以采用 $\tilde{\beta}_t$ 也可以采用 $\beta_t$，DPM 中有理论解释，DDPM 采用 $\sigma^2_t = \tilde{\beta}_t$。