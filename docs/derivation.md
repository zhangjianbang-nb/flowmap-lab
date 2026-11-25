# 数学推导：从 flow matching 到 flow map 蒸馏

本文把项目里的公式按“定义 → 训练目标 → 采样 → 蒸馏”四步写清楚。所有约定与代码一致：**t=0 是纯噪声，t=1 是数据**。

## 1. 概率路径与连续性方程

设数据分布为 $p_{\text{data}}(x_1)$，噪声分布为 $p_{\text{noise}}(x_0)=\mathcal N(0,I)$。一条从噪声到数据的概率路径 $p_t(x)$ 满足连续性方程

$$
\partial_t p_t(x) + \nabla\cdot\big(p_t(x)\,v_t(x)\big)=0,
$$

其中 $v_t(x)$ 是生成该路径的速度场。若我们能学到 $v_t$，就从 $x_0\sim p_{\text{noise}}$ 积分 ODE

$$
\frac{dx}{dt}=v_t(x)
$$

得到 $x_1\sim p_{\text{data}}$。

## 2. 条件路径与条件速度

本项目采用线性插值（条件于 $x_1$）：

$$
x_t=\alpha_t x_1+\sigma_t \varepsilon,\qquad \varepsilon\sim\mathcal N(0,I),
$$

于是 $p_t(x\mid x_1)=\mathcal N(\alpha_t x_1,\sigma_t^2 I)$。对 $t$ 求导：

$$
\frac{dx_t}{dt}=\alpha_t' x_1+\sigma_t' \varepsilon
=\alpha_t' x_1+\frac{\sigma_t'}{\sigma_t}\big(x_t-\alpha_t x_1\big).
$$

记条件速度

$$
v_t(x\mid x_1)=\alpha_t' x_1+\sigma_t'\,\frac{x-\alpha_t x_1}{\sigma_t}.
$$

代码里的回归目标直接写成 $\alpha_t' x_1+\sigma_t' \varepsilon$，因为训练时 $\varepsilon$ 是已知的采样量，二者等价。

## 3. Flow-matching 目标

Flow matching（Lipman et al., 2023）回归条件速度：

$$
\mathcal L_{\text{FM}}(\theta)=
\mathbb E_{t\sim U(0,1),\,x_1\sim p_{\text{data}},\,\varepsilon\sim\mathcal N(0,I)}
\left\|v_\theta(x_t,t)-\big(\alpha_t' x_1+\sigma_t'\varepsilon\big)\right\|^2.
$$

其最小化点满足

$$
v_\theta^\star(x,t)=\mathbb E\big[v_t(x\mid x_1)\mid x_t=x\big].
$$

并且 $v_\theta^\star$ 生成同一条 $p_t$ 路径（条件 FM 与边缘 FM 的等价性）。直观上，模型学的是“在当前噪声水平下，所有可能数据点的平均速度”——这也是为什么在高度对称的多峰分布上，条件期望会把多个 mode 平均掉，需要足够容量与训练步数才能恢复锐利样本。

## 4. Log-linear 路径

FLM 论文对语言使用 log-linear 信噪比：

$$
\alpha_t=e^{-k(1-t)},\qquad \sigma_t=1-e^{-k(1-t)}.
$$

于是

$$
\gamma_t=\log\frac{\alpha_t}{\sigma_t}=-k(1-t)
$$

是 $t$ 的线性函数。给定 $\gamma$，令 $a=\sigma(\gamma)$，则

$$
t=1+\frac{\log a}{k}
$$

是精确逆变换。代码中的 `ScheduleLUT` 用单调三次 Hermite 插值复现论文的查表实现；解析式是事实来源。

注意端点：$t=0$ 时 $\alpha_0=e^{-k}>0$，并非严格纯噪声；$t=1$ 时 $\sigma_1=0$，目标退化为 $\alpha_1' x_1$。训练时把 $t$ 夹到 $[10^{-3},1-10^{-3}]$ 避免退化。

## 5. Euler 采样

从 $x_0\sim\mathcal N(0,I)$ 出发，取 $N$ 步：

$$
x_{t+\Delta}=x_t+\Delta\,v_\theta(x_t,t),\qquad \Delta=\frac{1}{N}.
$$

`flowmap.flm.integrate` 支持逐样本不同的起止时间，供蒸馏使用。

## 6. Flow map 与蒸馏

定义**flow map**

$$
f^\star(x_t,t,s)=x_s,\qquad s>t,
$$

即从时刻 $t$ 的状态直接跳到时刻 $s$ 的状态。它满足半群性质

$$
f^\star\big(f^\star(x_t,t,u),u,s\big)=f^\star(x_t,t,s),\qquad t<u<s.
$$

训练时用教师 FLM 的 Euler 轨迹给出目标：

$$
x_s^{\text{teacher}}=\operatorname{Euler}\big(v_{\text{FLM}},x_t,t\to s\big),
$$

$$
\mathcal L_{\text{FMLM}}(\theta)=
\mathbb E_{t,s,x_1,\varepsilon}
\left\|f_\theta(x_t,t,s)-\operatorname{sg}\big(x_s^{\text{teacher}}\big)\right\|^2,
$$

其中 $\operatorname{sg}$ 表示停止梯度。推理时

$$
x_1=f_\theta(x_0,0,1)
$$

只需一次前向。

## 7. 误差从哪来

1. **教师误差**：蒸馏目标本身是近似轨迹，学生不会优于教师。
2. **条件期望的平均效应**：MSE 最优解是条件均值；多峰分布下，均值可能落在 mode 之间。增大容量、延长训练、或用更好的时间采样可以缓解。
3. **一步跳跃的跨度**：$t\to 1$ 的跨度越大，映射越难学；因此训练时混合“全跨度”和“随机中间跨度”（`full_jump_prob`）。

## 8. 与论文实现的关系

官方实现（`david3684/flm`）使用 log-linear LUT、DiT backbone、大规模语料与分布式训练。本项目保留同一套数学，但把模型缩到 MLP/TinyTransformer，把数据缩到 2-D 与 Markov 合成语言，以便在单机上验证：

- 损失是否按推导下降；
- Euler 采样是否把噪声搬到数据流形；
- 蒸馏后一步样本是否与教师多步轨迹一致；
- 调度与步数的消融是否呈现可解释的趋势。

## 参考

- Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023.
- Esser et al., *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis*, ICML 2024.
- FLM / FMLM: *Flow Map Language Models*（本仓库的工程对象；官方代码 [david3684/flm](https://github.com/david3684/flm)）。
