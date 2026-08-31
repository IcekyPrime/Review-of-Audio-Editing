19.Jan.26

## 架构

VAE + ALM + MMDiT

## 特点

扩散

ALM

数据集合成抽象、批量、公式化

## 笔记

### 数据

>   时间戳标注？

    利用 TAG（Text-to-Audio Grounding）模型 为每条事件描述标注其在音频中的起止时间（事件边界）。

    时间戳标注仅用于训练输入合成。

>   解耦的前景和背景组件？

    将音频中的声音分为 前景（Foreground, F）——短时、突发的单个声学事件（如狗叫、喇叭声），和 背景（Background, B）——持续的环境声音（如雨声、街道噪音）。为六种编辑操作提供明确的操作对象。

    抽象为组合函数，结合时间戳，批量合成。

>   特异性槽位？

    在指令模板中预留的可替换位置，用于填充具体的事件类型、时间戳、出现次数、响度因子、速度因子等参数。使得同一类操作（如添加）能生成大量多样且自然的指令。

### 输入处理？

源音频 $\mathbf{x}_in$ 和文本指令 $\mathbf{y}$ 共同经过 ALM 处理，输出 $\mathbf{H}\in\mathbb{R}^{L_q \times D_q}$ 。

*注：$L_q$ 和 $D_q$ 都是模型架构的超参数，序列长度和隐藏维度*

$\mathbf{H}$ 经过处理得到：

- 全局上下文：$\mathbf{c}_{global}$ 

    先池化后 AdaLN 调制。

- 细粒度序列特征：$\mathbf{H}_{seq}$

    $H_{seq}$ 即是 $H$ 本身。

>   AdaLN 调制：

    自适应层归一化（Adaptive Layer Normalization）。

而 VAE 仍然负责音频编解码工作。


### ALM的输出如何注入？

- 加噪语音的编码/潜变量将与源音频的潜变量沿通道维度拼接：

$$
\tilde{\mathbf{z}}_t = \mathbf{z}_t \oplus \mathbf{z}_{in} \in \mathbb{R}^{2C \times L}
$$

- 扩散时间步 $t$ 

>   扩散时间步？

    1. 推理时也需要t，与训练时随机采样t不同，推理时t是按照 DDIM 的确定性采样调度逐步给定的。

    2. 无论流匹配还是扩散，都需要t，包括推理过程。推理中启用了ODE数值求解器/离散调度器，自动输入t。

### 底层DDPM的DiT？

**DDPM（Denoising Diffusion Probabilistic Models，去噪扩散概率模型）** ，而不是流匹配。(\formulation\DDPM.md)

$$
\mathbf{z}_t = \sqrt{\bar{\alpha}_t}\mathbf{z}_0 + \sqrt{1-\bar{\alpha}_t}\epsilon,\quad \epsilon \sim \mathcal{N}(0,\mathbf{I})
$$




