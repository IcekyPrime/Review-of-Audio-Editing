21.Nov.24

## 架构

VAE + Transformer（流匹配）

## 特点

没有 MLLM 处理多模态输入，直接让DiT通过双向自注意力+海量数据学习语义空间

Edit 区域加权损失

## 笔记

### 没有MLLM如何处理输入?

文本：使用 Phi-3 的 tokenizer 将文本指令转换为 token 序列。

>   Phi-3？

    参考Qwen开源模型

图像：仅用VAE将源图像编码，再用线性层将潜变量展平为视觉 token 序列。

文本 token 和图像 token 被拼接成一个统一的、交错的 token 序列，作为 Transformer 的输入。

### 区域加权损失？

由于编辑任务中源图像和目标图像差异通常很小（只改局部），模型容易学到 “直接复制输入图像” 的捷径。

区域加权损失：

$$
w_{i,j}=
\begin{cases}
1 & \text{if } x_{i,j}=x'_{i,j}\\[6pt]
\dfrac{1}{\left\|x-x'\right\|^2} & \text{if } x_{i,j}\neq x'_{i,j}
\end{cases}
$$

最终损失函数：

$$
\mathcal{L}_{\text{weighted}} = \mathbb{E}\left[\mathbf{W} \odot \left\| (\mathbf{x}-\epsilon) - v_{\theta}(\mathbf{x}_t, t, c) \right\|^2\right]
$$

### 语义信息理解？

Transformer 的自注意力机制实现的——图像 token 和文本 token 在同一个序列中相互 attend，模型能自动定位到指令所指的对象。

>   自注意力？

通过自注意力机制（Self-Attention）让模型自主地学习跨模态的语义对齐。


