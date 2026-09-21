06.Feb.26


## 架构

VAE + LM + DiT

## 特点

- 少步蒸馏
- LM 训练

## 笔记



### 少步蒸馏？

>   蒸馏？

    蒸馏即师生蒸馏（见 Ming-UniAudio）：

    能力强的教师模型训练更小或更快的学生模型，学习教师的输出、中间表示或分布。常见蒸馏：

    - 分布匹配蒸馏：常用于扩散/流匹配
    - 对抗蒸馏：用判别器逼学生输出接近教师或真实分布

（师生是什么）（蒸馏技巧）

ACE-Step v1.5 的 DiT 训练步骤为：

- 预训练
- 多任务预训练
- **少步蒸馏** ：希望减少推理步数
- 强化学习

即使用 **蒸馏** 减少推理步数，教师模型和学生模型都是 DiT 本身，但是主动地把学生模型的推理步数压缩，以期望输出仍能对齐未压缩推理步数的自己。

>   为什么不需要 CFG？

>   **CFG（Classifier-Free Guidance，无分类器引导）**。它是扩散模型和流匹配模型里一种 **增强条件一致性** 的推理技巧。
>   在条件生成中，模型既要生成合理的内容，又要符合条件 $y$（比如文本、歌词、参考音频）。CFG 的做法是：
>   - 同时计算**条件预测**和**无条件预测**；
>   - 然后用一个引导尺度 $w$ 做外推：
>   $$\hat{v} = v_{\text{uncond}} + w \cdot (v_{\text{cond}} - v_{\text{uncond}})$$
>   或者写成：$$\hat{v} = (1-w) v_{\text{uncond}} + w v_{\text{cond}}$$
>   其中：
>   - $v_{\text{cond}}$：有条件时的预测；
>   - $v_{\text{uncond}}$：无条件时的预测；
>   - $w$：CFG 引导尺度，通常 $w > 1$。

>   直观理解：
>   - $w=1$：只用条件预测；
>   - $w>1$：把“条件方向”放大，让生成更贴合条件；
>   - 代价是多样性下降，可能出现过度饱和、伪影。

>   在扩散模型中，CFG 通常需要**两次前向传播**：一次带条件，一次不带条件。所以推理成本大约翻倍。

*只有 turbo 模型经过了少步蒸馏，base 和 sft 模型没有。*

### LM

主体没有使用 MLLM，数据集处理使用了 MLLM。而是使用了 LM。

LM 不直接处理原始音频，只处理文本和 5hz audio codes（by Audio tokenizer）。

- target latent：干净目标音频的 潜变量。训练时，它是监督目标；加噪后变成 Noised Target Latent，模型学习从它去噪回干净 target。

- Source Latent：条件/结构源潜变量。

5hz audio codes来自target latent。（但是这样的意义何在？LM 是冻结的，怎么去进行学习呢？）

LM 针对不同任务有不同模式：

- Planner：作用于生成。audio codes是输出。
- Listener：作用于理解。audio codes是输入。

Planner 模式下，训练时，干净的目标音频经过 VAE 编码成 25Hz target latent，经过：

- 目标音频 -> VAE -> 加噪给 DiT
- 目标音频 -> FSQ -> 给 LM 作为目标

所以意义在于 **LM 学习生成 audio codes，理解 caption 、 lyrics 到 audio codes 的映射关系** 。

>   MLLM？Qwen-2.5-Omni的音乐理解？

### 特殊组件

Lyric（歌词）-> Lyric Encoder -> 条件注入DiT

refer audio -> Timbre Encoder -> 注意力注入DiT

>   输入处理：Lyric Encoder & Timbre Encoder？

