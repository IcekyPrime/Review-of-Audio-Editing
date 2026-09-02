# 音频/图像 Edit 任务

Edit 的基础操作是 增（Add），删（Delete），改（Modify）。语音 Edit（Speech Editing） 是最早期的 Edit 任务中，它希望能对语音对应的文本内容进行 Edit ；而随着 Edit 操作的对象扩展到了背景音、音效事件，Edit 任务也扩展为了 通用音频编辑任务（Audio Editing）。

近段时间（2026.08），无论图像还是音频 Edit 模型，架构主流都是 **VAE + MLLM + DiT** ：

- VAE 负责音频/图像编码
- MLLM 负责指令与音频/图像的处理，使得得到的隐向量有图像的语义和指令的理解
- DiT 负责（主要使用 **流匹配** ）得到目标音频/图像

## 目录

本项目的文件结构如下：

### [Audio/](Audio/) — 音频编辑模型（7 个）

| 模型 | 论文 | 翻译 | 总结 |
|------|------|------|------|
| **[VoiceBox](Audio/VoiceBox/)** | [VoiceBox.pdf](Audio/VoiceBox/VoiceBox.pdf) | [VoiceBox翻译.pdf](Audio/VoiceBox/VoiceBox翻译.pdf) / [VoiceBox翻译.md](Audio/VoiceBox/VoiceBox翻译.md) | [VoiceBox总结.md](Audio/VoiceBox/VoiceBox总结.md) |
| **[VoiceCraft](Audio/VoiceCraft/)** | [VoiceCraft.pdf](Audio/VoiceCraft/VoiceCraft.pdf) | [VoiceCraft翻译.pdf](Audio/VoiceCraft/VoiceCraft翻译.pdf) / [VoiceCraft翻译.md](Audio/VoiceCraft/VoiceCraft翻译.md) | [VoiceCraft总结.md](Audio/VoiceCraft/VoiceCraft总结.md) |
| **[Ming-UniAudio](Audio/Ming-UniAudio/)** | [Ming-UniAudio.pdf](Audio/Ming-UniAudio/Ming-UniAudio.pdf) | [Ming-UniAudio翻译.pdf](Audio/Ming-UniAudio/Ming-UniAudio翻译.pdf) / [Ming-UniAudio翻译.md](Audio/Ming-UniAudio/Ming-UniAudio翻译.md) | [Ming-UniAudio总结.md](Audio/Ming-UniAudio/Ming-UniAudio总结.md) |
| **[MMEDIT](Audio/MMEDIT/)** | [MMEDIT.pdf](Audio/MMEDIT/MMEDIT.pdf) | [MMEDIT翻译.pdf](Audio/MMEDIT/MMEDIT翻译.pdf) / [MMEDIT翻译.md](Audio/MMEDIT/MMEDIT翻译.md) | [MMEDIT总结.md](Audio/MMEDIT/MMEDIT总结.md) |
| **[Audio-Omni](Audio/Audio-Omni/)** | [Audio-Omni.pdf](Audio/Audio-Omni/Audio-Omni.pdf) | [Audio-Omni翻译.pdf](Audio/Audio-Omni/Audio-Omni翻译.pdf) / [Audio-Omni翻译.md](Audio/Audio-Omni/Audio-Omni翻译.md) | — |
| **[UNISON](Audio/UNISON/)** | [UNISON.pdf](Audio/UNISON/UNISON.pdf) | [UNISON翻译.pdf](Audio/UNISON/UNISON翻译.pdf) / [UNISON翻译.md](Audio/UNISON/UNISON翻译.md) | — |
| **[CosyEdit](Audio/CosyEdit/)** | [CosyEdit.pdf](Audio/CosyEdit/CosyEdit.pdf) | [CosyEdit翻译.pdf](Audio/CosyEdit/CosyEdit翻译.pdf) / [CosyEdit翻译.md](Audio/CosyEdit/CosyEdit翻译.md) | [CosyEdit总结.md](Audio/CosyEdit/CosyEdit总结.md) |

### [Image/](Image/) — 图像编辑模型（5 个）

| 模型 | 论文 | 翻译 | 总结 |
|------|------|------|------|
| **[OmniGen](Image/OmniGen/)** | — | [OmniGen翻译.pdf](Image/OmniGen/OmniGen翻译.pdf) | [OmniGen总结.md](Image/OmniGen/OmniGen总结.md) |
| **[Z-Image](Image/Z-Image/)** | [Z-image.pdf](Image/Z-Image/Z-image.pdf) | [Z_Image翻译.pdf](Image/Z-Image/Z_Image翻译.pdf) | — |
| **[Qwen-Image-Edit](Image/Qwen-Image-Edit/)** | [Qwen-Image-Edit.pdf](Image/Qwen-Image-Edit/Qwen-Image-Edit.pdf) | [Qwen_Image_Edit翻译.pdf](Image/Qwen-Image-Edit/Qwen_Image_Edit翻译.pdf) | — |
| **[FireRed-Image-Edit](Image/FireRed-Image-Edit/)** | [FireRed-Image-Edit.pdf](Image/FireRed-Image-Edit/FireRed-Image-Edit.pdf) | [Firered-Image-Edit翻译.pdf](Image/FireRed-Image-Edit/Firered-Image-Edit翻译.pdf) | — |
| **[DreamOmni2](Image/Dreamomni2/)** | [DreamOmni2.pdf](Image/Dreamomni2/DreamOmni2.pdf) | [DreamOmni2翻译.pdf](Image/Dreamomni2/DreamOmni2翻译.pdf) | — |

### [Formulation/](Formulation/) — 基础理论

| 文件 | 内容 |
|------|------|
| [扩散模型.md](Formulation/扩散模型.md) | 扩散模型的数学原理、DDPM公式、训练与采样 |
| [流匹配损失.md](Formulation/流匹配损失.md) | 流匹配的数学原理、CFM/OT-CFM、与扩散的区别 |
| [VAE与编码.md](Formulation/VAE与编码.md) | VAE原理、离散token（声学/语义）与连续表示的对比 |

### 其他文件

- **Introduction.md**：本文档，项目总览与发展综述
- **.gitignore**：Git 忽略规则

---

## 音频编辑模型的发展综述

2023 年，流匹配的框架被 Meta 提出。**VoiceBox** 是 Meta 随后提出的、最早期的语音编辑模型之一，它提供了一个语音编辑任务范式：

>   语音编辑 = 基于参考语音生成全新的语音（可拼接）

VoiceBox 还使用了其他许多的组件，如 MFA 用于音素对齐、时长模型用于预测目标语音的时长。但是它奠定了 **流匹配** 的数学底层。

VoiceBox 使用了 Mel 谱，而其后 Meta 发布的 **VoiceCraft** 则使用了当时流行的 EnCodec 离散编解码。此外，区别于 VoiceBox 显式地拼接源语音和生成语音，VoiceCraft 设计了一种特殊的掩码填充方式：

$$Y = (X_1, <M_1>, X_5, X_6, <M_1>, X_2, X_3, X_4)$$

>   第一个 $M_1$ 标记了填充位置，第二个 $M_1$ 标记了生成内容的起始标记。$M_1$ 本身不对应生成内容，模型生成的内容从第二个 $M_1$ 之后开始。所以在生成时能看到上下文，但是生成是因果（自回归）的。

无论是 VoiceBox 还是 VoiceCraft ，它们都只是语音编辑模型，只需要输入目标文本，不需要其他复杂的指令。

语音编辑任务显然是不够的。 **Ming-UniAudio** 引入了多个基于声学属性的全新任务：去噪、速度/音高/音量、方言/情感转换。它是首个 **指令式、自由形式、无需时间戳** 的通用语音编辑模型。换言之，它的指令形式已经不局限于简单的目标文本。

Ming-UniAudio 设计了一个 VAE + LLM + DiT 的架构雏形。它通过设计 VAE 与 LLM 的联合训练，使得 **VAE** 对于作为文本分词处理和音频编辑\生成主干的 LLM 的语义空间更敏感。Ming-UniAudio 的评价 VAE 为：

>   统一的语义和声学表示，提供了比离散方法更好的性能。这种统一方法表明，未来实现无缝、自由形式的语音编辑现在是可行的。这一能力意义重大，因为它弥合了理解和生成之间的关键差距。

Ming-UniAudio 中的 LLM 并不是 MLLM ，它并不是处理音频，而是处理音频在 VAE 中的含有一定语义的中间输出。在之后，交大和AI Lab研发的 **MMEDIT** 以及其他的一些成果，引入了 MLLM 或者 ALM。这之后通用音频编辑模型的生成主干基本沿用了 **MLLM + DiT** 。

MLLM 处理后的潜变量会作为注意力注入到 DiT 中。那么原始的输入指令和参考音频是如何输入的？在 Ming-UniAudio 及之前，采用单流架构：

>   音频和文本的token直接拼接，使用同一个Transformer网络处理。

而 MMEDIT 以及 **Audio-Omni** 中，则是采用了 **双流** 架构（对应 **MMDiT**）：

>   音频和文本各设独立的处理通道，通过特定机制如交叉注意力进行信息交换和融合。

双流 在 Audio-Omni 中被证实有利于通用音频编辑。Audio-Omni 为了增强语义理解，还采用了 ConvNeXtV2 架构的字符级编码器，得到 $F_{trans}$ （转录文本特征）。

构造新特征有利于引导模型理解语义空间，而发掘旧的隐状态也有作用。**UNISON** 设计了显式的任务标签，并将 MLLM 的 **全部隐状态都作为注意力** 注入到 DiT 的对应层中。

除此以外，一些模型也聚焦于轻量化、少量数据集下的训练。 **CosyEdit** 基于 CosyVoice 改动，实现了这两点。

## 与此同时的图像编辑模型

和音频编辑模型一样，最终都采用了 **VAE + MLLM + DiT** 。

不同的是，图像编辑模型最终还要考虑编码问题。二维平面的位置关系复杂，且图像有丰富的宽高，最为关键的是，图像编辑的输入可能有多张图像，用相同的编码会导致空间位置混淆，产生复制-粘贴伪影。另一方面，前面提到，双流架构有利于编辑任务理解语义，但图像编辑里的双流针对的是一维的文本和二维的图像，如何让它们在联合注意力中交互？

**Z-Image** ， **FireRed-Image-Edit** 和 **Qwen-Image-Edit** 采用了相似的策略：在二维的位置编码基础上扩展为三维，加入一个维度分辨不同的图像。

**DreamOmni2** 则额外加入了位置编码偏移，类似于透视，即使是不同层，其位置编码的起始也不同，从新加入的维度上看各个图像互不重叠。

## 音频编辑模型的不足

这里主要是总结了一下各个论文的局限性自述：

**VoiceBox** & **VoiceCraft** ：

- 对话语音迁移能力不足，本质也算是数据集不够完整，仅在朗读语音上训练，常规对话还原不好

- 依赖音素化器和强制对齐器（后面会逐步实现端到端）

- 音频风格只能迁移，不能对每个声学属性独立控制

**Ming-UniAudio** ：

- 语义蒸馏仍需改进

- 指令跟随生成被认为是实现更丰富的编辑操作的先决条件

>   Instruction-following generation: we consider this capability to be a crucial prerequisite for more advanced general-purpose instruction-based speech editing.\
>   从指令直接生成符合要求的音频，不依赖源音频的编辑

- 模态覆盖有限（不过在 UNISON 中，额外加入了视频参考条件）

**MMEDIT** ：

- 生成多样性不足

- 多个声学事件在时间上重叠后，定位和编辑的困难

**Audio-Omni** 没有说明自己的局限性。

**UNISON** ：

- VAE 重建质量上限：

>   VAE reconstruction quality. UNISON relies on the pre-trained MMAudio VAE, which was originally designed for environmental sound synthesis. While it provides a compact and effective latent space for general audio, its reconstruction fidelity for speech—particularly high-frequency formant details, subtle prosodic variations, and breathy or whispered voice qualities—imposes an upper bound on overall output quality. This is especially noticeable for zero-shot TTS, where fine-grained speaker timbre nuances may be smoothed out during VAE encoding. A natural next step is to train a unified VAE with improved speech reconstruction, potentially adopting higher latent resolution or a multi-scale architecture that better preserves both spectral detail and temporal dynamics.

>   好的 VAE 会带来好的上限，似乎类似于零样本 TTS 系统发展中使用的声码器不断发展

- 合成训练数据的不真实（这也是 Audio-Omni 虽未说明但存在的局限性）

