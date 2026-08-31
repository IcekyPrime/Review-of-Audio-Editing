26.Apr.26

## 架构

MLLM + DiT


## 特点

首个指出 双流 有利于 统一音频理解、生成和编辑

额外引入 $F_{trans}$ ，加强时间敏感


## 笔记

### MLLM?

Multimodal Large Language Model，多模态大语言模型
- 常用：
    - Audio-Omni：Qwen2.5-Omni-3B
    - UNISON：Qwen2.5-Omni-7B
用于处理输入，包括文本指令、音频波形和视频。*这些输入由其各自的编码器进行token化*。
>   VAE？


### 双流？

>   流匹配？

>   单流？

    音频和文本的token直接拼接，使用同一个Transformer网络处理。计算量二次方增长。
    音频和文本各设独立的处理通道，通过特定机制如交叉注意力进行信息交换和融合。
    双流适合广域上的音频编辑：

    ```paper
    To effectively bridge the two components,we design a hybrid conditioning mechanism that disentangles inputs into two complementary streams:a High-Level Semantic stream,combining MLLM features and text embeddings for speech synthesis,injected viacross-attention to provide instructional guidance;and a Low-Level Signal stream,fusing mel-spectrogram and video sync features, concatenated with the input noise for precise temporal control.This separation is key to mastering the diverse requirements of sound, music,and speech within a single framework.
    我们设计了一种混合条件机制，将输入解耦为两个互补的流：一个高层语义流，结合MLLM特征和用于语音合成的文本嵌入，通过交叉注意力注入以提供指令性指导；以及一个低层信号流，融合梅尔频谱图和视频同步特征，与输入噪声拼接以提供精确的时间控制。这种分离是在单一框架内掌握声音、音乐和语音的多样化需求的关键。
    ```

    - 降低计算复杂度，增加参数量
    - 高保真，理解复杂指令


### F_{trans}/转录文本特征?

Audio-Omni认为，MLLM的多模态特征不够直接，文本通过基于ConvNeXtV2架构的字符级编码器转换为字符级嵌入，与多模态特征拼接组成高层语义特征。

>   多模态特征？
    
    MLLM的倒数第二层。

>   ConvNeXtV2?
        
    F5-TTS，E2-TTS

### 输入

总的来说，Audio-Omni的DiT输入组成为：

高层文本特征：
- 文本指令经过基于ConvNeXtV2的编码器处理后的转录文本特征
- 所有输入（文本指令、参考音频和视频）送入MLLM（内部有内置编码器）处理得到的多模态特征
上面两个特征维度拼接。

低层信号特征：
- 音频输入经梅尔编码器提取的梅尔频谱图特征
- Synchformer模型从输入视频提取的同步特征
上面两个特征维度拼接。

>   Synchformer?

作用方式：
低层信号特征通过元素级加法与时间嵌入融合，再与加噪音频经VAE处理后拼接，形成DiT的主要输入。
高层语义特征通过交叉注意力作为上下文注入。


### 数据集？

真实数据与合成数据。

>   自动化过程？


### 为什么会有视频输入？

根据视频生成同步音频/音乐的创新Edit任务。


## 代码

如何调用MLLM

如何取用中间层和次高层

如何注入

双流

