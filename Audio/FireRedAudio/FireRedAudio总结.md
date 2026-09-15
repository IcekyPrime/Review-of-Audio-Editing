27.Aug.26

## 架构

Audio Encoder + LLM + DiT  

## 特点

两个音频编解码路径

LLM 结合 ASR

## 笔记

### 两条路径：

在 FireRedAudio 的论文中，详细介绍了各音频编解码方式的优缺点：

（待补充）

输入是解耦的：

**Audio Encoder** ：语义侧

**RedAE Encoder** ：音频侧，高保真细节

显然输出为获得更好的质量，需要走 RedAE 路径。无论是零样本 TTS 还是语音编辑任务，都需要 RedAE Decoder 处理。

>   Patch Encoder：
    DiT 通过流匹配每步生成4帧（160ms）潜变量，其不仅要给 RedAE Decoder 处理，还要给 Patch Encoder 处理后给 LLM 处理。LLM 用于决定是否继续。

零样本 TTS 和编辑的参考音频处理的路径不同：

- 零样本 TTS：RedAE 路径。因为零样本 TTS 希望还原音色，需要保真。

- 编辑：Audio Encoder 路径。因为无论是语义编辑还是声学属性编辑，都需要理解音频语义。



### ASR 的融入：

不仅集成了 TTS 和 Edit，还引入了 ASR。

因为 Audio Encoder 的 token 注重语义性，FireRedAudio 希望直接让 LLM 依据 token 生成文本。此前的编辑模型虽然使用了 LLM，但是没有想过加入 ASR。

（LLM 如果是完整的（输出文字），那么 DiT 只采用中间层吗，训练时是如何统一训练的）

### LLM 的输入输出处理？

（别人的参考音频经过 VAE 处理和 MLLM 处理，这个为什么是 Encoder -> LLM 处理）





