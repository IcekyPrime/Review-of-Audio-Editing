# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
VoiceCraft 中 EnCodec 编解码集成代码

VoiceCraft 使用 EnCodec 作为音频的离散 tokenizer：
- 编码（Encode）：将原始波形音频编码为离散的 RVQ（残差向量量化）token
- 解码（Decode）：将离散 token 解码回波形音频

VoiceCraft 本身不包含 EnCodec 的实现，而是使用 Facebook/Meta 开源的
`encodec` Python 库中的预训练模型。本文档提取 VoiceCraft 中与 EnCodec
相关的参数、调用方式和集成逻辑。

EnCodec 关键特性：
- 24kHz 采样率的单声道音频
- 8 个码本（codebooks）的 RVQ 量化
- 每个码本 1024 个词元（VoiceCraft 配置使用 n_codebooks=4 或 8）
- 50Hz 的 token 帧率（即每秒 50 个时间步）
- 预训练模型 checkpoint: 6f79c6a8

参考文献:
- EnCodec: https://github.com/facebookresearch/encodec
- VoiceCraft: https://github.com/jasonppy/VoiceCraft
"""

import torch
import torch.nn as nn
import torchaudio
from typing import Optional, Tuple, List


# ============================================================
# 第一部分：EnCodec 模型封装
# 基于 encodec 库的 EncodecModel 封装，提供音频编码/解码接口
# ============================================================

class EnCodecWrapper(nn.Module):
    """
    EnCodec 封装器 - 提供音频编码和解码的统一接口

    VoiceCraft 使用 EnCodec 将音频转换为离散 token，或将生成的 token
    还原为波形音频。

    参数说明（VoiceCraft 中使用的配置）：
    - n_codebooks: 码本数量，VoiceCraft 使用 4 或 8
    - audio_vocab_size: 每个码本的词表大小，通常为 1024（或 2048）
    - encodec_sr: EnCodec 的 token 采样率，约为 50Hz（即 24kHz / 480 跳）
    - sample_rate: 原始音频采样率，24000 Hz
    """

    def __init__(
        self,
        n_codebooks: int = 8,
        audio_vocab_size: int = 2048,
        sample_rate: int = 24000,
        encodec_sr: int = 50,  # token per second (approximate)
        pretrained_model_name: str = "encodec_24khz",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()

        self.n_codebooks = n_codebooks
        self.audio_vocab_size = audio_vocab_size
        self.sample_rate = sample_rate
        self.encodec_sr = encodec_sr
        self.device = device

        # 加载预训练 EnCodec 模型
        # VoiceCraft 使用的是 facebook/encodec 的预训练模型
        self.model = self._load_encodec_model(pretrained_model_name)
        self.model.eval()
        self.model.to(device)

        # 静音 token（特定于 EnCodec 预训练模型 6f79c6a8）
        # 这些是 VoiceCraft 中硬编码的默认静音 token
        self.silence_tokens = [1388, 1898, 131]

    def _load_encodec_model(self, model_name: str):
        """
        加载预训练的 EnCodec 模型

        VoiceCraft 使用的是 encodec 库中的 24kHz 单声道模型。
        实际使用时需要安装 encodec 库：pip install encodec
        """
        try:
            from encodec import EncodecModel
            from encodec.utils import convert_audio

            self._convert_audio = convert_audio

            # 加载 24kHz 单声道预训练模型
            # checkpoint 6f79c6a8 是 VoiceCraft 论文中使用的版本
            model = EncodecModel.encodec_model_24khz()
            model.set_target_bandwidth(6.0)  # 6kbps 对应 8 个码本

            return model
        except ImportError:
            print("警告: 未安装 encodec 库。请运行: pip install encodec")
            print("将使用占位符模式，实际编码/解码功能不可用。")
            return None

    @torch.no_grad()
    def encode(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        将波形音频编码为离散 token

        Args:
            audio: [B, 1, T] 或 [B, T] 波形音频
            sample_rate: 输入音频的采样率

        Returns:
            codes: [B, K, T] 离散 token，其中 K 为码本数量，T 为时间步数
        """
        if self.model is None:
            raise RuntimeError("EnCodec 模型未加载，请先安装 encodec 库")

        # 确保音频形状正确: [B, channels, T]
        if audio.ndim == 2:
            audio = audio.unsqueeze(1)  # [B, T] -> [B, 1, T]

        # 转换为 EnCodec 所需的格式和采样率
        audio = self._convert_audio(
            audio, sample_rate, self.model.sample_rate, self.model.channels
        )
        audio = audio.to(self.device)

        # 编码
        with torch.no_grad():
            encoded_frames = self.model.encode(audio)

        # 提取 codes
        # encoded_frames 是一个 list，每个元素是 (codes, scale)
        # codes 的形状是 [B, K, T]
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)

        # 如果使用的码本数量少于总码本数，截取前 n_codebooks 个
        if codes.shape[1] > self.n_codebooks:
            codes = codes[:, :self.n_codebooks, :]

        return codes

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        将离散 token 解码为波形音频

        Args:
            codes: [B, K, T] 离散 token

        Returns:
            audio: [B, 1, T] 波形音频
        """
        if self.model is None:
            raise RuntimeError("EnCodec 模型未加载，请先安装 encodec 库")

        codes = codes.to(self.device)

        # 如果码本数少于模型要求，填充空 token
        if codes.shape[1] < self.model.quantizer.n_q:
            padding = torch.zeros(
                codes.shape[0],
                self.model.quantizer.n_q - codes.shape[1],
                codes.shape[2],
                dtype=codes.dtype,
                device=codes.device,
            )
            codes = torch.cat([codes, padding], dim=1)

        # 构建 encoded_frames 格式
        encoded_frames = [(codes, None)]

        # 解码
        with torch.no_grad():
            audio = self.model.decode(encoded_frames)

        return audio

    def encode_file(self, audio_path: str) -> torch.Tensor:
        """
        从文件加载音频并编码为 token

        Args:
            audio_path: 音频文件路径

        Returns:
            codes: [1, K, T] 离散 token
        """
        audio, sr = torchaudio.load(audio_path)
        audio = audio.unsqueeze(0)  # [1, C, T]
        return self.encode(audio, sr)

    def decode_to_file(self, codes: torch.Tensor, output_path: str):
        """
        将 token 解码为音频并保存到文件

        Args:
            codes: [1, K, T] 离散 token
            output_path: 输出文件路径
        """
        audio = self.decode(codes)
        audio = audio.squeeze(0).cpu()  # [C, T]
        torchaudio.save(output_path, audio, self.sample_rate)


# ============================================================
# 第二部分：VoiceCraft 中与 EnCodec 相关的参数和集成
# ============================================================

class VoiceCraftEncodecConfig:
    """
    VoiceCraft 中 EnCodec 相关的配置参数

    这些参数定义了 VoiceCraft 如何使用 EnCodec 的输出 token。
    """

    # EnCodec 基础参数
    N_CODEBOOKS = 8  # 码本数量（VoiceCraft 也有使用 4 的变体）
    AUDIO_VOCAB_SIZE = 2048  # 每个码本的词表大小
    SAMPLE_RATE = 24000  # 音频采样率 (Hz)
    ENCODEC_SR = 50  # token 采样率 (Hz)，约 24000 / 480 = 50

    # 特殊 token 定义（基于 EnCodec 词表扩展）
    # VoiceCraft 在 EnCodec 词表基础上添加了 3 个特殊 token
    N_SPECIAL_TOKENS = 3

    # 特殊 token 索引（相对于 EnCodec 词表的偏移）
    # empty_token = audio_vocab_size (即 2048)
    # eog_token = audio_vocab_size + 1 (即 2049) - End of Generation
    # audio_pad_token = audio_vocab_size + 2 (即 2050)

    # 预训练模型信息
    PRETRAINED_CHECKPOINT = "6f79c6a8"
    PRETRAINED_BANDWIDTH = "6kbps"  # 对应 8 个码本

    # 静音 token（特定于预训练模型 6f79c6a8）
    # 这些 token 代表静音/静默，VoiceCraft 在推理时会做特殊处理
    SILENCE_TOKENS = [1388, 1898, 131]

    @classmethod
    def get_special_token_mapping(cls, audio_vocab_size: int = 2048) -> dict:
        """
        获取特殊 token 的映射关系

        Returns:
            dict: 包含各特殊 token 的索引
        """
        return {
            "empty_token": audio_vocab_size,
            "eog_token": audio_vocab_size + 1,  # End of Generation
            "audio_pad_token": audio_vocab_size + 2,
            "total_vocab_size": audio_vocab_size + cls.N_SPECIAL_TOKENS,
        }

    @classmethod
    def audio_frames_to_tokens(cls, n_audio_samples: int) -> int:
        """
        将音频样本数转换为 EnCodec token 数

        EnCodec 24kHz 模型的 hop length 约为 480（24000 / 50 = 480）
        """
        hop_length = cls.SAMPLE_RATE // cls.ENCODEC_SR  # 480
        return n_audio_samples // hop_length

    @classmethod
    def tokens_to_audio_frames(cls, n_tokens: int) -> int:
        """
        将 EnCodec token 数转换为音频样本数
        """
        hop_length = cls.SAMPLE_RATE // cls.ENCODEC_SR  # 480
        return n_tokens * hop_length


# ============================================================
# 第三部分：VoiceCraft 数据预处理中的 EnCodec 使用
# ============================================================

class VoiceCraftDataPreprocessor:
    """
    VoiceCraft 数据预处理器 - 展示如何在数据处理中使用 EnCodec

    VoiceCraft 的训练数据处理流程：
    1. 加载原始音频
    2. 使用 EnCodec 编码为离散 token
    3. 处理文本（phoneme 或 text tokenizer）
    4. 构建训练样本对 (text, audio_tokens)
    """

    def __init__(
        self,
        n_codebooks: int = 8,
        audio_vocab_size: int = 2048,
        sample_rate: int = 24000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.n_codebooks = n_codebooks
        self.audio_vocab_size = audio_vocab_size
        self.sample_rate = sample_rate
        self.device = device

        # 初始化 EnCodec
        self.encodec = EnCodecWrapper(
            n_codebooks=n_codebooks,
            audio_vocab_size=audio_vocab_size,
            sample_rate=sample_rate,
            device=device,
        )

        # 特殊 token
        self.empty_token = audio_vocab_size
        self.eog_token = audio_vocab_size + 1
        self.audio_pad_token = audio_vocab_size + 2

    def process_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """
        处理单个音频文件

        Args:
            audio_path: 音频文件路径

        Returns:
            codes: [K, T] EnCodec token
            n_tokens: token 数量
        """
        codes = self.encodec.encode_file(audio_path)
        codes = codes.squeeze(0)  # [K, T]
        n_tokens = codes.shape[1]
        return codes, n_tokens

    def process_batch(
        self,
        audio_paths: List[str],
        max_tokens: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批量处理音频文件，返回 padding 后的 batch

        Args:
            audio_paths: 音频文件路径列表
            max_tokens: 最大 token 数（用于截断）

        Returns:
            codes_batch: [B, K, T] padding 后的 token
            lengths: [B] 每个样本的实际长度
        """
        all_codes = []
        all_lengths = []

        for path in audio_paths:
            codes, n_tokens = self.process_audio(path)
            if max_tokens and n_tokens > max_tokens:
                codes = codes[:, :max_tokens]
                n_tokens = max_tokens
            all_codes.append(codes)  # [K, T]
            all_lengths.append(n_tokens)

        # Padding
        max_len = max(all_lengths)
        padded_codes = []
        for codes in all_codes:
            if codes.shape[1] < max_len:
                pad = torch.full(
                    (self.n_codebooks, max_len - codes.shape[1]),
                    self.audio_pad_token,
                    dtype=codes.dtype,
                )
                codes = torch.cat([codes, pad], dim=1)
            padded_codes.append(codes)

        codes_batch = torch.stack(padded_codes, dim=0)  # [B, K, T]
        lengths = torch.tensor(all_lengths, dtype=torch.long)

        return codes_batch, lengths


# ============================================================
# 第四部分：VoiceCraft 推理中 EnCodec 的使用流程
# ============================================================

class VoiceCraftInferenceEncodec:
    """
    VoiceCraft 推理时的 EnCodec 使用流程

    完整的语音编辑推理流程：
    1. 加载原始音频 -> EnCodec 编码得到 token
    2. 指定编辑区间（mask interval）
    3. VoiceCraft 模型生成新的 token 填充掩码区域
    4. 将完整的 token 序列 -> EnCodec 解码得到波形音频
    """

    def __init__(
        self,
        voicecraft_model: nn.Module,
        n_codebooks: int = 8,
        audio_vocab_size: int = 2048,
        sample_rate: int = 24000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.voicecraft_model = voicecraft_model
        self.n_codebooks = n_codebooks
        self.audio_vocab_size = audio_vocab_size
        self.sample_rate = sample_rate
        self.device = device

        # 初始化 EnCodec
        self.encodec = EnCodecWrapper(
            n_codebooks=n_codebooks,
            audio_vocab_size=audio_vocab_size,
            sample_rate=sample_rate,
            device=device,
        )

    def edit_audio(
        self,
        audio_path: str,
        text_tokens: torch.Tensor,
        mask_interval: List[Tuple[float, float]],  # 以秒为单位
        top_k: int = 100,
        top_p: float = 0.9,
        temperature: float = 0.8,
    ) -> Tuple[torch.Tensor, int]:
        """
        执行语音编辑

        Args:
            audio_path: 原始音频路径
            text_tokens: [1, L] 文本 token
            mask_interval: 编辑区间列表，每个区间为 (start_sec, end_sec)
            top_k: top-k 采样参数
            top_p: top-p 采样参数
            temperature: 温度参数

        Returns:
            output_audio: [1, T] 输出音频波形
            sample_rate: 采样率
        """
        # Step 1: 编码音频
        audio_codes = self.encodec.encode_file(audio_path)  # [1, K, T]
        audio_codes = audio_codes.permute(0, 2, 1)  # [1, T, K]

        # Step 2: 将时间区间转换为 token 索引
        mask_interval_tokens = []
        for start_sec, end_sec in mask_interval:
            start_token = int(start_sec * self.encodec.encodec_sr)
            end_token = int(end_sec * self.encodec.encodec_sr)
            mask_interval_tokens.append([start_token, end_token])

        mask_interval_tensor = torch.tensor(
            mask_interval_tokens, dtype=torch.long
        ).unsqueeze(0)  # [1, M, 2]

        # Step 3: VoiceCraft 推理生成
        x_lens = torch.tensor([text_tokens.shape[1]], dtype=torch.long)
        x_lens = x_lens.to(self.device)
        text_tokens = text_tokens.to(self.device)
        audio_codes = audio_codes.to(self.device)
        mask_interval_tensor = mask_interval_tensor.to(self.device)

        with torch.no_grad():
            output_codes = self.voicecraft_model.inference(
                x=text_tokens,
                x_lens=x_lens,
                y=audio_codes,
                mask_interval=mask_interval_tensor,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )  # [1, K, new_T]

        # Step 4: 解码为波形
        output_audio = self.encodec.decode(output_codes)  # [1, 1, T]
        output_audio = output_audio.squeeze(1)  # [1, T]

        return output_audio, self.sample_rate

    def edit_audio_file(
        self,
        input_path: str,
        output_path: str,
        text_tokens: torch.Tensor,
        mask_interval: List[Tuple[float, float]],
        **kwargs,
    ):
        """
        编辑音频文件并保存

        Args:
            input_path: 输入音频路径
            output_path: 输出音频路径
            text_tokens: 文本 token
            mask_interval: 编辑区间（秒）
        """
        output_audio, sr = self.edit_audio(
            input_path, text_tokens, mask_interval, **kwargs
        )
        torchaudio.save(output_path, output_audio.cpu(), sr)


# ============================================================
# 第五部分：EnCodec RVQ（残差向量量化）原理说明
# ============================================================

"""
EnCodec RVQ (Residual Vector Quantization) 原理简述：

EnCodec 使用残差向量量化将连续的音频表示离散化为多个码本的 token。

基本思想：
1. 第一个码本量化器对编码器输出进行量化，得到第一个量化结果
2. 计算量化残差（原始输出 - 量化结果）
3. 第二个码本量化器对残差进行量化
4. 重复上述过程，直到达到指定的码本数量（VoiceCraft 使用 4 或 8 个）

数学表达：
    z_hat = sum_{k=1}^{K} e_k(c_k)
    其中 c_k 是第 k 个码本的 token 索引，e_k 是第 k 个码本的 embedding

码本延迟模式（DelayedPattern）：
VoiceCraft 使用延迟模式来处理多码本序列的自回归生成：
- 第 0 个码本正常生成
- 第 1 个码本延迟 1 步生成（基于第 0 个码本的输出）
- 第 k 个码本延迟 k 步生成
这样可以确保每个码本的生成都能利用前面码本的信息。
"""


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("VoiceCraft EnCodec 编解码集成说明")
    print("=" * 60)

    # 打印配置信息
    config = VoiceCraftEncodecConfig()
    special_tokens = VoiceCraftEncodecConfig.get_special_token_mapping()

    print("\nEnCodec 配置参数:")
    print(f"  码本数量 (n_codebooks): {config.N_CODEBOOKS}")
    print(f"  词表大小 (vocab_size): {config.AUDIO_VOCAB_SIZE}")
    print(f"  音频采样率: {config.SAMPLE_RATE} Hz")
    print(f"  Token 采样率: {config.ENCODEC_SR} Hz")
    print(f"  预训练 Checkpoint: {config.PRETRAINED_CHECKPOINT}")
    print(f"  静音 token: {config.SILENCE_TOKENS}")

    print("\n特殊 Token 映射:")
    for name, value in special_tokens.items():
        print(f"  {name}: {value}")

    print("\n" + "=" * 60)
    print("VoiceCraft 中 EnCodec 的使用流程:")
    print("=" * 60)
    print("""
训练阶段：
  1. 使用 EnCodec 对训练音频进行编码，得到离散 token [K, T]
  2. VoiceCraft 以文本为条件，学习预测被掩码的 token
  3. 使用交叉熵损失计算每个码本的预测误差

推理阶段（语音编辑）：
  1. 使用 EnCodec 编码原始音频，得到 token
  2. 指定需要编辑的区间（mask interval）
  3. VoiceCraft 自回归生成掩码部分的新 token
  4. 将原始非掩码 token 和生成的 token 拼接
  5. 使用 EnCodec 解码完整 token，得到编辑后的音频波形
    """)

    print("\n" + "=" * 60)
    print("注意事项:")
    print("=" * 60)
    print("1. VoiceCraft 本身不包含 EnCodec 实现，需安装 encodec 库")
    print("   安装命令: pip install encodec")
    print("2. VoiceCraft 使用预训练模型 checkpoint 6f79c6a8")
    print("3. 静音 token 是特定于该 checkpoint 的，换模型需重新指定")
    print("4. EnCodec 的 hop length 约为 480 (24000/50=480)")
    print("5. VoiceCraft 支持使用前 N 个码本（如 4 个）以降低计算量")
