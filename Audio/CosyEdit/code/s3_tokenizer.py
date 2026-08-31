# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue, Junyang Chen)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
CosyEdit / CosyVoice S³ 语音分词器 (Speech Semantic Similarity Tokenizer)

S³ tokenizer 是 CosyVoice 中使用的离散语音 tokenizer，用于将连续语音
转换为离散的语义 token 序列。在 CosyEdit 中保持不变。

核心特性：
1. 基于自监督语音表示 + K-Means 聚类的离散化
2. 语义级别的 token（而非声学级），适合 LLM 处理
3. 单层离散 token（与 EnCodec 的多层 RVQ 不同）
4. 帧率约 50Hz（与 EnCodec 类似）

架构流程：
  波形 → 梅尔频谱图 → SSL 编码器 (Whisper / WavLM) → 语义特征 → K-Means 量化 → 离散 token

参考文献:
- CosyVoice: https://github.com/FunAudioLLM/CosyVoice
- S³: Speech Semantic Similarity tokenizer
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 第一部分：梅尔频谱提取
# ============================================================

class LogMelFeatureExtractor(nn.Module):
    """
    对数梅尔频谱图特征提取器

    将波形音频转换为对数梅尔频谱图，作为语音 tokenizer 的输入。

    Args:
        sample_rate: 采样率 (默认 22050 Hz)
        n_fft: FFT 大小
        hop_length: 帧移
        win_length: 窗长
        n_mels: 梅尔频带数
        f_min: 最低频率
        f_max: 最高频率
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels

        # 梅尔滤波器组
        from torchaudio.transforms import MelScale
        self.mel_scale = MelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=f_max,
            n_stft=n_fft // 2 + 1,
        )

        # 窗口函数
        self.register_buffer('window', torch.hann_window(win_length))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [B, T] 波形音频

        Returns:
            log_mel: [B, n_mels, T_frames] 对数梅尔频谱图
        """
        # STFT
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            center=True,
        )

        # 幅度谱
        magnitude = stft.abs().pow(2)

        # 梅尔谱
        mel_spec = self.mel_scale(magnitude)

        # 对数梅尔谱
        log_mel = torch.log(torch.clamp(mel_spec, min=1e-5))

        return log_mel


# ============================================================
# 第二部分：自监督语音编码器 (Whisper Encoder)
# ============================================================

class WhisperSSLFeatureExtractor(nn.Module):
    """
    Whisper 编码器作为自监督语音特征提取器

    使用 Whisper 编码器从梅尔频谱图提取高层语义特征。
    这是 S³ tokenizer 的核心组件——提取语义表征。

    注意：在实际 CosyVoice 中，可能使用的是 Whisper 编码器的中间层输出，
    或者其他 SSL 模型（如 WavLM、Hubert 等）。

    Args:
        d_model: Whisper 编码器隐藏层维度
        n_head: 注意力头数
        n_layer: 编码器层数
        n_mels: 输入梅尔频带数
    """

    def __init__(
        self,
        d_model: int = 1024,
        n_head: int = 16,
        n_layer: int = 24,
        n_mels: int = 80,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_mels = n_mels

        # 输入卷积（Whisper 风格：两个 1D 卷积下采样）
        self.conv1 = nn.Conv1d(n_mels, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.gelu = nn.GELU()

        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        从梅尔频谱图提取 SSL 语义特征

        Args:
            mel_spec: [B, n_mels, T] 梅尔频谱图

        Returns:
            ssl_features: [B, T', D_model] 自监督语义特征
        """
        # 输入卷积
        x = self.gelu(self.conv1(mel_spec))
        x = self.gelu(self.conv2(x))  # [B, D, T/2]

        x = x.transpose(1, 2)  # [B, T/2, D]

        # Transformer 编码器
        x = self.encoder(x)
        x = self.layer_norm(x)

        return x


# ============================================================
# 第三部分：K-Means 离散化
# ============================================================

class KMeansQuantizer(nn.Module):
    """
    K-Means 量化器

    将连续的 SSL 特征向量量化为最近的聚类中心索引（离散 token）。

    S³ tokenizer 使用 K-Means 聚类构建码本，将连续的语义特征空间
    离散化为有限的 token 集合。

    Args:
        n_clusters: 聚类中心数量（词表大小）
        d_feat: 输入特征维度
        codebook_path: 预训练码本文件路径
    """

    def __init__(self, n_clusters: int, d_feat: int, codebook_path: Optional[str] = None):
        super().__init__()

        self.n_clusters = n_clusters
        self.d_feat = d_feat

        # 码本（聚类中心）
        self.register_buffer(
            'codebook',
            torch.randn(n_clusters, d_feat)
        )

        if codebook_path is not None:
            self._load_codebook(codebook_path)

    def _load_codebook(self, codebook_path: str):
        """加载预训练的 K-Means 码本"""
        import numpy as np
        codebook = np.load(codebook_path)
        self.codebook = nn.Parameter(
            torch.from_numpy(codebook).float(),
            requires_grad=False
        )
        assert codebook.shape == (self.n_clusters, self.d_feat), \
            f"Codebook shape mismatch: {codebook.shape} vs ({self.n_clusters}, {self.d_feat})"

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        将连续特征量化为离散 token id

        Args:
            features: [B, T, D] 连续特征

        Returns:
            token_ids: [B, T] 离散 token id (LongTensor)
        """
        B, T, D = features.shape

        # 计算每个特征向量到所有聚类中心的距离
        # features: [B, T, D] -> [B*T, D]
        flat_features = features.reshape(-1, D)
        # codebook: [K, D]
        # distances: [B*T, K]
        distances = torch.cdist(flat_features.unsqueeze(0), self.codebook.unsqueeze(0)).squeeze(0)

        # 取最近邻
        token_ids = torch.argmin(distances, dim=-1)  # [B*T]
        token_ids = token_ids.reshape(B, T).long()

        return token_ids

    def decode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        将离散 token id 还原为聚类中心向量（用于可视化或分析）

        Args:
            token_ids: [B, T] token id

        Returns:
            features: [B, T, D] 聚类中心向量
        """
        return F.embedding(token_ids, self.codebook)


# ============================================================
# 第四部分：S³ 语音分词器（完整流程）
# ============================================================

class S3SpeechTokenizer(nn.Module):
    """
    S³ (Speech Semantic Similarity) 语音分词器

    CosyVoice / CosyEdit 使用的离散语音 tokenizer。
    将连续波形音频转换为离散的语义 token 序列。

    完整流程：
    波形 → 对数梅尔谱 → Whisper SSL 编码器 → K-Means 量化 → 离散语义 token

    特点：
    - 单层级离散化（vs EnCodec 的多层 RVQ）
    - 语义级别的 token（适合 LLM 处理）
    - 帧率约 25-50 Hz（取决于下采样率）
    - 词表大小通常为 4096 或 8192

    Args:
        sample_rate: 音频采样率
        n_mels: 梅尔频带数
        n_fft: FFT 大小
        hop_length: 帧移
        d_ssl: SSL 特征维度
        n_ssl_layers: SSL 编码器层数
        n_ssl_heads: SSL 注意力头数
        n_clusters: K-Means 聚类数（词表大小）
        codebook_path: 码本文件路径
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 256,
        d_ssl: int = 1024,
        n_ssl_layers: int = 24,
        n_ssl_heads: int = 16,
        n_clusters: int = 4096,
        codebook_path: Optional[str] = None,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_clusters = n_clusters
        self.vocab_size = n_clusters  # 别名

        # 1. 梅尔频谱提取
        self.mel_extractor = LogMelFeatureExtractor(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

        # 2. SSL 特征提取（Whisper 编码器）
        self.ssl_encoder = WhisperSSLFeatureExtractor(
            d_model=d_ssl,
            n_head=n_ssl_heads,
            n_layer=n_ssl_layers,
            n_mels=n_mels,
        )

        # 3. K-Means 量化器
        self.quantizer = KMeansQuantizer(
            n_clusters=n_clusters,
            d_feat=d_ssl,
            codebook_path=codebook_path,
        )

        # 帧率计算
        # 原始帧率: sample_rate / hop_length
        # Whisper 卷积下采样: / 2
        self.frame_rate = sample_rate / hop_length / 2

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor, waveform_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将波形音频编码为离散语义 token

        Args:
            waveform: [B, T] 波形音频
            waveform_lengths: [B] 每个音频的有效长度（可选）

        Returns:
            tokens: [B, T'] 离散语义 token (LongTensor)
            token_lengths: [B] 每个样本的 token 数
        """
        # Step 1: 提取梅尔频谱
        log_mel = self.mel_extractor(waveform)  # [B, n_mels, T_mel]

        # Step 2: SSL 特征提取
        ssl_features = self.ssl_encoder(log_mel)  # [B, T_ssl, D]

        # Step 3: K-Means 量化
        tokens = self.quantizer(ssl_features)  # [B, T_ssl]

        # 计算 token 长度
        if waveform_lengths is not None:
            # 近似：按比例计算
            mel_frames = torch.ceil(waveform_lengths / self.mel_extractor.hop_length).long()
            token_lengths = torch.ceil(mel_frames.float() / 2).long()  # /2 for conv2 stride
        else:
            token_lengths = torch.full(
                (tokens.shape[0],), tokens.shape[1],
                dtype=torch.long, device=tokens.device
            )

        return tokens, token_lengths

    @torch.no_grad()
    def encode_from_mel(self, log_mel: torch.Tensor) -> torch.Tensor:
        """
        从梅尔频谱直接编码（用于 GOT-CFM 等场景）

        Args:
            log_mel: [B, n_mels, T] 对数梅尔频谱图

        Returns:
            tokens: [B, T'] 离散语义 token
        """
        ssl_features = self.ssl_encoder(log_mel)
        tokens = self.quantizer(ssl_features)
        return tokens


# ============================================================
# 第五部分：特殊 token 定义
# ============================================================

class S3SpecialTokens:
    """
    S³ 语音 tokenizer 的特殊 token 定义

    在 CosyVoice / CosyEdit 中，语音 token 词表会添加一些特殊 token，
    用于控制序列的开始、结束、分隔等。
    """

    # 基于 K-Means 的语义 token 范围: [0, n_clusters-1]
    # 特殊 token 从 n_clusters 开始编号

    @staticmethod
    def get_special_tokens(n_clusters: int = 4096) -> dict:
        """
        获取特殊 token 映射

        Args:
            n_clusters: K-Means 聚类数

        Returns:
            dict: 特殊 token 名称到 id 的映射
        """
        return {
            # 语义 token: 0 ~ n_clusters-1
            'eos': n_clusters,          # End of Sequence (也用作 sos)
            'pad': n_clusters + 1,      # Padding
            'unk': n_clusters + 2,      # Unknown
            'sep': n_clusters + 3,      # Separator (分隔文本和语音)
            'trans_sep': n_clusters + 4, # Transition Separator (OICL/ZICL 中的过渡 token)
        }

    @classmethod
    def total_vocab_size(cls, n_clusters: int = 4096) -> int:
        """总词表大小（语义 token + 特殊 token）"""
        return n_clusters + 5  # 5 个特殊 token


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CosyEdit / CosyVoice S³ 语音分词器")
    print("=" * 70)

    print("""
架构说明：
  波形音频 [B, T]
    │
    ▼  STFT + Mel Filterbank
  对数梅尔谱 [B, 80, T_mel]
    │
    ▼  Whisper SSL Encoder (24 层)
  语义特征 [B, T_ssl, 1024]
    │
    ▼  K-Means 量化
  离散语义 token [B, T_ssl] (0 ~ 4095)

关键参数：
  采样率:          22050 Hz
  梅尔频带数:      80
  SSL 特征维度:    1024
  SSL 层数:        24 (Whisper base/large 配置)
  K-Means 聚类数:  4096 (词表大小)
  帧率:            ~43 Hz (22050 / 256 / 2 ≈ 43.1)
  特殊 token 数:   5 (eos, pad, unk, sep, trans_sep)

在 CosyEdit 中的作用：
  1. 将原始语音编码为 μ_ori（条件输入）
  2. 将目标语音编码为 μ_tar（训练目标）
  3. LLM 自回归预测 μ_tar
  4. 预测的 token 序列送入 GOT-CFM 生成梅尔谱

与其他 tokenizer 的对比：
  - EnCodec: 多层 RVQ (4-8层)，声学级，连续 + 离散
  - S³: 单层 K-Means，语义级，纯离散
  - Vocos / SoundStream: 类似 EnCodec
    """)
