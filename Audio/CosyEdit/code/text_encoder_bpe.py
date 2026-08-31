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
CosyEdit / CosyVoice 文本编码器 + BPE 分词器

从 CosyVoice 继承的文本处理模块，在 CosyEdit 中保持不变。

功能：
1. BPE (Byte-Pair Encoding) 分词：将文本转换为 subword token
2. 文本编码器：基于 Transformer 的文本编码，输出文本嵌入序列

架构：
- BPE Tokenizer: sentencepiece / tokenizers 库实现
- Text Encoder: 多层 Transformer Encoder
  - 输入: BPE token ids [B, L]
  - 输出: 文本编码序列 [B, L, D_text_enc]

参考文献:
- CosyVoice: https://github.com/FunAudioLLM/CosyVoice
- CosyEdit: https://github.com/FunAudioLLM/CosyEdit
"""

import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 第一部分：BPE 分词器
# ============================================================

class BPETokenizer:
    """
    BPE (Byte-Pair Encoding) 文本分词器

    基于 sentencepiece 或 tokenizers 库实现。
    将原始文本转换为 subword token id 序列。

    CosyVoice / CosyEdit 使用 BPE 对多语言文本进行分词，
    支持中文、英文、日语、粤语、韩语等多种语言。

    Args:
        vocab_file: BPE 词表文件路径
        model_type: 模型类型 ('bpe' 或 'unigram')
    """

    def __init__(self, vocab_file: str, model_type: str = 'bpe'):
        self.vocab_file = vocab_file
        self.model_type = model_type

        # 特殊 token
        self.special_tokens = {
            '<sos/eos>': -1,  # 起始/结束 token
            '<blank>': 0,     # 空白 token (CTC 风格)
            '<unk>': 1,       # 未知 token
        }

        self._tokenizer = None
        self._load_tokenizer()

    def _load_tokenizer(self):
        """加载 BPE 分词器模型"""
        try:
            # 尝试使用 sentencepiece
            import sentencepiece as spm
            self._tokenizer = spm.SentencePieceProcessor()
            self._tokenizer.Load(self.vocab_file)
            self._backend = 'sentencepiece'
        except ImportError:
            try:
                # 备选: tokenizers 库
                from tokenizers import Tokenizer
                self._tokenizer = Tokenizer.from_file(self.vocab_file)
                self._backend = 'tokenizers'
            except ImportError:
                raise ImportError("Please install sentencepiece or tokenizers")

    @property
    def vocab_size(self) -> int:
        """词表大小"""
        if self._backend == 'sentencepiece':
            return self._tokenizer.get_piece_size()
        else:
            return self._tokenizer.get_vocab_size()

    def encode(self, text: str) -> List[int]:
        """
        将文本编码为 token id 列表

        Args:
            text: 输入文本

        Returns:
            list of token ids
        """
        if self._backend == 'sentencepiece':
            return self._tokenizer.encode(text, out_type=int)
        else:
            return self._tokenizer.encode(text).ids

    def decode(self, token_ids: List[int]) -> str:
        """
        将 token id 列表解码为文本

        Args:
            token_ids: token id 列表

        Returns:
            解码后的文本
        """
        if self._backend == 'sentencepiece':
            return self._tokenizer.decode(token_ids)
        else:
            return self._tokenizer.decode(token_ids)

    def tokenize(self, text: str) -> List[str]:
        """
        将文本分词为 subword 字符串列表

        Args:
            text: 输入文本

        Returns:
            list of subword tokens
        """
        if self._backend == 'sentencepiece':
            return self._tokenizer.encode(text, out_type=str)
        else:
            return self._tokenizer.encode(text).tokens


# ============================================================
# 第二部分：Transformer 文本编码器
# ============================================================

class MultiHeadAttention(nn.Module):
    """多头注意力层"""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]
            mask: [B, L] padding mask (True = masked)

        Returns:
            [B, L, D]
        """
        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.nhead, self.d_k).transpose(1, 2)  # [B, H, L, d]
        k = self.k_proj(x).view(B, L, self.nhead, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.nhead, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)

        if mask is not None:
            # mask: [B, L] -> [B, 1, 1, L]
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)  # [B, H, L, d]
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        out = self.out_proj(out)

        return out


class FeedForward(nn.Module):
    """前馈网络"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = 'relu'):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU() if activation == 'relu' else nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.activation(self.fc1(x))))


class TransformerEncoderLayer(nn.Module):
    """Transformer 编码器层"""

    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float = 0.1,
                 activation: str = 'relu', normalize_before: bool = True):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.normalize_before = normalize_before

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN or Post-LN
        if self.normalize_before:
            x = x + self.dropout1(self.self_attn(self.norm1(x), mask))
            x = x + self.dropout2(self.feed_forward(self.norm2(x)))
        else:
            x = self.norm1(x + self.dropout1(self.self_attn(x, mask)))
            x = self.norm2(x + self.dropout2(self.feed_forward(x)))
        return x


class PositionalEncoding(nn.Module):
    """正弦位置编码"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, D]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]

        Returns:
            [B, L, D]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TextEncoder(nn.Module):
    """
    CosyVoice / CosyEdit 文本编码器

    基于 Transformer Encoder 的文本编码模型。
    将 BPE token 序列编码为文本特征序列，供 LLM 使用。

    架构：
    Token Embedding → Positional Encoding → N × Transformer Encoder Layer → Output

    Args:
        vocab_size: BPE 词表大小
        d_model: 隐藏层维度（默认 512）
        nhead: 注意力头数（默认 8）
        num_layers: 编码器层数（默认 6）
        d_ff: 前馈网络维度（默认 2048）
        dropout: dropout 概率
        activation: 激活函数 ('relu' 或 'gelu')
        normalize_before: 是否使用 Pre-LN
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = 'relu',
        normalize_before: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # Token 嵌入
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        # Transformer 编码器层
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                normalize_before=normalize_before,
            )
            for _ in range(num_layers)
        ])

        # 最终层归一化（Pre-LN 架构需要）
        if normalize_before:
            self.final_norm = nn.LayerNorm(d_model)
        else:
            self.final_norm = None

    def forward(self, text_tokens: torch.Tensor, text_lengths: torch.Tensor) -> torch.Tensor:
        """
        编码文本 token 序列

        Args:
            text_tokens: [B, L] BPE token id 序列
            text_lengths: [B] 每个样本的有效长度

        Returns:
            text_enc: [B, L, D_text_enc] 编码后的文本特征
        """
        B, L = text_tokens.shape

        # Embedding
        x = self.token_embedding(text_tokens)  # [B, L, D]

        # Positional encoding
        x = self.positional_encoding(x)

        # Padding mask
        mask = torch.arange(L, device=text_tokens.device).unsqueeze(0) >= text_lengths.unsqueeze(1)
        # mask: [B, L], True = masked/padding

        # Transformer encoder layers
        for layer in self.layers:
            x = layer(x, mask)

        # Final norm
        if self.final_norm is not None:
            x = self.final_norm(x)

        return x


# ============================================================
# 第三部分：文本前端（完整文本处理流水线）
# ============================================================

class TextFrontend(nn.Module):
    """
    CosyVoice / CosyEdit 文本前端

    完整的文本处理流水线：
    原始文本 → 文本规范化 → BPE 分词 → 文本编码器 → 文本特征

    在 CosyEdit 中，文本前端完全继承自 CosyVoice，没有改动。

    Args:
        bpe_model_path: BPE 模型文件路径
        text_encoder_config: 文本编码器配置字典
    """

    def __init__(self, bpe_model_path: str, text_encoder_config: dict):
        super().__init__()

        # BPE 分词器
        self.tokenizer = BPETokenizer(bpe_model_path)

        # 文本编码器
        self.encoder = TextEncoder(
            vocab_size=self.tokenizer.vocab_size,
            **text_encoder_config,
        )

    @property
    def d_model(self) -> int:
        return self.encoder.d_model

    def encode_text(self, texts: List[str], device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批量编码文本

        Args:
            texts: 文本列表
            device: 计算设备

        Returns:
            text_enc: [B, L_max, D] 编码后的文本特征
            text_lengths: [B] 每个文本的长度
        """
        # BPE 编码
        token_ids_list = [self.tokenizer.encode(text) for text in texts]
        text_lengths = [len(ids) for ids in token_ids_list]
        max_len = max(text_lengths)

        # Padding
        padded_ids = []
        for ids, length in zip(token_ids_list, text_lengths):
            pad_len = max_len - length
            padded_ids.append(ids + [0] * pad_len)  # 0 = <blank> / pad

        text_tokens = torch.tensor(padded_ids, dtype=torch.long, device=device)
        text_lengths = torch.tensor(text_lengths, dtype=torch.long, device=device)

        # 编码
        text_enc = self.encoder(text_tokens, text_lengths)

        return text_enc, text_lengths


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CosyEdit / CosyVoice 文本编码器 + BPE 分词器")
    print("=" * 70)

    print("""
架构说明：
  原始文本
    │
    ▼  BPE Tokenizer (sentencepiece)
  BPE token ids [B, L]
    │
    ▼  Token Embedding + Positional Encoding
  嵌入向量 [B, L, D]
    │
    ▼  N × Transformer Encoder Layer
  文本编码 [B, L, D_text_enc]
    │
    ▼  送入 LLM 主干（与语音 token 拼接）

关键参数（CosyVoice-300M 默认配置）：
  BPE 词表大小:   ~50,000
  文本编码器维度:  512
  注意力头数:      8
  编码器层数:      6
  FFN 维度:        2048
  激活函数:        ReLU
  归一化方式:      Pre-LN

在 CosyEdit 中的角色：
  - 完全继承自 CosyVoice，无改动
  - 用于编码目标文本 (target text) 和原始文本 (original text)
  - OICL 模式: 编码原始文本 + 目标文本
  - ZICL 模式: 只编码目标文本
    """)
