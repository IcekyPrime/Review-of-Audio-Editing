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
CosyEdit LLM 主干 + OICL/ZICL 训练范式

LLM (Large Language Model) 是 CosyEdit 的核心组件，负责将文本条件和
原始语音 token 自回归地转换为目标语音 token。

CosyEdit 基于 CosyVoice 进行后训练（post-training），LLM 架构保持不变，
但通过 OICL 和 ZICL 两种互补的训练范式来适配语音编辑任务。

架构：
- 主干：Transformer Decoder（因果自注意力）
- 输入：说话人嵌入 + 文本编码 + 原始语音 token + 过渡 token
- 输出：目标语音 token（自回归预测）
- 参数量：约 300M (CosyVoice-300M)

OICL (One-shot In-Context Learning):
  输入序列：[SOS, spk_emb, X_ori, X_tar, sep, μ_ori, trans_sep, μ_tar, EOS]
  提供原始文本+语音对作为对齐上下文，帮助模型学习隐式对齐

ZICL (Zero-shot In-Context Learning):
  输入序列：[SOS, spk_emb, X_tar, μ_ori, trans_sep, μ_tar, EOS]
  只提供目标文本和原始语音，迫使模型从差异中推断编辑边界

混合训练：L = λ * L_ZICL + (1-λ) * L_OICL，λ 通常取 0.4

参考文献:
- CosyVoice: https://github.com/FunAudioLLM/CosyVoice
- CosyEdit: https://github.com/FunAudioLLM/CosyEdit
"""

import math
import random
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 第一部分：LLM 基础组件
# ============================================================

class RMSNorm(nn.Module):
    """RMS 归一化"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class RotaryPositionalEmbedding(nn.Module):
    """旋转位置编码 (RoPE)"""

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x: torch.Tensor, seq_len: int = None, offset: int = 0):
        """
        Args:
            x: [B, H, T, D]
            seq_len: 序列长度
            offset: 位置偏移（用于 KV cache）

        Returns:
            cos, sin: [1, 1, T, D]
        """
        if seq_len is None:
            seq_len = x.shape[2]

        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype) + offset
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().unsqueeze(0).unsqueeze(0), emb.sin().unsqueeze(0).unsqueeze(0)


def rotate_half(x):
    """将向量的前半和后半旋转拼接"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """应用旋转位置编码"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class CausalSelfAttention(nn.Module):
    """
    因果自注意力层（LLM 解码器使用）

    支持：
    - Rotary Position Embedding (RoPE)
    - KV Cache（推理加速）
    - Grouped Query Attention (可选)
    """

    def __init__(self, d_model: int, n_head: int, n_kv_head: int = None,
                 dropout: float = 0.0, layer_idx: int = 0):
        super().__init__()
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        self.d_head = d_model // n_head
        self.layer_idx = layer_idx

        # GQA 支持
        assert n_head % self.n_kv_head == 0
        self.n_rep = n_head // self.n_kv_head

        self.q_proj = nn.Linear(d_model, n_head * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_head * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_head * self.d_head, bias=False)
        self.o_proj = nn.Linear(n_head * self.d_head, d_model, bias=False)

        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: [B, L, D]
            rope_cos, rope_sin: RoPE 编码
            mask: 因果注意力 mask
            past_key_value: 过去的 KV cache
            use_cache: 是否返回新的 KV cache

        Returns:
            out: [B, L, D]
            present_key_value: 新的 KV cache（如果 use_cache=True）
        """
        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.n_head, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_kv_head, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_kv_head, self.d_head).transpose(1, 2)

        # RoPE
        q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)

        # KV Cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        # GQA: 扩展 KV 头数匹配 Q 头数
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Scaled Dot-Product Attention（因果）
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=mask is None,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, -1)
        out = self.o_proj(attn_output)

        present_key_value = (k[:, :self.n_kv_head], v[:, :self.n_kv_head]) if use_cache else None

        return out, present_key_value


class FeedForward(nn.Module):
    """
    FFN with SwiGLU activation
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class TransformerDecoderLayer(nn.Module):
    """Transformer 解码器层（Pre-LN + RoPE + SwiGLU FFN）"""

    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float = 0.0,
                 n_kv_head: int = None, layer_idx: int = 0):
        super().__init__()
        self.self_attn = CausalSelfAttention(d_model, n_head, n_kv_head, dropout, layer_idx)
        self.mlp = FeedForward(d_model, d_ff, dropout)
        self.input_layernorm = RMSNorm(d_model)
        self.post_attention_layernorm = RMSNorm(d_model)

    def forward(self, x, rope_cos, rope_sin, mask=None, past_key_value=None, use_cache=False):
        # Self-attention
        residual = x
        x = self.input_layernorm(x)
        x, present_key_value = self.self_attn(
            x, rope_cos, rope_sin, mask, past_key_value, use_cache
        )
        x = residual + x

        # FFN
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x, present_key_value


# ============================================================
# 第二部分：LLM 主干模型
# ============================================================

class CosyVoiceLLM(nn.Module):
    """
    CosyVoice / CosyEdit LLM 主干模型

    基于 Transformer Decoder 的因果语言模型，用于自回归预测语音 token。

    输入组成（按顺序拼接）：
    1. 起始 token (SOS/EOS)
    2. 说话人嵌入 (speaker embedding) - 从原始语音中提取
    3. 文本编码 (text encoding) - BPE + Text Encoder
    4. 原始语音 token (μ_ori) - S³ tokenizer 编码
    5. 过渡 token (trans_sep) - 标记条件和生成的边界
    6. 目标语音 token (μ_tar) - 自回归预测目标
    7. 结束 token (EOS)

    嵌入层设计：
    - 文本编码通过线性投影到 LLM 维度
    - 说话人嵌入通过线性投影到 LLM 维度
    - 语音 token 通过 Embedding 层映射
    - 特殊 token（SOS、分隔符等）也有对应的 embedding

    Args:
        d_model: LLM 隐藏层维度
        n_head: 注意力头数
        n_layer: 解码器层数
        d_ff: FFN 维度
        vocab_size: 语音 token 词表大小（含特殊 token）
        text_enc_dim: 文本编码器输出维度
        spk_emb_dim: 说话人嵌入维度
        n_kv_head: GQA KV 头数（None = MHA）
        dropout: dropout 概率
        max_seq_len: 最大序列长度
    """

    def __init__(
        self,
        d_model: int = 1024,
        n_head: int = 16,
        n_layer: int = 12,
        d_ff: int = 4096,
        vocab_size: int = 4101,  # 4096 语义 token + 5 特殊 token
        text_enc_dim: int = 512,
        spk_emb_dim: int = 192,
        n_kv_head: int = None,
        dropout: float = 0.0,
        max_seq_len: int = 4096,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.n_layer = n_layer
        self.vocab_size = vocab_size

        # Embedding 层
        # 语音 token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # 文本编码投影
        self.text_proj = nn.Linear(text_enc_dim, d_model)
        # 说话人嵌入投影
        self.spk_proj = nn.Linear(spk_emb_dim, d_model)

        # RoPE
        self.rope = RotaryPositionalEmbedding(d_model // n_head, max_seq_len)

        # Transformer 解码器层
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                n_head=n_head,
                d_ff=d_ff,
                dropout=dropout,
                n_kv_head=n_kv_head,
                layer_idx=i,
            )
            for i in range(n_layer)
        ])

        # 最终层归一化
        self.norm = RMSNorm(d_model)

        # 输出头（预测下一个语音 token）
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 权重共享（可选）
        # self.token_embedding.weight = self.lm_head.weight

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """构建因果注意力 mask"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        text_enc: torch.Tensor,
        spk_emb: torch.Tensor,
        text_len: torch.Tensor,
        audio_len: torch.Tensor,
        prefix_end_pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        LLM 前向传播（训练用）

        Args:
            input_ids: [B, L_audio] 语音 token 序列 (含原始语音 + 目标语音 + 特殊 token)
            text_enc: [B, L_text, D_text] 文本编码序列
            spk_emb: [B, D_spk] 说话人嵌入
            text_len: [B] 文本长度
            audio_len: [B] 语音 token 总长度
            prefix_end_pos: [B] 前缀（条件部分）的结束位置

        Returns:
            logits: [B, L, vocab_size] 预测 logits
            hidden_states: [B, L, D] 最后一层隐藏状态（用于 CFM 条件）
        """
        B = input_ids.shape[0]
        device = input_ids.device

        # 1. 嵌入各部分
        # 说话人嵌入：[B, D] -> [B, 1, D]
        spk_emb_proj = self.spk_proj(spk_emb).unsqueeze(1)

        # 文本编码：[B, L_text, D_text] -> [B, L_text, D]
        text_emb = self.text_proj(text_enc)

        # 语音 token：[B, L_audio] -> [B, L_audio, D]
        audio_emb = self.token_embedding(input_ids)

        # 2. 拼接输入序列：[spk, text, audio]
        # 注意：实际拼接顺序取决于任务类型（OICL/ZICL/TTS）
        # 这里简化为统一接口，具体拼接由 prepare_input 完成
        inputs_embeds = torch.cat([spk_emb_proj, text_emb, audio_emb], dim=1)

        seq_len = inputs_embeds.shape[1]

        # 3. RoPE
        rope_cos, rope_sin = self.rope(inputs_embeds, seq_len=seq_len)

        # 4. 因果注意力 mask
        causal_mask = self._build_causal_mask(seq_len, device)

        # 5. Transformer 解码器层
        hidden_states = inputs_embeds
        past_key_values = None

        for layer in self.layers:
            hidden_states, _ = layer(
                hidden_states, rope_cos, rope_sin,
                mask=causal_mask, past_key_value=None, use_cache=False
            )

        hidden_states = self.norm(hidden_states)

        # 6. 输出 logits
        logits = self.lm_head(hidden_states)

        return logits, hidden_states

    @torch.no_grad()
    def generate(
        self,
        prefix_embeds: torch.Tensor,
        max_new_tokens: int = 1000,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        eos_token_id: int = 4096,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        自回归生成（推理用）

        Args:
            prefix_embeds: [1, L_prefix, D] 前缀嵌入（条件部分）
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_k: top-k 采样
            top_p: top-p (nucleus) 采样
            eos_token_id: EOS token id

        Returns:
            generated_tokens: [1, L_gen] 生成的 token
            hidden_states_list: 每一步的隐藏状态（用于 CFM）
        """
        B = prefix_embeds.shape[0]
        device = prefix_embeds.device
        assert B == 1, "Generation only supports batch_size=1"

        # 初始化
        past_key_values = [None] * self.n_layer
        generated_tokens = []
        hidden_states_list = []

        # 处理前缀（prefill 阶段）
        hidden_states = prefix_embeds
        seq_len = prefix_embeds.shape[1]
        rope_cos, rope_sin = self.rope(hidden_states, seq_len=seq_len)

        for i, layer in enumerate(self.layers):
            hidden_states, past_kv = layer(
                hidden_states, rope_cos, rope_sin,
                mask=None, past_key_value=None, use_cache=True
            )
            past_key_values[i] = past_kv

        hidden_states = self.norm(hidden_states)
        next_token_logits = self.lm_head(hidden_states[:, -1:, :])  # [1, 1, vocab]

        # 生成循环
        for step in range(max_new_tokens):
            # 采样
            logits = next_token_logits[:, -1, :] / temperature
            # Top-K / Top-P
            if top_k > 0:
                top_k_val = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, top_k_val)
                logits[logits < v[:, [-1]]] = float('-inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]

            # 检查 EOS
            if next_token.item() == eos_token_id:
                break

            generated_tokens.append(next_token.squeeze(1))
            hidden_states_list.append(hidden_states[:, -1:, :])  # 记录最后 token 的隐藏状态

            # 下一步嵌入
            next_emb = self.token_embedding(next_token)  # [1, 1, D]
            seq_len += 1

            # 计算新位置的 RoPE
            rope_cos_step, rope_sin_step = self.rope(next_emb, seq_len=1, offset=seq_len - 1)

            # 通过每一层
            hidden_states = next_emb
            new_past_key_values = []
            for i, layer in enumerate(self.layers):
                hidden_states, past_kv = layer(
                    hidden_states, rope_cos_step, rope_sin_step,
                    mask=None, past_key_value=past_key_values[i], use_cache=True
                )
                new_past_key_values.append(past_kv)
            past_key_values = new_past_key_values

            hidden_states = self.norm(hidden_states)
            next_token_logits = self.lm_head(hidden_states)

        if len(generated_tokens) == 0:
            return torch.zeros(1, 0, dtype=torch.long, device=device), []

        generated_tokens = torch.cat(generated_tokens, dim=0).unsqueeze(0)  # [1, L]
        return generated_tokens, hidden_states_list


# ============================================================
# 第三部分：OICL / ZICL 训练范式
# ============================================================

class CosyEditTrainingWrapper(nn.Module):
    """
    CosyEdit 训练包装器 - 实现 OICL 和 ZICL 两种训练范式

    OICL (One-shot In-Context Learning):
      序列格式：[SOS, spk_emb, X_ori, X_tar, sep, μ_ori, trans_sep, μ_tar, EOS]
      提供原始文本+语音对作为显式对齐上下文

    ZICL (Zero-shot In-Context Learning):
      序列格式：[SOS, spk_emb, X_tar, μ_ori, trans_sep, μ_tar, EOS]
      只提供目标文本和原始语音，迫使模型从差异推断编辑边界

    混合训练：L = λ * L_ZICL + (1-λ) * L_OICL

    Args:
        llm: CosyVoiceLLM 模型
        zicl_ratio: ZICL 样本比例 λ (默认 0.4)
        eos_token_id: EOS token id
        ignore_index: 忽略的 token id（用于 loss 计算）
    """

    def __init__(
        self,
        llm: CosyVoiceLLM,
        zicl_ratio: float = 0.4,
        eos_token_id: int = 4096,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.llm = llm
        self.zicl_ratio = zicl_ratio
        self.eos_token_id = eos_token_id
        self.ignore_index = ignore_index

    def _build_oicl_sequence(
        self,
        text_enc_ori: torch.Tensor,
        text_enc_tar: torch.Tensor,
        spk_emb: torch.Tensor,
        audio_tokens_ori: torch.Tensor,
        audio_tokens_tar: torch.Tensor,
        text_len_ori: torch.Tensor,
        text_len_tar: torch.Tensor,
        audio_len_ori: torch.Tensor,
        audio_len_tar: torch.Tensor,
        sep_token_id: int,
        trans_sep_token_id: int,
        sos_token_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        构建 OICL 训练序列

        S_OICL = [S, v, X_ori, X_tar, sep, μ_ori, trans_sep, μ_tar, E]

        Args:
            text_enc_ori: [B, L_ori, D] 原始文本编码
            text_enc_tar: [B, L_tar, D] 目标文本编码
            spk_emb: [B, D_spk] 说话人嵌入
            audio_tokens_ori: [B, T_ori] 原始语音 token
            audio_tokens_tar: [B, T_tar] 目标语音 token
            ... 长度信息和特殊 token

        Returns:
            input_embeds: [B, L_total, D] 输入嵌入
            labels: [B, L_total] 目标 label（-100 表示忽略）
            prefix_end_pos: [B] 前缀结束位置（条件部分的末尾）
        """
        B = text_enc_ori.shape[0]
        device = text_enc_ori.device

        # 嵌入各部分
        spk_emb_proj = self.llm.spk_proj(spk_emb).unsqueeze(1)  # [B, 1, D]
        text_emb_ori = self.llm.text_proj(text_enc_ori)  # [B, L_ori, D]
        text_emb_tar = self.llm.text_proj(text_enc_tar)  # [B, L_tar, D]

        # sep token embedding
        sep_emb = self.llm.token_embedding(
            torch.full((B, 1), sep_token_id, dtype=torch.long, device=device)
        )
        # trans_sep token embedding
        trans_sep_emb = self.llm.token_embedding(
            torch.full((B, 1), trans_sep_token_id, dtype=torch.long, device=device)
        )
        # sos token embedding
        sos_emb = self.llm.token_embedding(
            torch.full((B, 1), sos_token_id, dtype=torch.long, device=device)
        )

        # 语音 token 嵌入
        audio_emb_ori = self.llm.token_embedding(audio_tokens_ori)  # [B, T_ori, D]
        audio_emb_tar = self.llm.token_embedding(audio_tokens_tar)  # [B, T_tar, D]

        # EOS token (用于目标语音末尾)
        eos_emb = self.llm.token_embedding(
            torch.full((B, 1), self.eos_token_id, dtype=torch.long, device=device)
        )

        # 拼接完整输入：[sos, spk, X_ori, X_tar, sep, μ_ori, trans_sep, μ_tar, eos]
        input_embeds = torch.cat([
            sos_emb,          # [B, 1, D]
            spk_emb_proj,     # [B, 1, D]
            text_emb_ori,     # [B, L_ori, D]
            text_emb_tar,     # [B, L_tar, D]
            sep_emb,          # [B, 1, D]
            audio_emb_ori,    # [B, T_ori, D]
            trans_sep_emb,    # [B, 1, D]
            audio_emb_tar,    # [B, T_tar, D]
            eos_emb,          # [B, 1, D]
        ], dim=1)

        # 构建 labels：只对目标语音部分和 EOS 计算 loss
        # 前缀部分：sos + spk + X_ori + X_tar + sep + μ_ori + trans_sep = 忽略
        # 目标部分：μ_tar + eos = 计算 loss
        prefix_len = 2 + text_len_ori + text_len_tar + 1 + audio_len_ori + 1  # sos + spk + ...
        total_len = prefix_len + audio_len_tar + 1  # + eos

        labels = torch.full(
            (B, input_embeds.shape[1]), self.ignore_index, dtype=torch.long, device=device
        )
        # 目标语音 token 部分
        # 注意：需要处理变长的情况（padding）
        for b in range(B):
            p_start = prefix_len[b].item()
            p_end = p_start + audio_len_tar[b].item()
            labels[b, p_start:p_end] = audio_tokens_tar[b, :audio_len_tar[b]]
            labels[b, p_end] = self.eos_token_id

        return input_embeds, labels, prefix_len

    def _build_zicl_sequence(
        self,
        text_enc_tar: torch.Tensor,
        spk_emb: torch.Tensor,
        audio_tokens_ori: torch.Tensor,
        audio_tokens_tar: torch.Tensor,
        text_len_tar: torch.Tensor,
        audio_len_ori: torch.Tensor,
        audio_len_tar: torch.Tensor,
        trans_sep_token_id: int,
        sos_token_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        构建 ZICL 训练序列

        S_ZICL = [S, v, X_tar, μ_ori, trans_sep, μ_tar, E]

        注意：μ_ori 放在 trans_sep 之前，确保原始语音被视为条件而非生成前缀
        """
        B = text_enc_tar.shape[0]
        device = text_enc_tar.device

        spk_emb_proj = self.llm.spk_proj(spk_emb).unsqueeze(1)
        text_emb_tar = self.llm.text_proj(text_enc_tar)
        audio_emb_ori = self.llm.token_embedding(audio_tokens_ori)
        audio_emb_tar = self.llm.token_embedding(audio_tokens_tar)
        trans_sep_emb = self.llm.token_embedding(
            torch.full((B, 1), trans_sep_token_id, dtype=torch.long, device=device)
        )
        sos_emb = self.llm.token_embedding(
            torch.full((B, 1), sos_token_id, dtype=torch.long, device=device)
        )
        eos_emb = self.llm.token_embedding(
            torch.full((B, 1), self.eos_token_id, dtype=torch.long, device=device)
        )

        # 拼接：[sos, spk, X_tar, μ_ori, trans_sep, μ_tar, eos]
        input_embeds = torch.cat([
            sos_emb,
            spk_emb_proj,
            text_emb_tar,
            audio_emb_ori,
            trans_sep_emb,
            audio_emb_tar,
            eos_emb,
        ], dim=1)

        # Labels
        prefix_len = 2 + text_len_tar + audio_len_ori + 1  # sos + spk + X_tar + μ_ori + trans_sep

        labels = torch.full(
            (B, input_embeds.shape[1]), self.ignore_index, dtype=torch.long, device=device
        )
        for b in range(B):
            p_start = prefix_len[b].item()
            p_end = p_start + audio_len_tar[b].item()
            labels[b, p_start:p_end] = audio_tokens_tar[b, :audio_len_tar[b]]
            labels[b, p_end] = self.eos_token_id

        return input_embeds, labels, prefix_len

    def forward(
        self,
        batch: dict,
    ) -> dict:
        """
        前向传播（训练）

        Args:
            batch: 包含以下字段的字典：
                - text_enc_ori: 原始文本编码
                - text_enc_tar: 目标文本编码
                - text_len_ori: 原始文本长度
                - text_len_tar: 目标文本长度
                - spk_emb: 说话人嵌入
                - audio_tokens_ori: 原始语音 token
                - audio_tokens_tar: 目标语音 token
                - audio_len_ori: 原始语音长度
                - audio_len_tar: 目标语音长度
                - sep_token_id: 分隔 token id
                - trans_sep_token_id: 过渡分隔 token id
                - sos_token_id: SOS token id

        Returns:
            dict: 包含 loss 和其他指标
        """
        B = batch['text_enc_ori'].shape[0]
        device = batch['text_enc_ori'].device

        # 随机决定每个样本使用 OICL 还是 ZICL
        use_zicl = torch.rand(B, device=device) < self.zicl_ratio

        # 分别构建 OICL 和 ZICL 序列
        # （实际实现中可以混合 batch，这里简化处理）
        oicl_input, oicl_labels, oicl_prefix = self._build_oicl_sequence(
            batch['text_enc_ori'], batch['text_enc_tar'], batch['spk_emb'],
            batch['audio_tokens_ori'], batch['audio_tokens_tar'],
            batch['text_len_ori'], batch['text_len_tar'],
            batch['audio_len_ori'], batch['audio_len_tar'],
            batch['sep_token_id'], batch['trans_sep_token_id'], batch['sos_token_id'],
        )

        zicl_input, zicl_labels, zicl_prefix = self._build_zicl_sequence(
            batch['text_enc_tar'], batch['spk_emb'],
            batch['audio_tokens_ori'], batch['audio_tokens_tar'],
            batch['text_len_tar'], batch['audio_len_ori'], batch['audio_len_tar'],
            batch['trans_sep_token_id'], batch['sos_token_id'],
        )

        # 按比例混合（简化实现：对整个 batch 选择一种模式）
        # 实际训练中可以 sample-level 混合
        if torch.rand(1).item() < self.zicl_ratio:
            input_embeds = zicl_input
            labels = zicl_labels
        else:
            input_embeds = oicl_input
            labels = oicl_labels

        # LLM 前向
        seq_len = input_embeds.shape[1]
        rope_cos, rope_sin = self.llm.rope(input_embeds, seq_len=seq_len)

        causal_mask = self.llm._build_causal_mask(seq_len, device)

        hidden_states = input_embeds
        for layer in self.llm.layers:
            hidden_states, _ = layer(
                hidden_states, rope_cos, rope_sin,
                mask=causal_mask, past_key_value=None, use_cache=False
            )
        hidden_states = self.llm.norm(hidden_states)

        logits = self.llm.lm_head(hidden_states)

        # 计算 loss
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.ignore_index,
        )

        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': hidden_states,
        }


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CosyEdit LLM 主干 + OICL/ZICL 训练范式")
    print("=" * 70)

    print("""
LLM 架构：
  Transformer Decoder (因果自注意力)
  - 层数: 12
  - 维度: 1024
  - 头数: 16
  - FFN: 4096 (SwiGLU)
  - 归一化: RMSNorm (Pre-LN)
  - 位置编码: RoPE
  - 参数量: ~300M

输入序列组成：
  ┌─────────────────────────────────────────────────────────────┐
  │ SOS │ spk_emb │ 文本编码 │ 语音 token │ trans_sep │ 目标语音 │ EOS │
  └──────────────────┬───────────────────┴───────────┬───────────┘
                     │                               │
                条件部分 (prefix)              生成部分 (target)

OICL vs ZICL：

  OICL (One-shot In-Context Learning):
    [S, v, X_ori, X_tar, sep, μ_ori, Ω, μ_tar, E]
    ↑ 提供原始文本+语音作为对齐上下文
    优点：对齐更准确，未编辑区域保真度高
    缺点：可能复制原始语音而非执行编辑

  ZICL (Zero-shot In-Context Learning):
    [S, v, X_tar, μ_ori, Ω, μ_tar, E]
    ↑ 只提供目标文本和原始语音
    优点：更强的编辑能力，更好的指令遵循
    缺点：对齐精度降低

混合训练：
  L = λ * L_ZICL + (1-λ) * L_OICL
  λ = 0.4 (论文最优值)
  平衡编辑准确性与未编辑区域保真度

在 CosyEdit 中的改动：
  - 完全继承 CosyVoice 的 LLM 架构
  - 通过后训练 (SFT) 适配语音编辑任务
  - 引入 OICL + ZICL 互补训练范式
  - 推理时使用 OICL 范式（效果更好）
    """)
