# Copyright (c) Microsoft Corporation.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
MingTok-Audio (VAE) 核心代码提取

MingTok-Audio 是 Ming-UniAudio 的音频 VAE tokenizer，具有以下特点：
- 连续潜在表示（而非离散 token），生成质量优于离散 token
- 编码器-语义模块-解码器 三段式架构
- 三阶段训练：声学重建 → 语义特征蒸馏 → 与 LLM 联合训练
- 语义模块移植自 Whisper large-v3 编码器，迫使潜空间迎合 LLM 的语义空间

核心设计：
1. Encoder: Qwen2 Transformer 主干 + 线性投影 → 声学潜变量 z (64维)
2. Semantic Module: Whisper 编码器（蒸馏）→ 统一语义特征 z_uni (高维)
3. Decoder: Qwen2 Transformer 主干 + ISTFTHead → 波形重建

参考文献:
- Ming-UniAudio: https://github.com/MingAudio/Ming-UniAudio
- Whisper: https://github.com/openai/whisper
"""

import math
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel, PretrainedConfig
from diffusers.models.autoencoders.autoencoder_oobleck import OobleckDiagonalGaussianDistribution


# ============================================================
# 第一部分：VAE 配置
# ============================================================

class AudioVAEconfig(PretrainedConfig):
    """AudioVAE 配置类

    包含编码器、解码器、语义模块、判别器等所有子模块的配置参数，
    以及各损失函数的权重系数。
    """
    def __init__(
        self,
        enc_kwargs: dict = None,
        semantic_module_kwargs: dict = None,
        dec_kwargs: dict = None,
        hifi_gan_disc_kwargs: dict = None,
        spec_disc_kwargs: dict = None,
        lambda_disc=1.0,
        lambda_mel_loss=15,
        lambda_adv=1.0,
        lambda_feat_match_loss=1.0,
        lambda_semantic=5.0,
        init_method='normal',
        patch_size=-1,
        **kwargs
    ):
        self.enc_kwargs = enc_kwargs
        self.semantic_module_kwargs = semantic_module_kwargs
        self.dec_kwargs = dec_kwargs
        self.hifi_gan_disc_kwargs = hifi_gan_disc_kwargs
        self.spec_disc_kwargs = spec_disc_kwargs
        self.lambda_disc = lambda_disc
        self.lambda_mel_loss = lambda_mel_loss
        self.lambda_adv = lambda_adv
        self.lambda_feat_match_loss = lambda_feat_match_loss
        self.lambda_semantic = lambda_semantic
        self.init_method = init_method
        self.patch_size = patch_size
        super().__init__(**kwargs)


# ============================================================
# 第二部分：Whisper 语义编码器
# ============================================================

class LayerNorm(nn.LayerNorm):
    """自定义 LayerNorm，处理精度转换"""
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    """自定义 Linear，处理精度转换"""
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class MultiHeadAttention(nn.Module):
    """Whisper 风格的多头注意力，使用 Flash Attention 和 RoPE"""

    def __init__(self, n_state: int, n_head: int, layer_idx: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)
        self.layer_idx = layer_idx

        # RoPE (Rotary Positional Embeddings)
        from torchtune.modules import RotaryPositionalEmbeddings
        self.rotary_embed = RotaryPositionalEmbeddings(dim=n_state // n_head)

    def forward(self, x: Tensor, past_key_values=None):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        wv, qk, past_key_values = self.qkv_attention(q, k, v, past_key_values=past_key_values)
        return self.out(wv), qk, past_key_values

    def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, past_key_values=None):
        q = q.view(*q.shape[:2], self.n_head, -1)  # [B, T, nhead, dm]
        k = k.view(*k.shape[:2], self.n_head, -1)
        v = v.view(*v.shape[:2], self.n_head, -1)

        if past_key_values is not None:
            from transformers.cache_utils import DynamicCache
            past_seen_tokens = past_key_values.get_seq_length(self.layer_idx) if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + q.size(1), device=q.device
            )
            cache_position = cache_position.unsqueeze(0)
        else:
            cache_position = None

        q = self.rotary_embed(q, input_pos=cache_position)
        k = self.rotary_embed(k, input_pos=cache_position)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx, {"cache_position": cache_position})

        # Flash Attention
        from flash_attn import flash_attn_func
        a = flash_attn_func(
            q.permute(0, 2, 1, 3),
            k.permute(0, 2, 1, 3),
            v.permute(0, 2, 1, 3),
            causal=True
        )
        out = a.flatten(start_dim=2)
        qk = None

        return out, qk, past_key_values


class ResidualAttentionBlock(nn.Module):
    """Whisper 残差注意力块"""

    def __init__(self, n_state: int, n_head: int, layer_idx: int):
        super().__init__()
        self.attn = MultiHeadAttention(n_state, n_head, layer_idx)
        self.attn_ln = LayerNorm(n_state)
        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)
        self.layer_idx = layer_idx

    def forward(self, x: Tensor, past_key_values=None):
        attn_out, _, past_key_values = self.attn(self.attn_ln(x), past_key_values=past_key_values)
        x = x + attn_out
        x = x + self.mlp(self.mlp_ln(x))
        return x, past_key_values


class WhisperAudioEncoder(nn.Module):
    """
    Whisper 音频编码器（语义模块）

    移植自 Whisper large-v3 编码器，但移除了原始的卷积输入层，
    直接接收声学潜变量的投影作为输入。

    作为语义蒸馏的学生模型，从原始 Whisper large-v3 蒸馏语义知识。
    在第三阶段训练中，通过 LLM 的文本预测损失来对齐语义空间。

    Args:
        n_state: 隐藏层维度（Whisper large-v3 为 1280）
        n_head: 注意力头数
        n_layer: 层数
    """

    def __init__(self, n_state: int, n_head: int, n_layer: int):
        super().__init__()

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, layer_idx=i) for i in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)
        self.audio_emb_dim = n_state

    def forward(self, whipser_feats: Tensor, use_cache=False, past_key_values=None, **kwargs):
        """
        Args:
            whipser_feats: [B, T, D] 输入特征（声学潜变量投影后的特征）
            use_cache: 是否使用 KV cache
            past_key_values: 过去的 KV cache

        Returns:
            x: [B, T, D] 输出的统一语义特征
            past_key_values: 更新后的 KV cache
        """
        if past_key_values is None and use_cache:
            from transformers.cache_utils import DynamicCache
            past_key_values = DynamicCache()

        x = whipser_feats

        for block in self.blocks:
            x, past_key_values = block(x, past_key_values=past_key_values)

        x = self.ln_post(x)

        return x, past_key_values

    @classmethod
    def from_pretrained(cls, dims):
        """从预训练配置创建模型（实际加载权重在外部完成）"""
        audio_encoder = cls(
            dims['n_state'],
            dims['n_head'],
            dims['n_layer'],
        )
        audio_encoder.audio_emb_dim = dims['n_state']
        return audio_encoder


# ============================================================
# 第三部分：ISTFT 解码头
# ============================================================

class ISTFT(nn.Module):
    """
    自定义 ISTFT 实现，支持 same padding 和流式推理

    Args:
        n_fft: FFT 大小
        hop_length: 帧移
        win_length: 窗长
        padding: 填充类型 "center" 或 "same"
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int, padding: str = "same"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

        self.buffer_len = self.win_length - self.hop_length

    def _buffer_process(self, x, buffer, pad, last_chunk=False, streaming=False):
        """流式推理时的缓冲区处理"""
        if streaming:
            if buffer is None:
                x = x[:, pad:]
            if buffer is not None:
                x[:, :self.buffer_len] += buffer
            buffer = x[:, -self.buffer_len:]
            if not last_chunk:
                x = x[:, :-self.buffer_len]
            else:
                x = x[:, :-pad]
        else:
            x = x[:, pad:-pad]
        return x, buffer

    def forward(self, spec: torch.Tensor, audio_buffer=None, window_buffer=None,
                streaming=False, last_chunk=False):
        """
        Args:
            spec: [B, N, T] 复数频谱
            audio_buffer: 流式音频缓冲区
            window_buffer: 流式窗函数缓冲区
            streaming: 是否流式模式
            last_chunk: 是否为最后一块

        Returns:
            y: [B, L] 重建的时域信号
        """
        if self.padding == "center":
            return torch.istft(spec, self.n_fft, self.hop_length, self.win_length, self.window, center=True)
        elif self.padding == "same":
            pad = (self.win_length - self.hop_length) // 2
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        assert spec.dim() == 3
        B, N, T = spec.shape

        # Inverse FFT
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]

        # Overlap and Add
        output_size = (T - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft, output_size=(1, output_size), kernel_size=(1, self.win_length), stride=(1, self.hop_length),
        )[:, 0, 0, :]

        y, audio_buffer = self._buffer_process(y, audio_buffer, pad, last_chunk=last_chunk, streaming=streaming)

        # Window envelope normalization
        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = torch.nn.functional.fold(
            window_sq, output_size=(1, output_size), kernel_size=(1, self.win_length), stride=(1, self.hop_length),
        ).squeeze(0).squeeze(0)

        window_envelope, window_buffer = self._buffer_process(
            window_envelope, window_buffer, pad, last_chunk=last_chunk, streaming=streaming
        )
        window_envelope = window_envelope.squeeze()

        assert (window_envelope > 1e-11).all()
        y = y / window_envelope

        return y, audio_buffer, window_buffer


class ISTFTHead(nn.Module):
    """
    ISTFT 解码头 - 将 Transformer 输出映射为 STFT 系数并反变换为波形

    预测幅度和相位，通过 ISTFT 重建时域音频。
    支持流式推理（用于自回归生成场景）。

    Args:
        dim: 输入隐藏维度
        n_fft: FFT 大小
        hop_length: 帧移
        padding: 填充类型
    """

    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "same"):
        super().__init__()
        out_dim = n_fft + 2  # mag + phase (实部+虚部)
        self.out = torch.nn.Linear(dim, out_dim)
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding)

    def forward(self, x: torch.Tensor, audio_buffer=None, window_buffer=None,
                streaming=False, last_chunk=False):
        """
        Args:
            x: [B, L, H] Transformer 输出

        Returns:
            audio: [B, 1, T] 重建的时域音频
            x_pred: [B, F, T] 预测的频谱
        """
        x_pred = self.out(x)
        x_pred = x_pred.transpose(1, 2)
        mag, p = x_pred.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2)  # 防止幅度过大
        # 构造复数频谱
        x_real = torch.cos(p)
        y_imag = torch.sin(p)
        S = mag * (x_real + 1j * y_imag)
        audio, audio_buffer, window_buffer = self.istft(
            S, audio_buffer=audio_buffer, window_buffer=window_buffer,
            streaming=streaming, last_chunk=last_chunk
        )
        return audio.unsqueeze(1), x_pred, audio_buffer, window_buffer


# ============================================================
# 第四部分：VAE 编码器和解码器
# ============================================================

class Encoder(nn.Module):
    """
    MingTok-Audio 编码器

    将波形音频编码为声学潜变量。
    使用 Qwen2 Transformer 作为主干，先将波形分帧，
    通过线性投影后送入 Transformer，最后输出高斯分布的均值和方差。

    架构：
    波形 → 分帧 → FC1 → FC2 → Qwen2 Transformer → FC3 → μ, σ (VAE 后验)

    Args:
        encoder_args: Qwen2 编码器配置字典
        input_dim: 输入帧大小（默认 320）
        hop_size: 帧移（默认 320，即无重叠）
        latent_dim: 潜变量维度（默认 64，输出为 2*latent_dim 含 μ 和 logvar）
    """

    def __init__(self, encoder_args, input_dim=320, hop_size=320, latent_dim=64):
        super().__init__()
        from transformers import Qwen2Model, Qwen2Config
        config = Qwen2Config.from_dict(config_dict=encoder_args)
        self.encoder = Qwen2Model(config)
        self.input_dim = input_dim
        self.hop_size = hop_size
        self.latent_dim = latent_dim

        # 输入投影层
        self.fc1 = nn.Linear(input_dim, config.hidden_size, bias=False)
        self.fc2 = nn.Linear(config.hidden_size, config.hidden_size)
        # 输出层：输出 2*latent_dim（均值 + log 方差）
        self.fc3 = nn.Linear(config.hidden_size, latent_dim * 2)
        self.norm = nn.LayerNorm(config.hidden_size)

    def get_frames(self, x):
        """将波形切分为帧
        
        Args:
            x: [B, T] 波形
            
        Returns:
            frames: [B, num_frames, input_dim] 帧序列
        """
        num_frames_total = (x.size(-1) + self.hop_size - 1) // self.hop_size
        expected_len = (num_frames_total - 1) * self.hop_size + self.input_dim
        padding_needed = expected_len - x.size(-1)
        waveform = F.pad(x, (0, padding_needed), value=0.0)
        frames = waveform.unfold(dimension=-1, size=self.input_dim, step=self.hop_size)  # [B, T, d]
        return frames

    def forward(self, waveform):
        """
        Args:
            waveform: [B, T] 输入波形

        Returns:
            x: [B, T_frame, 2*latent_dim] 编码后的高斯分布参数
            waveform_padded: [B, 1, T_padded] padding 后的波形
        """
        x = self.get_frames(waveform)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.encoder(inputs_embeds=x)
        x = x.last_hidden_state
        x = self.fc3(x)  # [B, T, 2*latent_dim]
        return x, waveform.unsqueeze(1)


class Decoder(nn.Module):
    """
    MingTok-Audio 解码器

    将声学潜变量解码回波形音频，包含语义模块。
    使用 Qwen2 Transformer 作为主干，最后通过 ISTFTHead 输出波形。

    架构：
    潜变量 z → FC1 → FC2 → Whisper Semantic Module → FC3 → Qwen2 Transformer → ISTFTHead → 波形
                           ↓
                     unified_emb (统一语义特征，送给 LLM)

    Args:
        decoder_args: Qwen2 解码器配置字典
        output_dim: 输出帧大小
        latent_dim: 潜变量维度
        semantic_model: Whisper 语义编码器（可选）
        patch_size: patch 大小
    """

    def __init__(self, decoder_args, output_dim=320, latent_dim=64, semantic_model=None, patch_size=-1):
        super().__init__()
        from transformers import Qwen2Model, Qwen2Config
        config = Qwen2Config.from_dict(config_dict=decoder_args)
        self.decoder = Qwen2Model(config)
        self.output_dim = output_dim
        self.latent_dim = latent_dim

        # 输入投影
        self.fc1 = nn.Linear(latent_dim, config.hidden_size)

        # 语义模块通路
        if semantic_model is not None:
            self.gelu = nn.GELU()
            self.fc2 = nn.Linear(config.hidden_size, semantic_model.audio_emb_dim)
            self.semantic_model = semantic_model
            self.fc3 = nn.Linear(semantic_model.audio_emb_dim, config.hidden_size)
        else:
            self.semantic_model = None

        self.hop_length = output_dim
        # ISTFT 输出头
        self.head = ISTFTHead(
            dim=config.hidden_size,
            n_fft=self.hop_length * 4,
            hop_length=self.hop_length,
            padding="same"
        )
        self.patch_size = patch_size

    def forward(self, x, only_semantic_emb=False, past_key_values=None, use_cache=False):
        """
        完整前向传播（训练用）

        Args:
            x: [B, T_frame, D_latent] 声学潜变量
            only_semantic_emb: 是否只返回语义特征（不经过声学解码）
            past_key_values: KV cache
            use_cache: 是否使用 cache

        Returns:
            waveform: [B, 1, T_wav] 重建的波形
            unified_emb: [B, T_frame, D_unified] 统一语义特征
        """
        x = self.fc1(x)

        if self.semantic_model is not None:
            # 投影到语义模块输入维度
            x = self.fc2(self.gelu(x))
            # 通过 Whisper 语义编码器
            x, past_key_values = self.semantic_model(
                whipser_feats=x, past_key_values=past_key_values, use_cache=use_cache
            )
            unified_emb = x
            if only_semantic_emb:
                return unified_emb, past_key_values
            # 投影回 Transformer 维度
            x = self.fc3(x)
        else:
            unified_emb = None

        # Qwen2 Transformer 解码
        x = self.decoder(inputs_embeds=x)
        x = x.last_hidden_state

        # ISTFT 输出波形
        x, _ = self.head(x)

        return x, unified_emb

    def low_level_reconstruct(self, x, past_key_values=None, use_cache=False,
                              audio_buffer=None, window_buffer=None, last_chunk=False):
        """
        低层声学重建（流式推理用，跳过语义模块）

        仅使用 Qwen2 Transformer + ISTFTHead 进行解码，
        支持 KV cache 和流式 ISTFT。

        Args:
            x: [B, T_frame, D_latent] 声学潜变量
            past_key_values: KV cache
            use_cache: 是否使用 cache
            audio_buffer: 音频流式缓冲区
            window_buffer: 窗函数流式缓冲区
            last_chunk: 是否为最后一块

        Returns:
            waveform: [B, 1, T_wav] 重建波形
            audio_buffer: 更新后的音频缓冲区
            window_buffer: 更新后的窗函数缓冲区
            past_key_values: 更新后的 KV cache
        """
        x = self.fc1(x)
        outputs = self.decoder(inputs_embeds=x, past_key_values=past_key_values, use_cache=use_cache)
        past_key_values = outputs.past_key_values
        x = outputs.last_hidden_state

        x, _, audio_buffer, window_buffer = self.head(
            x, streaming=use_cache, audio_buffer=audio_buffer,
            window_buffer=window_buffer, last_chunk=last_chunk
        )

        return x, audio_buffer, window_buffer, past_key_values


# ============================================================
# 第五部分：AudioVAE 主类
# ============================================================

class AudioVAE(PreTrainedModel):
    """
    MingTok-Audio VAE 主模型

    完整的音频 VAE tokenizer，包含：
    - 编码器（Encoder）：波形 → 声学潜变量
    - 语义模块（Semantic Module）：声学潜变量 → 统一语义特征
    - 解码器（Decoder）：潜变量 → 波形

    三阶段训练：
    1. 声学重建训练：只训练编码器 + 解码器，VAE-GAN 损失
    2. 语义特征蒸馏：只训练语义模块，从 Whisper large-v3 蒸馏
    3. 与 LLM 联合训练：冻结 LLM，用文本预测损失更新语义模块

    关键接口：
    - encode_latent: 波形 → 声学潜变量 z
    - encode_unified_emb_from_latent: 声学潜变量 → 统一语义特征 z_uni
    - encode_unified_emb_from_waveform: 波形 → 统一语义特征（端到端）
    - decode: 声学潜变量 → 波形
    """

    config_class = AudioVAEconfig

    def __init__(self, config: AudioVAEconfig):
        super().__init__(config)

        # 编码器
        self.encoder = Encoder(
            encoder_args=config.enc_kwargs['backbone'],
            input_dim=config.enc_kwargs['input_dim'],
            hop_size=config.enc_kwargs.get('hop_size', 320),
            latent_dim=config.enc_kwargs['latent_dim'],
        )

        # 语义模块（可选）
        if config.semantic_module_kwargs is not None:
            semantic_model = WhisperAudioEncoder.from_pretrained(
                dims=config.semantic_module_kwargs['whisper_encoder']
            )
            self.semantic_emb_dim = config.semantic_module_kwargs['whisper_encoder']['n_state']
        else:
            semantic_model = None

        # 解码器
        self.decoder = Decoder(
            decoder_args=config.dec_kwargs['backbone'],
            output_dim=config.dec_kwargs['output_dim'],
            latent_dim=config.dec_kwargs['latent_dim'],
            semantic_model=semantic_model,
        )

        self.post_init()

    def _init_weights(self, module):
        """权重初始化"""
        std = 0.02
        if isinstance(module, nn.Linear):
            if self.config.init_method == 'kaiming':
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            else:
                module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @torch.inference_mode()
    def encode_latent(self, waveform, waveform_length):
        """
        编码波形为声学潜变量

        Args:
            waveform: [B, T_wav] 输入波形
            waveform_length: [B] 每个波形的实际长度

        Returns:
            z: [B, T_frame, D_latent] 采样后的声学潜变量
            frame_num: [B] 每段音频的帧数
        """
        frame_num = torch.ceil(
            waveform_length / self.config.enc_kwargs['input_dim']
        ).to(torch.int32)
        h, y = self.encoder(waveform)
        h = h.transpose(1, 2)  # [B, d, T]

        # 对角高斯分布 + 重参数化采样
        posterior = OobleckDiagonalGaussianDistribution(h)
        latent = posterior.sample()  # [B, d/2, T]
        latent = latent.transpose(1, 2)  # [B, T, d/2]
        return latent, frame_num

    @torch.inference_mode()
    def encode_unified_emb_from_latent(self, latent, past_key_values=None, use_cache=False):
        """
        将声学潜变量映射为统一语义特征（通过语义模块）

        Args:
            latent: [B, T_frame, D_latent] 声学潜变量
            past_key_values: KV cache
            use_cache: 是否使用 cache

        Returns:
            unified_emb: [B, T_frame, D_unified] 统一语义特征
            past_key_values: 更新后的 KV cache
        """
        unified_emb, past_key_values = self.decoder(
            latent, only_semantic_emb=True,
            past_key_values=past_key_values, use_cache=use_cache
        )
        return unified_emb, past_key_values

    @torch.inference_mode()
    def encode_unified_emb_from_waveform(self, waveform, waveform_length):
        """
        端到端编码：波形 → 声学潜变量 → 统一语义特征

        Args:
            waveform: [B, T_wav] 输入波形
            waveform_length: [B] 波形长度

        Returns:
            unified_emb: 统一语义特征
            latent: 声学潜变量
            frame_num: 帧数
        """
        latent, frame_num = self.encode_latent(waveform, waveform_length)
        unified_emb, past_key_values = self.encode_unified_emb_from_latent(latent)
        return unified_emb, latent, frame_num

    @torch.inference_mode()
    def decode(self, latent, past_key_values=None, use_cache=False,
               audio_buffer=None, window_buffer=None, last_chunk=False):
        """
        从声学潜变量重建波形

        Args:
            latent: [B, T_frame, D_latent] 声学潜变量

        Returns:
            waveform: [B, 1, T_wav] 重建的波形
        """
        waveform, audio_buffer, window_buffer, past_key_values = self.decoder.low_level_reconstruct(
            latent, past_key_values=past_key_values, use_cache=use_cache,
            audio_buffer=audio_buffer, window_buffer=window_buffer, last_chunk=last_chunk
        )
        return waveform, audio_buffer, window_buffer, past_key_values


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MingTok-Audio (VAE) 架构说明")
    print("=" * 70)

    print("""
架构总览：
  波形 (B, T_wav)
    │
    ▼  Encoder (Qwen2 Transformer)
  声学潜变量 z (B, T_frame, 64)
    │
    ▼  Semantic Module (Whisper Encoder)
  统一语义特征 z_uni (B, T_frame, 1280)  → 送给 LLM
    │
    ▼  Decoder (Qwen2 Transformer + ISTFTHead)
  重建波形 (B, T_wav)

三阶段训练：
  阶段 1 - 声学重建：
    只训练 Encoder + Decoder
    损失：VAE-GAN (reconstruction + adversarial + feature matching)
    目标：高保真波形重建

  阶段 2 - 语义特征蒸馏：
    只训练 Semantic Module
    教师：原始 Whisper large-v3 编码器
    学生：移除卷积层的 Whisper 编码器
    目标：蒸馏语义知识

  阶段 3 - 与 LLM 联合训练：
    冻结 LLM，只更新 Semantic Module
    损失：LLM 预测文本的交叉熵损失 L_align
    目标：迫使 z_uni 迎合 LLM 的语义空间
    """)

    print("=" * 70)
    print("关键参数：")
    print("=" * 70)
    print("  输入维度 (input_dim): 320")
    print("  帧移 (hop_size): 320")
    print("  潜变量维度 (latent_dim): 64")
    print("  语义特征维度: 1280 (Whisper large-v3)")
    print("  编码器主干: Qwen2 Transformer")
    print("  解码器主干: Qwen2 Transformer")
    print("  语义模块: Whisper Encoder (蒸馏)")
    print("  输出头: ISTFTHead (n_fft=1280, hop=320)")
    print("  采样率: 16kHz (hop=320 → 50fps)")
