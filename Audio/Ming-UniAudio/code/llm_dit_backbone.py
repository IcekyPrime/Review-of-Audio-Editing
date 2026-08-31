# Copyright (c) Microsoft Corporation.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Ming-UniAudio LLM + DiT 模型主干代码提取

Ming-UniAudio 是首个指令式、自由形式、无需时间戳的通用语音编辑模型。
模型主体为 LLM + 双任务头设计：

架构总览：
                    ┌─────────────┐
    文本指令 + 音频  │    LLM      │  统一语义特征 (z_uni)
     统一嵌入输入 ──▶│  (Qwen2)    │────▶  隐藏状态
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
        ┌───────────┐            ┌───────────┐
        │  文本头   │            │  扩散头   │
        │ (LM Head) │            │  (DiT)    │
        └─────┬─────┘            └─────┬─────┘
              │                        │
              ▼                        ▼
        CoT (思维链)           生成的音频潜变量
                              (通过 VAE 解码为波形)

关键设计：
1. LLM 作为主干，同时处理文本和音频（统一嵌入空间）
2. 文本头输出 CoT（Chain of Thought），用于推理增强
3. 扩散头（DiT）接收 LLM 隐藏状态作为条件，生成音频潜变量
4. 编辑任务同时使用两个头：CoT 不送给扩散头，但隐藏状态会送给扩散头
5. 采用 Flow Matching (CFM) 而非传统扩散，采样步数少、质量高

参考文献:
- Ming-UniAudio: https://github.com/MingAudio/Ming-UniAudio
- DiT: https://arxiv.org/abs/2212.09748
- Flow Matching: https://arxiv.org/abs/2210.02747
"""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from x_transformers.x_transformers import RotaryEmbedding, apply_rotary_pos_emb


# ============================================================
# 第一部分：DiT 基础模块
# ============================================================

class RMSNorm(nn.Module):
    """RMS 归一化层"""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.native_rms_norm = float(torch.__version__[:3]) >= 2.4

    def forward(self, x):
        if self.native_rms_norm:
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                x = x.to(self.weight.dtype)
            x = F.rms_norm(x, normalized_shape=(x.shape[-1],), weight=self.weight, eps=self.eps)
        else:
            variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                x = x.to(self.weight.dtype)
            x = x * self.weight
        return x


class FeedForward(nn.Module):
    """前馈网络层（GELU 激活）"""

    def __init__(self, dim, dim_out=None, mult=4, dropout=0.0, approximate: str = "none"):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        activation = nn.GELU(approximate=approximate)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), activation)
        self.ff = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.ff(x)


class Attention(nn.Module):
    """
    自注意力层（DiT 使用）

    支持：
    - Rotary Position Embedding (RoPE)
    - QK Norm (RMSNorm)
    - 可选 Flash Attention 后端
    - 非因果注意力（DiT 不需要因果 mask）
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        qk_norm: Optional[str] = None,
        pe_attn_head: int | None = None,
        attn_backend: str = "torch",
        attn_mask_enabled: bool = True,
    ):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Attention requires PyTorch 2.0")

        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.dropout = dropout

        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)

        # QK Norm
        if qk_norm is None:
            self.q_norm = None
            self.k_norm = None
        elif qk_norm == "rms_norm":
            self.q_norm = RMSNorm(dim_head, eps=1e-6)
            self.k_norm = RMSNorm(dim_head, eps=1e-6)
        else:
            raise ValueError(f"Unimplemented qk_norm: {qk_norm}")

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, dim))
        self.to_out.append(nn.Dropout(dropout))

        self.pe_attn_head = pe_attn_head
        self.attn_backend = attn_backend
        self.attn_mask_enabled = attn_mask_enabled

    def forward(self, x: float, mask=None, rope=None) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] 输入特征
            mask: [B, L] padding mask
            rope: RoPE 位置编码

        Returns:
            [B, L, D] 注意力输出
        """
        batch_size = x.shape[0]

        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        # Reshape for multi-head attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)  # [B, H, L, d]
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        # QK Norm
        if self.q_norm is not None:
            query = self.q_norm(query)
        if self.k_norm is not None:
            key = self.k_norm(key)

        # 应用 RoPE
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)

            if self.pe_attn_head is not None:
                pn = self.pe_attn_head
                query[:, :pn, :, :] = apply_rotary_pos_emb(query[:, :pn, :, :], freqs, q_xpos_scale)
                key[:, :pn, :, :] = apply_rotary_pos_emb(key[:, :pn, :, :], freqs, k_xpos_scale)
            else:
                query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
                key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        # Scaled Dot-Product Attention（非因果）
        x = F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        x = x.to(query.dtype)
        x = self.to_out[0](x)
        x = self.to_out[1](x)

        # 应用 mask
        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)

        return x


class DiTBlock(nn.Module):
    """
    DiT (Diffusion Transformer) Block

    标准的 DiT 块结构：
    - AdaLN-Zero 风格的条件调制（通过时间步和条件的 embedding 相加实现）
    - 注意力 + 前馈网络的残差结构
    - RMSNorm 归一化

    Args:
        hidden_size: 隐藏层维度
        num_heads: 注意力头数
        mlp_ratio: MLP 扩展倍率
        dropout: dropout 概率
        qk_norm: QK 归一化类型
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        dropout=0.1,
        qk_norm=None,
        pe_attn_head=None,
        attn_backend="flash_attn",
        attn_mask_enabled=True,
        **kwargs
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            dim=hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            dropout=dropout,
            qk_norm=qk_norm,
            pe_attn_head=pe_attn_head,
            attn_backend=attn_backend,
            attn_mask_enabled=attn_mask_enabled,
        )
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = FeedForward(dim=hidden_size, mult=mlp_ratio, dropout=dropout, approximate="tanh")

    def forward(self, x, mask, rope):
        """
        Args:
            x: [B, L, D] 输入
            mask: [B, L] padding mask
            rope: RoPE 位置编码

        Returns:
            [B, L, D] 输出
        """
        x = x + self.attn(self.norm1(x), mask=mask, rope=rope)
        x = x + self.mlp(self.norm2(x))
        return x


class FinalLayer(nn.Module):
    """
    DiT 最终输出层

    将隐藏状态映射回输出通道数（潜变量维度）。
    """

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        return x


# ============================================================
# 第二部分：DiT 主模型
# ============================================================

class SinusPositionEmbedding(nn.Module):
    """正弦位置编码（用于时间步嵌入）"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimestepEmbedder(nn.Module):
    """
    时间步嵌入器

    将扩散时间步 t 映射为条件嵌入，用于 DiT 的条件调制。
    使用正弦位置编码 + MLP。
    """

    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(freq_embed_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, timestep):
        """
        Args:
            timestep: [B] 时间步

        Returns:
            [B, D] 时间步嵌入
        """
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        time = self.time_mlp(time_hidden)  # [B, D]
        return time


class CondEmbedder(nn.Module):
    """
    LLM 条件嵌入器

    将 LLM 的隐藏状态（条件）投影到 DiT 的隐藏维度。
    支持 Classifier-Free Guidance 的条件 dropout。

    Args:
        input_feature_size: LLM 隐藏状态维度
        hidden_size: DiT 隐藏层维度
        dropout_prob: 条件 dropout 概率（用于 CFG）
    """

    def __init__(self, input_feature_size, hidden_size, dropout_prob):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.cond_embedder = nn.Linear(input_feature_size, hidden_size)

    def cond_drop(self, llm_cond):
        """随机丢弃条件（用于 CFG 训练）"""
        bsz = llm_cond.shape[0]
        drop_latent_mask = torch.rand(bsz) < self.dropout_prob
        drop_latent_mask = drop_latent_mask.unsqueeze(-1).unsqueeze(-1).to(llm_cond.dtype).to(llm_cond.device)
        fake_latent = torch.zeros(llm_cond.shape).to(llm_cond.device)
        llm_cond = drop_latent_mask * fake_latent + (1 - drop_latent_mask) * llm_cond
        return llm_cond

    def forward(self, llm_cond, train):
        """
        Args:
            llm_cond: [B, L_cond, D_llm] LLM 隐藏状态条件
            train: 是否训练模式（决定是否应用 dropout）

        Returns:
            [B, L_cond, D_dit] 投影后的条件嵌入
        """
        use_dropout = self.dropout_prob > 0
        if train and use_dropout:
            llm_cond = self.cond_drop(llm_cond)

        llm_cond = self.cond_embedder(llm_cond)
        return llm_cond


class DiT(nn.Module):
    """
    Diffusion Transformer (DiT) - 扩散头

    Ming-UniAudio 的扩散生成头，以 LLM 隐藏状态为条件，
    生成音频的声学潜变量。

    输入结构（按序列顺序拼接）：
    [时间嵌入 + 条件嵌入, 历史潜变量, 当前待生成潜变量]

    关键设计：
    1. 时间步嵌入 + LLM 条件嵌入 相加作为全局条件
    2. 历史潜变量（已生成部分）作为上下文
    3. 当前带噪潜变量 x_t 作为输入
    4. 使用 RoPE 进行位置编码
    5. 支持 Classifier-Free Guidance (CFG)

    Args:
        in_channels: 输入/输出潜变量通道数（= VAE latent_dim）
        hidden_size: DiT 隐藏层维度
        depth: DiT 层数
        num_heads: 注意力头数
        mlp_ratio: MLP 扩展倍率
        llm_cond_dim: LLM 条件的维度
        cfg_dropout_prob: CFG dropout 概率
    """

    def __init__(
        self,
        in_channels=4,
        hidden_size=1024,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        llm_cond_dim=896,
        cfg_dropout_prob=0.1,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads

        # 时间步嵌入
        self.t_embedder = TimestepEmbedder(hidden_size)
        # 输入潜变量投影
        self.x_embedder = nn.Linear(in_channels, hidden_size)
        # LLM 条件投影
        self.c_embedder = CondEmbedder(llm_cond_dim, hidden_size, cfg_dropout_prob)

        self.hidden_size = hidden_size

        # RoPE 位置编码
        self.rotary_embed = RotaryEmbedding(hidden_size // num_heads)

        # DiT 层
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **kwargs)
            for _ in range(depth)
        ])

        # 最终输出层
        self.final_layer = FinalLayer(hidden_size, self.out_channels)

    def forward(self, x, t, c, latent_history, mask=None):
        """
        DiT 前向传播

        Args:
            x: [B, patch_size, D] 当前带噪潜变量（待预测的流）
            t: [B] 扩散时间步
            c: [B, L_cond, D_llm] LLM 隐藏状态条件
            latent_history: [B, L_history, D] 已生成的历史潜变量
            mask: [B, L_x] 当前 x 的 mask

        Returns:
            [B, L_total, D_out] 预测的流（含条件和历史部分）
        """
        # 时间步嵌入: [B, 1, D]
        t = self.t_embedder(t).unsqueeze(1)
        # 当前潜变量投影
        x_now = self.x_embedder(x)
        # 历史潜变量投影
        x_history = self.x_embedder(latent_history)
        # 拼接历史 + 当前
        x = torch.cat([x_history, x_now], dim=1)
        # 条件投影
        c = self.c_embedder(c, self.training)
        # 条件 = 时间 + LLM 条件（广播相加）
        y = t + c

        # 拼接顺序：[条件, 历史潜变量, 当前潜变量]
        x = torch.cat([y, x], dim=1)

        # RoPE 位置编码
        rope = self.rotary_embed.forward_from_seq_len(x.shape[1])

        # Mask 扩展（条件部分和历史部分也需要 mask）
        if mask is not None:
            mask_pad = mask.clone().detach()[:, :1].expand(-1, x_history.shape[1] + c.shape[1])
            mask = torch.cat([mask_pad, mask], dim=-1)

        # DiT 层前向
        for block in self.blocks:
            x = block(x, mask, rope)

        # 最终输出
        x = self.final_layer(x)
        return x

    def forward_with_cfg(self, x, t, c, cfg_scale, latent_history, patch_size):
        """
        Classifier-Free Guidance 推理

        同时进行条件和无条件前向，然后加权组合。

        Args:
            x: [B, patch_size, D] 当前带噪潜变量
            t: [] 或 [B] 时间步
            c: [B, L_cond, D_llm] LLM 条件
            cfg_scale: CFG 强度
            latent_history: [B, L_history, D] 历史潜变量
            patch_size: 当前 patch 的大小

        Returns:
            [B, patch_size, D_out] 引导后的预测
        """
        if not cfg_scale == 1:
            # 拼接条件和无条件 batch
            x = torch.cat([x, x], dim=0)
            latent_history = torch.cat([latent_history, latent_history], dim=0)
            fake_latent = torch.zeros(c.shape).to(c.device)
            c = torch.cat([c, fake_latent], dim=0)

        if t.ndim == 0:
            t = t.repeat(x.shape[0])

        model_out = self.forward(x, t, c, latent_history)
        # 只取当前 patch 部分的输出
        return model_out[:, -patch_size:, :]


# ============================================================
# 第三部分：Conditional Flow Matching (CFM)
# ============================================================

class Solver:
    """
    ODE 求解器 - 用于流匹配推理

    使用欧拉法求解流匹配 ODE，支持带噪声的随机采样。

    Args:
        func: 速度场函数 v(t, x)
        y0: 初始状态（噪声）
        sigma: 噪声强度
        temperature: 温度参数
    """

    def __init__(self, func, y0, sigma=0.25, temperature=1.5) -> None:
        self.func = func
        self.y0 = y0
        self.sigma = sigma
        self.temperature = temperature

    def integrate(self, t):
        """
        积分求解 ODE

        Args:
            t: 时间步序列

        Returns:
            solution: 每一步的解
        """
        solution = torch.empty(len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
        solution[0] = self.y0

        j = 1
        y0 = self.y0
        for t0, t1 in zip(t[:-1], t[1:]):
            dt = t1 - t0
            f0 = self.func(t0, y0)
            dy = dt * f0
            y1 = y0 + dy

            # 线性插值记录中间点
            while j < len(t) and t1 >= t[j]:
                solution[j] = self._linear_interp(t0, t1, y0, y1, t[j])
                j += 1

            # 添加随机噪声（随机采样）
            noise = torch.randn_like(y0)
            shift = self.sigma * (self.temperature ** 0.5) * (abs(dt) ** 0.5) * noise
            y0 = y1 + shift

        return solution

    def _linear_interp(self, t0, t1, y0, y1, t):
        """线性插值"""
        if t == t0:
            return y0
        if t == t1:
            return y1
        slope = (t - t0) / (t1 - t0)
        return y0 + slope * (y1 - y0)


def get_epss_timesteps(n, device, dtype):
    """
    EPSS (Empirically Pruned Step Sampling) 时间步调度

    为少步采样优化的预定义时间步，在低 NFE（Number of Function Evaluations）下效果更好。
    """
    dt = 1 / 32
    predefined_timesteps = {
        5: [0, 2, 4, 8, 16, 32],
        6: [0, 2, 4, 6, 8, 16, 32],
        7: [0, 2, 4, 6, 8, 16, 24, 32],
        10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
        12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
    }
    t = predefined_timesteps.get(n, [])
    if not t:
        return torch.linspace(0, 1, n + 1, device=device, dtype=dtype)
    return dt * torch.tensor(t, device=device, dtype=dtype)


class CFM(nn.Module):
    """
    Conditional Flow Matching (CFM)

    流匹配框架，包装 DiT 模型进行训练和采样。

    训练目标：预测速度场 u_t(x) = x_1 - x_0
    - x_0: 噪声（标准高斯）
    - x_1: 目标数据（真实潜变量）
    - x_t = (1-t) * x_0 + t * x_1: 插值路径

    损失：MSE(predicted_flow, true_flow)

    推理：从噪声出发，沿预测的速度场积分到数据分布。

    Args:
        model: DiT 模型
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, cond, target, latent_history, mask, patch_size):
        """
        训练前向：计算流匹配损失

        Args:
            cond: [B, L_cond, D_llm] LLM 条件（隐藏状态）
            target: [B, patch_size, D_latent] 目标潜变量（干净数据 x_1）
            latent_history: [B, L_history, D_latent] 历史潜变量
            mask: [B, patch_size] mask
            patch_size: 当前 patch 大小

        Returns:
            loss: 标量 MSE 损失
        """
        x1 = target
        batch, dtype = x1.shape[0], x1.dtype

        # 采样噪声 x_0 ~ N(0, I)
        x0 = torch.randn_like(x1)

        # 随机采样时间步 t ~ U(0, 1)
        time = torch.rand((batch,), dtype=dtype, device=self.device)

        # 构造 x_t = (1-t) * x_0 + t * x_1
        t = time.unsqueeze(-1).unsqueeze(-1)
        x = (1 - t) * x0 + t * x1

        # 真实流 = x_1 - x_0
        flow = x1 - x0

        # DiT 预测流
        pred = self.model(x=x, t=time, c=cond, latent_history=latent_history, mask=mask.to(torch.bool))
        pred = pred[:, -patch_size:, :]

        # MSE 损失（仅计算非 mask 区域）
        loss = F.mse_loss(pred, flow, reduction="none")
        mask = (mask == 1)
        loss = loss[mask]

        return loss.mean()

    @torch.no_grad()
    def sample(
        self,
        noise,
        c,
        latent_history,
        steps=10,
        cfg_scale=1.0,
        sway_sampling_coef=-1.0,
        seed: int | None = None,
        use_epss=True,
        patch_size=1,
    ):
        """
        推理采样：从噪声生成潜变量

        Args:
            noise: [B, D_latent, T] 初始噪声
            c: [B, L_cond, D_llm] LLM 条件
            latent_history: [B, L_history, D_latent] 历史潜变量
            steps: 采样步数
            cfg_scale: CFG 强度
            sway_sampling_coef: Sway 采样系数
            seed: 随机种子
            use_epss: 是否使用 EPSS 时间步调度
            patch_size: patch 大小

        Returns:
            out: [B, T, D_latent] 生成的潜变量
            trajectory: 完整采样轨迹
        """
        def fn(t, x):
            """速度场函数"""
            if cfg_scale < 1e-5:
                pred = self.model(
                    x=x, time=t, latent_history=latent_history
                )
                return pred

            # CFG：条件 + 无条件预测
            pred_cfg = self.model.forward_with_cfg(
                x=x, t=t, c=c, latent_history=latent_history,
                cfg_scale=cfg_scale, patch_size=patch_size,
            )
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            # 引导公式：pred + w * (pred - null_pred)
            return pred + (pred - null_pred) * cfg_scale

        y0 = noise.transpose(1, 2)
        t_start = 0

        # 时间步调度
        if t_start == 0 and use_epss:
            t = get_epss_timesteps(steps, device=self.device, dtype=noise.dtype)
        else:
            t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=noise.dtype)

        # Sway 采样时间步调整
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        # ODE 求解
        solver = Solver(fn, y0)
        trajectory = solver.integrate(t)
        sampled = trajectory[-1]
        out = sampled

        return out, trajectory


# ============================================================
# 第四部分：FlowLoss - 扩散损失包装
# ============================================================

class FlowLoss(nn.Module):
    """
    扩散损失模块

    封装 CFM + DiT，提供统一的训练损失计算和采样接口。

    Args:
        z_channels: 潜变量通道数
        llm_cond_dim: LLM 条件维度
        **kwargs: DiT 的其他参数
    """

    def __init__(self, z_channels, llm_cond_dim, **kwargs):
        super(FlowLoss, self).__init__()
        self.z_channels = z_channels
        self.cfm = CFM(
            model=DiT(
                in_channels=z_channels,
                llm_cond_dim=llm_cond_dim,
                **kwargs
            )
        )

    def forward(self, cond, target, latent_history, mask, patch_size):
        """计算训练损失"""
        return self.cfm(
            cond=cond, target=target,
            latent_history=latent_history, mask=mask,
            patch_size=patch_size
        )

    def sample(self, z, latent_history, cfg=1.0, patch_size=1):
        """采样生成潜变量"""
        noise = torch.randn(z.shape[0], self.z_channels, latent_history.shape[1]).cuda()
        sampled_token_latent = self.cfm.sample(
            noise=noise, c=z, latent_history=latent_history,
            cfg_scale=cfg, patch_size=patch_size
        )
        return sampled_token_latent


# ============================================================
# 第五部分：LLM + 双任务头 整体架构说明
# ============================================================

"""
Ming-UniAudio LLM + DiT 整体架构详解
=====================================

一、输入处理
-----------
1. 文本指令 → 文本 tokenizer → 文本 embedding → 投影到 LLM 维度
2. 音频波形 → VAE.encode_unified_emb → 统一语义特征 z_uni → 投影到 LLM 维度
3. 将文本 token 和音频 token 按顺序拼接成统一序列

二、LLM 主干
-----------
- 使用 Qwen2 系列 LLM 作为主干（如 Qwen2.5-7B 等）
- 输入：文本 + 音频 的统一嵌入序列
- 输出：每个位置的隐藏状态 [B, L, D_llm]
- 注意力：因果自注意力（自回归）

三、双任务头
-----------
1. 文本头（LM Head）
   - 位置：整个序列的所有 token 位置
   - 功能：预测下一个 token（包括文本 CoT 和特殊 token）
   - 输出：词表大小的 logits
   - 作用：生成 Chain-of-Thought (CoT) 推理步骤，增强编辑能力

2. 扩散头（DiT + CFM）
   - 位置：在特殊触发 token 后激活
   - 功能：以 LLM 隐藏状态为条件，生成音频潜变量
   - 输出：音频潜变量 [B, T, D_latent]
   - 作用：生成/编辑音频内容

四、编辑任务的工作流程
----------------------
1. 输入：指令文本 + 原始音频（通过 VAE 编码为 z_uni）
2. LLM 处理整个序列，生成 CoT（文本头）
3. 当遇到生成触发时，LLM 的隐藏状态作为条件传给扩散头
4. 扩散头（DiT）以自回归 patch 方式生成音频潜变量
5. 生成的潜变量通过 VAE.decode 还原为波形

五、训练目标
-----------
- 文本损失：交叉熵（LM Head 预测文本 token）
- 扩散损失：流匹配 MSE（DiT 预测速度场）
- 加权组合总损失

六、关键创新点
-------------
1. VAE 三阶段训练，语义模块主动迎合 LLM 语义空间
2. LLM 作为统一主干，同时处理理解和生成
3. 文本头 + 扩散头的双头设计，兼顾推理和生成
4. Flow Matching 替代扩散，采样效率高
5. 无需时间戳的自由形式编辑
"""


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Ming-UniAudio LLM + DiT 模型主干说明")
    print("=" * 70)

    print("""
模型架构：

  ┌─────────────────────────────────────────────────────┐
  │              统一输入序列 (LLM 输入)                  │
  │  [文本指令] [音频语义 z_uni] [特殊标记] [CoT...]      │
  └────────────────────────┬────────────────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  LLM (Qwen2)    │  ← 主干模型
                  │  因果自注意力   │
                  └────────┬────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
      ┌───────────────┐        ┌───────────────┐
      │   文本头      │        │   扩散头      │
      │  (LM Head)    │        │  (DiT + CFM)  │
      └───────┬───────┘        └───────┬───────┘
              │                        │
              ▼                        ▼
        CoT 文本输出          音频潜变量 z_audio
                                     │
                                     ▼
                              VAE Decoder
                                     │
                                     ▼
                                 波形音频

DiT 输入结构：
  [时间嵌入 + LLM条件, 历史潜变量, 当前带噪潜变量]
  ↓
  多层 DiT Block (RoPE + RMSNorm + FFN)
  ↓
  预测流 (velocity field)

关键参数：
  DiT hidden_size:  1024
  DiT depth:        28 层
  DiT num_heads:    16
  DiT mlp_ratio:    4.0
  潜变量维度:        64 (VAE latent_dim)
  LLM 条件维度:      896 (或对应 LLM hidden_size)
  CFG dropout:      0.1
  采样步数:          5-16 步 (EPSS 调度)
    """)
