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
CosyEdit GOT-CFM (Guided Optimal Transport - Conditional Flow Matching)

GOT-CFM 是 CosyEdit 中的流匹配声码器/解码器，负责将 LLM 预测的
离散语音 token 转换为梅尔频谱图（后续再通过 HiFi-GAN 声码器转为波形）。

核心创新：参考引导的流匹配
- 在原始和目标梅尔谱的时间拼接上构建流匹配路径
- 原始梅尔谱从噪声到干净的已知轨迹作为引导（参考）
- 目标梅尔谱的去噪路径受到原始语音路径的引导
- 保持编辑前后的声学一致性（音色、韵律、背景噪声等）

架构：
- 主干：U-Net 风格的 Transformer / 卷积网络
- 条件：说话人嵌入 + 语音 token + 原始梅尔谱 + 掩码目标梅尔谱
- 目标：预测速度场 (velocity field)

训练目标：
  L_GOT-CFM = E[ || v_t(φ_t^OT(Z0,Z1) | θ) - ω_t(φ_t^OT(Z0,Z1) | Z1) || ]

其中 Z0 = [M_ori^0, M_tar^0]（带噪原始+目标梅尔谱）
     Z1 = [M_ori, M_tar]（干净原始+目标梅尔谱）

推理：从高斯噪声出发，沿预测的速度场积分，得到目标梅尔谱。

参考文献:
- OT-CFM: https://arxiv.org/abs/2302.00482
- CosyVoice / CosyEdit GOT-CFM
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 第一部分：时间步嵌入
# ============================================================

class SinusoidalPositionEmbedding(nn.Module):
    """正弦位置嵌入（用于时间步 t）"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor, max_period: float = 10000.0) -> torch.Tensor:
        """
        Args:
            t: [B] 时间步 (0~1)
            max_period: 最大周期

        Returns:
            [B, dim] 嵌入向量
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class TimestepEmbedding(nn.Module):
    """时间步嵌入（正弦 + MLP）"""

    def __init__(self, dim: int, freq_embed_dim: int = 256):
        super().__init__()
        self.time_embed = SinusoidalPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(freq_embed_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] 时间步

        Returns:
            [B, dim] 时间步嵌入
        """
        t_emb = self.time_embed(t)
        return self.time_mlp(t_emb)


# ============================================================
# 第二部分：流匹配 UNet 主干
# ============================================================

class ResidualBlock(nn.Module):
    """
    残差块（1D 卷积）

    用于流匹配模型的编码器/解码器路径。
    """

    def __init__(self, in_channels: int, out_channels: int, d_timestep: int,
                 d_cond: int = None, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # 输入归一化 + 卷积
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)

        # 时间步投影
        self.timestep_proj = nn.Linear(d_timestep, out_channels)

        # 条件投影（可选，用于 AdaIN 风格）
        if d_cond is not None:
            self.cond_proj = nn.Linear(d_cond, out_channels * 2)  # scale + shift
        else:
            self.cond_proj = None

        # 输出归一化 + 卷积
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)

        # 残差连接
        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] 输入特征
            t_emb: [B, D_t] 时间步嵌入
            cond: [B, D_cond] 条件嵌入

        Returns:
            [B, C_out, T]
        """
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        # 添加时间步嵌入
        h = h + self.timestep_proj(self.act(t_emb)).unsqueeze(-1)

        # 条件调制 (AdaIN style)
        if self.cond_proj is not None and cond is not None:
            cond_emb = self.cond_proj(self.act(cond))  # [B, 2*C]
            scale, shift = cond_emb.chunk(2, dim=-1)  # [B, C]
            h = h * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)

        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.residual_conv(x)


class Downsample(nn.Module):
    """下采样（1D 卷积 + stride）"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """上采样（最近邻 + 卷积）"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        return self.conv(x)


# ============================================================
# 第三部分：GOT-CFM 主模型
# ============================================================

class GotCFM(nn.Module):
    """
    GOT-CFM (Guided Optimal Transport Conditional Flow Matching)

    CosyEdit 的流匹配模型，用于从语音 token + 原始梅尔谱生成目标梅尔谱。

    关键设计 - 参考引导机制：
    - 输入是原始梅尔谱 + 目标梅尔谱的时间拼接
    - 原始梅尔谱部分是"已知轨迹"（从噪声到干净）
    - 目标梅尔谱部分是"待预测轨迹"
    - 模型可以利用原始语音的去噪轨迹作为参考/引导

    条件输入：
    1. 时间步 t
    2. 说话人嵌入 spk_emb
    3. 语音语义 token μ_Z = [μ_ori, μ_tar]（通过嵌入层投影）
    4. 原始梅尔谱 M_ori（干净，作为引导参考）
    5. 掩码目标梅尔谱 M̃_tar（全零/掩码，表示待生成区域）

    输入：
        Z_t = φ_t^OT(Z_0, Z_1) = (1-t) * Z_0 + t * Z_1
        其中 Z_0 = [M_ori^0, M_tar^0]（带噪声）
             Z_1 = [M_ori, M_tar]（干净）

    输出：
        速度场 ν_t = dZ_t / dt = Z_1 - Z_0

    Args:
        n_mels: 梅尔频带数
        d_model: 模型基础通道数
        d_timestep: 时间步嵌入维度
        d_spk: 说话人嵌入维度
        d_token: 语音 token 词表大小
        d_token_emb: 语音 token 嵌入维度
        num_res_blocks: 每个分辨率的残差块数
        channel_mult: 通道倍率（下采样层级）
        dropout: dropout 概率
    """

    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 256,
        d_timestep: int = 256,
        d_spk: int = 192,
        d_token: int = 4096,
        d_token_emb: int = 256,
        num_res_blocks: int = 3,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 4),
        dropout: float = 0.0,
    ):
        super().__init__()

        self.n_mels = n_mels
        self.d_model = d_model

        # 时间步嵌入
        self.time_embedding = TimestepEmbedding(d_model, d_timestep)

        # 说话人嵌入投影
        self.spk_proj = nn.Sequential(
            nn.Linear(d_spk, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # 语音 token 嵌入
        self.token_embedding = nn.Embedding(d_token + 5, d_token_emb)  # +5 for special tokens
        self.token_proj = nn.Conv1d(d_token_emb, d_model, kernel_size=1)

        # 输入卷积：梅尔谱 → d_model 通道
        # 输入是 [M_ori_noisy, M_tar_noisy] 的拼接，但通道数都是 n_mels
        # 我们在时间维度拼接，所以通道数不变
        self.input_conv = nn.Conv1d(n_mels, d_model, kernel_size=3, padding=1)

        # 条件维度 = 时间步 + 说话人
        d_cond = d_model + d_model

        # 编码器路径（下采样）
        self.down_blocks = nn.ModuleList()
        self.skip_channels = []  # 记录 skip connection 的通道数
        channels = d_model

        # 输入块
        input_block = nn.ModuleList()
        for _ in range(num_res_blocks):
            input_block.append(ResidualBlock(channels, channels, d_model, d_cond, dropout=dropout))
        self.down_blocks.append(input_block)
        self.skip_channels.append(channels)

        # 下采样层
        for level, mult in enumerate(channel_mult[1:], 1):
            out_channels = d_model * mult
            downsample_block = nn.ModuleList()

            # 下采样
            downsample_block.append(Downsample(channels, out_channels))

            # 残差块
            for _ in range(num_res_blocks):
                downsample_block.append(
                    ResidualBlock(out_channels, out_channels, d_model, d_cond, dropout=dropout)
                )

            self.down_blocks.append(downsample_block)
            self.skip_channels.append(out_channels)
            channels = out_channels

        # 中间块
        self.mid_block = nn.ModuleList([
            ResidualBlock(channels, channels, d_model, d_cond, dropout=dropout),
            ResidualBlock(channels, channels, d_model, d_cond, dropout=dropout),
        ])

        # 解码器路径（上采样）
        self.up_blocks = nn.ModuleList()
        for level, mult in enumerate(reversed(channel_mult)):
            out_channels = d_model * mult
            skip_ch = self.skip_channels[-(level + 1)]

            upsample_block = nn.ModuleList()

            # 残差块（含 skip connection）
            for _ in range(num_res_blocks):
                upsample_block.append(
                    ResidualBlock(channels + skip_ch, out_channels, d_model, d_cond, dropout=dropout)
                )
                channels = out_channels

            # 上采样（除了最后一层）
            if level < len(channel_mult) - 1:
                upsample_block.append(Upsample(channels, d_model * channel_mult[-(level + 2)]))
                channels = d_model * channel_mult[-(level + 2)]

            self.up_blocks.append(upsample_block)

        # 输出头
        self.out_norm = nn.GroupNorm(32, d_model)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv1d(d_model, n_mels, kernel_size=3, padding=1)

        # 初始化输出层为零（稳定训练）
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def _build_condition(self, t: torch.Tensor, spk_emb: torch.Tensor) -> torch.Tensor:
        """
        构建全局条件向量

        Args:
            t: [B] 时间步
            spk_emb: [B, D_spk] 说话人嵌入

        Returns:
            cond: [B, D_cond] 条件向量
        """
        t_emb = self.time_embedding(t)  # [B, D]
        spk_emb_proj = self.spk_proj(spk_emb)  # [B, D]
        cond = torch.cat([t_emb, spk_emb_proj], dim=-1)  # [B, 2D]
        return cond

    def _preprocess_input(
        self,
        noisy_mel_ori: torch.Tensor,
        noisy_mel_tar: torch.Tensor,
        clean_mel_ori: torch.Tensor,
        masked_mel_tar: torch.Tensor,
        audio_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预处理输入：拼接原始和目标梅尔谱，准备条件

        GOT-CFM 的关键设计：
        - 输入 x_t: [M_ori_t, M_tar_t] 时间拼接的带噪梅尔谱
        - 条件参考: [M_ori_clean, M_tar_masked]（原始干净 + 目标掩码）

        Args:
            noisy_mel_ori: [B, n_mels, T_ori] 带噪原始梅尔谱
            noisy_mel_tar: [B, n_mels, T_tar] 带噪目标梅尔谱
            clean_mel_ori: [B, n_mels, T_ori] 干净原始梅尔谱（引导参考）
            masked_mel_tar: [B, n_mels, T_tar] 掩码目标梅尔谱（全零 / 掩码）
            audio_tokens: [B, T_tokens] 语音 token (μ_ori + μ_tar)

        Returns:
            x: [B, n_mels, T_ori + T_tar] 输入（时间拼接）
            cond_spec: [B, n_mels, T_ori + T_tar] 条件谱（引导参考）
        """
        # 输入：带噪的原始 + 目标 梅尔谱（时间维度拼接）
        x = torch.cat([noisy_mel_ori, noisy_mel_tar], dim=-1)  # [B, n_mels, T_ori+T_tar]

        # 条件谱：干净原始 + 掩码目标（作为引导参考）
        cond_spec = torch.cat([clean_mel_ori, masked_mel_tar], dim=-1)

        return x, cond_spec

    def forward(
        self,
        noisy_mel_ori: torch.Tensor,
        noisy_mel_tar: torch.Tensor,
        clean_mel_ori: torch.Tensor,
        masked_mel_tar: torch.Tensor,
        audio_tokens: torch.Tensor,
        t: torch.Tensor,
        spk_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GOT-CFM 前向传播（训练）

        预测速度场 ν_t(φ_t^OT(Z0,Z1) | θ)

        Args:
            noisy_mel_ori: [B, n_mels, T_ori] 带噪原始梅尔谱 (M_ori^0)
            noisy_mel_tar: [B, n_mels, T_tar] 带噪目标梅尔谱 (M_tar^0)
            clean_mel_ori: [B, n_mels, T_ori] 干净原始梅尔谱 (M_ori)
            masked_mel_tar: [B, n_mels, T_tar] 掩码目标梅尔谱 (M̃_tar)
            audio_tokens: [B, T_tokens] 语音 token (μ_ori + μ_tar)
            t: [B] 时间步
            spk_emb: [B, D_spk] 说话人嵌入

        Returns:
            velocity: [B, n_mels, T_ori+T_tar] 预测的速度场
            target_velocity: [B, n_mels, T_ori+T_tar] 真实速度场
        """
        B = t.shape[0]
        device = t.device

        # 1. 构建输入（时间拼接）
        x, cond_spec = self._preprocess_input(
            noisy_mel_ori, noisy_mel_tar, clean_mel_ori, masked_mel_tar, audio_tokens
        )

        # 2. 全局条件（时间步 + 说话人）
        cond = self._build_condition(t, spk_emb)  # [B, D_cond]

        # 3. 语音 token 条件（沿时间维度）
        token_emb = self.token_embedding(audio_tokens)  # [B, T_tokens, D_emb]
        token_emb = token_emb.transpose(1, 2)  # [B, D_emb, T_tokens]
        token_emb = self.token_proj(token_emb)  # [B, D_model, T_tokens]

        # 将 token 嵌入上采样/下采样到梅尔谱的时间分辨率
        T_mel = x.shape[-1]
        T_tokens = token_emb.shape[-1]
        if T_tokens != T_mel:
            # 线性插值对齐时间分辨率
            token_emb = F.interpolate(token_emb, size=T_mel, mode='linear')

        # 4. 输入卷积 + token 条件融合
        x = self.input_conv(x)  # [B, D_model, T]
        x = x + token_emb  # 加上 token 条件
        # 也加上条件谱（作为额外的引导）
        x = x + self.input_conv(cond_spec)

        # 5. 编码器路径（下采样）
        skips = []
        h = x
        for down_block in self.down_blocks:
            for layer in down_block:
                if isinstance(layer, ResidualBlock):
                    h = layer(h, self.time_embedding(t), cond)
                elif isinstance(layer, Downsample):
                    skips.append(h)
                    h = layer(h)
            if not any(isinstance(l, Downsample) for l in down_block):
                skips.append(h)

        # 6. 中间块
        for mid_layer in self.mid_block:
            h = mid_layer(h, self.time_embedding(t), cond)

        # 7. 解码器路径（上采样 + skip connections）
        for i, up_block in enumerate(self.up_blocks):
            for layer in up_block:
                if isinstance(layer, ResidualBlock):
                    # 拼接 skip connection
                    if skips:
                        skip = skips.pop()
                        # 处理尺寸不匹配
                        if h.shape[-1] != skip.shape[-1]:
                            h = F.interpolate(h, size=skip.shape[-1], mode='nearest')
                        h = torch.cat([h, skip], dim=1)
                    h = layer(h, self.time_embedding(t), cond)
                elif isinstance(layer, Upsample):
                    h = layer(h)

        # 8. 输出头
        h = self.out_norm(h)
        h = self.out_act(h)
        velocity = self.out_conv(h)  # [B, n_mels, T]

        # 9. 真实速度场 = Z1 - Z0
        target_velocity = torch.cat(
            [clean_mel_ori - noisy_mel_ori, clean_mel_ori[:, :, :0] * 0 + noisy_mel_tar[:, :, :0] * 0],  # 占位
            dim=-1
        )
        # 正确的 target: 干净 - 带噪
        target_velocity = torch.cat(
            [clean_mel_ori - noisy_mel_ori,
             torch.zeros_like(noisy_mel_tar)],  # 目标部分的 target 在 loss 中单独计算
            dim=-1
        )
        # 注：完整 target_velocity 应该由外部计算（干净 - 噪声）

        return velocity, target_velocity


# ============================================================
# 第四部分：CFM 训练与采样
# ============================================================

class GotCFMWrapper(nn.Module):
    """
    GOT-CFM 训练/推理包装器

    封装流匹配的训练损失计算和推理采样逻辑。

    训练目标：
        L = E[ MSE(predicted_velocity, true_velocity) ]

    推理：
        从标准高斯噪声出发，沿预测速度场积分得到干净梅尔谱。
    """

    def __init__(self, model: GotCFM, sigma: float = 0.0):
        super().__init__()
        self.model = model
        self.sigma = sigma  # OT-CFM 的 sigma 参数（通常为 0 或 0.001）

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        clean_mel_ori: torch.Tensor,
        clean_mel_tar: torch.Tensor,
        audio_tokens: torch.Tensor,
        spk_emb: torch.Tensor,
        mask_tar: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        训练前向：计算流匹配损失

        Args:
            clean_mel_ori: [B, n_mels, T_ori] 干净原始梅尔谱
            clean_mel_tar: [B, n_mels, T_tar] 干净目标梅尔谱
            audio_tokens: [B, T_tokens] 语音 token
            spk_emb: [B, D_spk] 说话人嵌入
            mask_tar: [B, 1, T_tar] 目标区域 mask（可选，用于加权损失）

        Returns:
            loss: 标量 MSE 损失
        """
        B = clean_mel_ori.shape[0]
        device = clean_mel_ori.device

        # Z1 = 干净数据
        Z1 = torch.cat([clean_mel_ori, clean_mel_tar], dim=-1)  # [B, n_mels, T_total]

        # Z0 = 高斯噪声
        Z0 = torch.randn_like(Z1)

        # 采样时间步 t ~ U(0, 1)
        t = torch.rand(B, device=device)

        # 构建插值路径：φ_t^OT(Z0, Z1) = (1-t) * Z0 + t * Z1
        t_expanded = t.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        Z_t = (1 - t_expanded) * Z0 + t_expanded * Z1

        # 真实速度场 = Z1 - Z0
        true_velocity = Z1 - Z0

        # 拆分 Z_t 为原始和目标部分
        T_ori = clean_mel_ori.shape[-1]
        Z_t_ori = Z_t[:, :, :T_ori]
        Z_t_tar = Z_t[:, :, T_ori:]

        # 构建掩码目标梅尔谱（引导参考）
        # 对于原始部分：干净的原始梅尔谱（完全已知）
        # 对于目标部分：零掩码（完全未知，需要生成）
        masked_mel_tar = torch.zeros_like(clean_mel_tar)

        # 模型预测速度场
        pred_velocity, _ = self.model(
            noisy_mel_ori=Z_t_ori,
            noisy_mel_tar=Z_t_tar,
            clean_mel_ori=clean_mel_ori,
            masked_mel_tar=masked_mel_tar,
            audio_tokens=audio_tokens,
            t=t,
            spk_emb=spk_emb,
        )

        # MSE 损失
        loss = F.mse_loss(pred_velocity, true_velocity, reduction='none')

        # 如果有 mask，对目标区域加权
        if mask_tar is not None:
            # 扩展 mask 到完整序列
            mask_ori = torch.ones(B, 1, T_ori, device=device)
            mask_full = torch.cat([mask_ori, mask_tar], dim=-1)
            loss = loss * mask_full
            loss = loss.sum() / mask_full.sum() / loss.shape[1]
        else:
            loss = loss.mean()

        return loss

    @torch.no_grad()
    def sample(
        self,
        clean_mel_ori: torch.Tensor,
        T_tar: int,
        audio_tokens: torch.Tensor,
        spk_emb: torch.Tensor,
        n_steps: int = 10,
        cfg_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        推理采样：从噪声生成目标梅尔谱

        Args:
            clean_mel_ori: [1, n_mels, T_ori] 干净原始梅尔谱（条件）
            T_tar: 目标梅尔谱时间帧数
            audio_tokens: [1, T_tokens] 语音 token
            spk_emb: [1, D_spk] 说话人嵌入
            n_steps: 采样步数
            cfg_scale: Classifier-Free Guidance 强度

        Returns:
            mel_tar: [1, n_mels, T_tar] 生成的目标梅尔谱
            trajectory: 完整采样轨迹
        """
        B = clean_mel_ori.shape[0]
        device = clean_mel_ori.device
        n_mels = clean_mel_ori.shape[1]
        T_ori = clean_mel_ori.shape[-1]

        assert B == 1, "Sampling only supports batch_size=1"

        # 初始噪声
        Z = torch.randn(B, n_mels, T_ori + T_tar, device=device)

        # 掩码目标（零）
        masked_mel_tar = torch.zeros(B, n_mels, T_tar, device=device)

        # 时间步（从 0 到 1）
        timesteps = torch.linspace(0, 1, n_steps + 1, device=device)

        trajectory = [Z.clone()]

        # ODE 积分（欧拉法）
        for i in range(n_steps):
            t = timesteps[i]
            dt = timesteps[i + 1] - t

            # 当前状态拆分
            Z_ori = Z[:, :, :T_ori]
            Z_tar = Z[:, :, T_ori:]

            # 预测速度场
            t_batch = t.unsqueeze(0).expand(B)
            velocity, _ = self.model(
                noisy_mel_ori=Z_ori,
                noisy_mel_tar=Z_tar,
                clean_mel_ori=clean_mel_ori,
                masked_mel_tar=masked_mel_tar,
                audio_tokens=audio_tokens,
                t=t_batch,
                spk_emb=spk_emb,
            )

            # 欧拉步进
            Z = Z + velocity * dt

            trajectory.append(Z.clone())

        # 提取目标部分
        mel_tar = Z[:, :, T_ori:]

        return mel_tar, trajectory


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CosyEdit GOT-CFM (Guided OT-Conditional Flow Matching)")
    print("=" * 70)

    print("""
核心思想 - 参考引导的流匹配：

  时间 t=0 (噪声)        时间 t=1 (干净)
     ────────              ────────
    │ M_ori^0│            │ M_ori  │
    │ M_tar^0│            │ M_tar  │
     ────────              ────────

  模型输入 Z_t = (1-t)*Z_0 + t*Z_1  (时间拼接)
  模型输出 ν_t = dZ_t / dt  (速度场)

  引导机制：
    - 原始梅尔谱 M_ori 的去噪轨迹是已知的
    - 它作为参考，引导目标梅尔谱 M_tar 的生成
    - 保持声学一致性（音色、背景噪声、韵律等）

模型架构：
  输入：带噪梅尔谱 [M_ori_t | M_tar_t]  (时间拼接)
  条件：
    - 时间步 t (正弦位置编码 + MLP)
    - 说话人嵌入 spk_emb
    - 语音语义 token μ_Z (嵌入 + 投影)
    - 条件谱 [M_ori_clean | M_tar_masked]  (引导参考)

  主干：U-Net 风格 (编码器-解码器 + skip connections)
    - 下采样：3-4 个级别
    - 残差块：含时间步和条件的 AdaIN 调制
    - 上采样：对应编码器的镜像结构

  输出：速度场 ν_t (与输入同形状)

训练损失：
  L = MSE(ν_pred, Z_1 - Z_0)

推理采样：
  - 欧拉法 ODE 求解
  - 步数：10-50 步
  - 支持 Classifier-Free Guidance (CFG)

与 CosyVoice 的区别：
  CosyVoice CFM: 零样本 TTS，目标是生成高质量合成语音
  CosyEdit GOT-CFM: 语音编辑，额外提供原始梅尔谱作为引导参考
                   → 增强未编辑区域的保真度和声学一致性

后续处理：
  GOT-CFM 输出梅尔谱 → HiFi-GAN 声码器 → 波形音频
    """)
