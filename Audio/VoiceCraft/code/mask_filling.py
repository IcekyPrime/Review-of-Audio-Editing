# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
VoiceCraft 掩码填充（Mask Filling）核心代码提取

VoiceCraft 将语音编辑任务建模为掩码填充任务：
- 训练时：随机选择音频片段进行掩码，模型需要根据文本条件和未掩码的音频上下文
  来预测被掩码的部分
- 推理时：用户指定需要编辑的区间作为掩码，模型生成新的音频 token 填充该区间

核心设计：
1. 随机采样掩码区间 (prepare_mask_intervals)
2. 重排序列：非掩码部分在前，掩码部分在后，中间用 mask token 分隔
3. 应用 DelayedPattern 进行码本间的延迟排列
4. 插入可学习的 mask embedding
5. 自回归生成掩码部分的音频 token
6. 还原模式并拼接回完整序列
"""

import random
from collections import namedtuple
from dataclasses import dataclass
from functools import lru_cache
import typing as tp
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 第一部分：码本模式（Codebook Pattern）相关
# 用于多码本序列的交错排列，支持延迟模式、并行模式等
# ============================================================

LayoutCoord = namedtuple('LayoutCoord', ['t', 'q'])  # (timestep, codebook index)
PatternLayout = tp.List[tp.List[LayoutCoord]]  # Sequence of coordinates


@dataclass
class Pattern:
    """Base implementation of a pattern over a sequence with multiple codebooks.

    The codebook pattern consists in a layout, defining for each sequence step
    the list of coordinates of each codebook timestep in the resulting interleaved sequence.
    """
    layout: PatternLayout
    timesteps: int
    n_q: int

    def __post_init__(self):
        assert len(self.layout) > 0
        assert self.layout[0] == []
        self._validate_layout()
        self._build_reverted_sequence_scatter_indexes = lru_cache(100)(self._build_reverted_sequence_scatter_indexes)
        self._build_pattern_sequence_scatter_indexes = lru_cache(100)(self._build_pattern_sequence_scatter_indexes)

    def _validate_layout(self):
        """验证模式布局的有效性"""
        q_timesteps = {q: 0 for q in range(self.n_q)}
        for s, seq_coords in enumerate(self.layout):
            if len(seq_coords) > 0:
                qs = set()
                for coord in seq_coords:
                    qs.add(coord.q)
                    last_q_timestep = q_timesteps[coord.q]
                    assert coord.t >= last_q_timestep, \
                        f"Past timesteps are found in the sequence for codebook = {coord.q} at step {s}"
                    q_timesteps[coord.q] = coord.t
                assert len(qs) == len(seq_coords), \
                    f"Multiple entries for a same codebook are found at step {s}"

    @property
    def num_sequence_steps(self):
        return len(self.layout) - 1

    @property
    def max_delay(self):
        max_t_in_seq_coords = 0
        for seq_coords in self.layout[1:]:
            for coords in seq_coords:
                max_t_in_seq_coords = max(max_t_in_seq_coords, coords.t + 1)
        return max_t_in_seq_coords - self.timesteps

    @property
    def valid_layout(self):
        valid_step = len(self.layout) - self.max_delay
        return self.layout[:valid_step]

    def _build_pattern_sequence_scatter_indexes(self, timesteps: int, n_q: int, keep_only_valid_steps: bool,
                                                device: tp.Union[torch.device, str] = 'cpu'):
        """构建模式序列的散射索引"""
        assert n_q == self.n_q, f"invalid number of codebooks: {n_q} != {self.n_q}"
        assert timesteps <= self.timesteps, "invalid number of timesteps"
        ref_layout = self.valid_layout if keep_only_valid_steps else self.layout
        import numpy as np
        indexes = torch.zeros(n_q, len(ref_layout), dtype=torch.long).numpy()
        mask = torch.zeros(n_q, len(ref_layout), dtype=torch.bool).numpy()
        indexes[:] = n_q * timesteps  # special token position
        for s, sequence_coords in enumerate(ref_layout):
            for coords in sequence_coords:
                if coords.t < timesteps:
                    indexes[coords.q, s] = coords.t + coords.q * timesteps
                    mask[coords.q, s] = 1
        indexes = torch.from_numpy(indexes).to(device)
        mask = torch.from_numpy(mask).to(device)
        return indexes, mask

    def build_pattern_sequence(self, z: torch.Tensor, special_token: int, keep_only_valid_steps: bool = False):
        """从输入张量构建对应模式的序列（交错排列）
        
        Args:
            z: [B, K, T] 多码本序列
            special_token: 填充特殊 token
            keep_only_valid_steps: 是否只保留有效步
            
        Returns:
            values: [B, K, S] 交错排列后的序列
            indexes: [K, S] 索引
            mask: [K, S] 有效掩码
        """
        B, K, T = z.shape
        indexes, mask = self._build_pattern_sequence_scatter_indexes(
            T, K, keep_only_valid_steps=keep_only_valid_steps, device=str(z.device)
        )
        z = z.view(B, -1)
        z = torch.cat([z, torch.zeros_like(z[:, :1]) + special_token], dim=1)
        values = z[:, indexes.view(-1)]
        values = values.view(B, K, indexes.shape[-1])
        return values, indexes, mask

    def _build_reverted_sequence_scatter_indexes(self, sequence_steps: int, n_q: int,
                                                 keep_only_valid_steps: bool = False,
                                                 is_model_output: bool = False,
                                                 device: tp.Union[torch.device, str] = 'cpu'):
        """构建还原序列的散射索引"""
        ref_layout = self.valid_layout if keep_only_valid_steps else self.layout
        timesteps = self.timesteps
        assert n_q == self.n_q
        assert sequence_steps <= len(ref_layout)
        if is_model_output:
            ref_layout = ref_layout[1:]
        import numpy as np
        indexes = torch.zeros(n_q, timesteps, dtype=torch.long).numpy()
        mask = torch.zeros(n_q, timesteps, dtype=torch.bool).numpy()
        indexes[:] = n_q * sequence_steps
        for s, sequence_codes in enumerate(ref_layout):
            if s < sequence_steps:
                for code in sequence_codes:
                    if code.t < timesteps:
                        indexes[code.q, code.t] = s + code.q * sequence_steps
                        mask[code.q, code.t] = 1
        indexes = torch.from_numpy(indexes).to(device)
        mask = torch.from_numpy(mask).to(device)
        return indexes, mask

    def revert_pattern_logits(self, logits: torch.Tensor, special_token: float, keep_only_valid_steps: bool = False):
        """将模型输出的 logits 还原回原始序列排列
        
        Args:
            logits: [B, card, K, S] 模型输出 logits
            special_token: 特殊 token 值
            keep_only_valid_steps: 是否只保留有效步
            
        Returns:
            values: [B, card, K, T] 还原后的 logits
        """
        B, card, K, S = logits.shape
        indexes, mask = self._build_reverted_sequence_scatter_indexes(
            S, K, keep_only_valid_steps, is_model_output=True, device=logits.device
        )
        logits = logits.reshape(B, card, -1)
        logits = torch.cat([logits, torch.zeros_like(logits[:, :, :1]) + special_token], dim=-1)
        values = logits[:, :, indexes.view(-1)]
        values = values.view(B, card, K, indexes.shape[-1])
        return values, indexes, mask


class CodebooksPatternProvider(ABC):
    """码本模式提供器抽象基类"""
    def __init__(self, n_q: int, cached: bool = True):
        assert n_q > 0
        self.n_q = n_q
        self.get_pattern = lru_cache(100)(self.get_pattern)

    @abstractmethod
    def get_pattern(self, timesteps: int) -> Pattern:
        raise NotImplementedError()


class DelayedPatternProvider(CodebooksPatternProvider):
    """延迟模式提供器 - VoiceCraft 默认使用的模式
    
    码本在序列中具有不同的延迟，第 q 个码本延迟 q 步。
    例如 timesteps=4, n_q=3 时：
    原始: [[1,2,3,4], [1,2,3,4], [1,2,3,4]]
    结果: [[S,1,2,3,4], [S,S,1,2,3], [S,S,S,1,2]]
    (S 为特殊 token)
    """
    def __init__(self, n_q: int, delays: tp.Optional[tp.List[int]] = None,
                 flatten_first: int = 0, empty_initial: int = 0):
        super().__init__(n_q)
        if delays is None:
            delays = list(range(n_q))
        self.delays = delays
        self.flatten_first = flatten_first
        self.empty_initial = empty_initial
        assert len(self.delays) == self.n_q
        assert sorted(self.delays) == self.delays

    def get_pattern(self, timesteps: int) -> Pattern:
        out: PatternLayout = [[]]
        max_delay = max(self.delays)
        if self.empty_initial:
            out += [[] for _ in range(self.empty_initial)]
        if self.flatten_first:
            for t in range(min(timesteps, self.flatten_first)):
                for q in range(self.n_q):
                    out.append([LayoutCoord(t, q)])
        for t in range(self.flatten_first, timesteps + max_delay):
            v = []
            for q, delay in enumerate(self.delays):
                t_for_q = t - delay
                if t_for_q >= self.flatten_first:
                    v.append(LayoutCoord(t_for_q, q))
            out.append(v)
        return Pattern(out, n_q=self.n_q, timesteps=timesteps)


# ============================================================
# 第二部分：掩码填充核心逻辑
# ============================================================

def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """生成 padding mask
    
    Args:
        lengths: [B] 每个样本的长度
        max_len: 最大长度
        
    Returns:
        [B, max_len] bool tensor，被 mask 的位置为 True
    """
    assert lengths.ndim == 1
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)
    return expaned_lengths >= lengths.unsqueeze(-1)


class VoiceCraftMaskFilling(nn.Module):
    """
    VoiceCraft 掩码填充模块
    
    核心思想：将语音编辑任务建模为掩码填充（mask filling）任务。
    训练时随机掩码部分音频片段，推理时由用户指定编辑区间。
    
    关键流程：
    1. prepare_mask_intervals: 采样掩码区间
    2. rearrange: 将序列重排为 [非掩码部分, mask, 掩码部分, mask, ...] 形式
    3. shift: 应用延迟模式（DelayedPattern）
    4. insert_mask: 插入 mask token 占位符
    5. embed_y: 嵌入音频 token 和 mask embedding
    6. remove_mask + revert_pattern: 生成后还原序列
    """

    def __init__(self, n_codebooks: int, audio_vocab_size: int, d_model: int,
                 max_n_spans: int = 5, mask_sample_dist: str = "uniform",
                 mask_len_min: int = 20, mask_len_max: int = 200,
                 min_gap: int = 50, shuffle_mask_embedding: bool = True,
                 special_first: bool = False, n_special: int = 3,
                 audio_pad_token: int = None, empty_token: int = None,
                 eog: int = None, eos: int = -1):
        super().__init__()

        self.n_codebooks = n_codebooks
        self.audio_vocab_size = audio_vocab_size
        self.d_model = d_model
        self.max_n_spans = max_n_spans
        self.mask_sample_dist = mask_sample_dist
        self.mask_len_min = mask_len_min
        self.mask_len_max = mask_len_max
        self.min_gap = min_gap
        self.shuffle_mask_embedding = shuffle_mask_embedding
        self.special_first = special_first
        self.n_special = n_special
        self.eos_value = eos

        # 特殊 token 定义
        self.empty_token = empty_token if empty_token is not None else audio_vocab_size
        self.eog_value = eog if eog is not None else audio_vocab_size + 1
        self.audio_pad_token = audio_pad_token if audio_pad_token is not None else audio_vocab_size + 2

        # EOG token 参数（不可训练）
        self.eog = nn.Parameter(
            torch.full((self.n_codebooks, 1), self.eog_value, dtype=torch.long),
            requires_grad=False
        )
        if self.eos_value > 0:
            self.eos = nn.Parameter(
                torch.full((self.n_codebooks, 1), self.eos_value, dtype=torch.long),
                requires_grad=False
            )

        # 码本模式 - 使用延迟模式
        self.pattern = DelayedPatternProvider(n_q=self.n_codebooks)

        # 可学习的 mask embedding
        self.mask_embedding = nn.Parameter(
            torch.randn(self.max_n_spans, self.d_model),
            requires_grad=True
        )

        # 音频 token 嵌入层（每个码本一个）
        n_audio_tokens = self.audio_vocab_size + self.n_special
        self.audio_embedding = nn.ModuleList([
            nn.Embedding(n_audio_tokens, self.d_model)
            for _ in range(self.n_codebooks)
        ])

    # --------------------------------------------------------
    # 1. 掩码区间采样
    # --------------------------------------------------------
    def prepare_mask_intervals(self, y_lens):
        """随机采样掩码区间
        
        对于每个样本，随机选择若干个不重叠的区间作为掩码区域。
        
        Args:
            y_lens: [B] 每个样本的音频 token 长度
            
        Returns:
            mask_intervals: list of list of (start, end) 每个样本的掩码区间
            non_mask_intervals: list of list of (start, end) 每个样本的非掩码区间
        """
        mask_intervals = []
        non_mask_intervals = []

        for i, y_len in enumerate(y_lens):
            # 决定掩码 span 的数量
            if self.mask_sample_dist == "uniform":
                n_spans = random.choice(range(1, self.max_n_spans + 1))
            elif "poisson" in self.mask_sample_dist.lower():
                param = float(self.mask_sample_dist[len("poisson"):])
                poisson_sample = torch.poisson(torch.tensor([param]))
                n_spans = int(poisson_sample.clamp(1, self.max_n_spans).item())
            else:
                n_spans = 1

            # 随机采样起始位置（排序后）
            starts = random.sample(range(1, y_len - 1 - self.mask_len_min), n_spans)
            starts = sorted(starts)

            # 移除距离太近的起始点
            for j in range(len(starts) - 1, 0, -1):
                if starts[j] - starts[j - 1] < self.min_gap:
                    del starts[j]
            assert len(starts) > 0, f"no masked span left, y_len: {y_len}, n_spans: {n_spans}"

            # 计算每个掩码区间的长度
            temp_starts = starts + [y_len]
            gaps = [temp_starts[j + 1] - temp_starts[j] for j in range(len(temp_starts) - 1)]

            ends = []
            for j, (start, gap) in enumerate(zip(starts, gaps)):
                mask_len = random.randint(self.mask_len_min, self.mask_len_max)
                # 确保掩码不重叠
                if mask_len > gap - 1:
                    temp_mask_start = 1
                    temp_mask_end = gap - 1
                    mask_len = random.randint(temp_mask_start, temp_mask_end)
                ends.append(start + mask_len)

            mask_intervals.append([(s, e) for s, e in zip(starts, ends)])
            non_mask_intervals.append(
                [(ns, ne) for ns, ne in zip([0] + ends, starts + [y_len])]
            )

        return mask_intervals, non_mask_intervals

    # --------------------------------------------------------
    # 2. 序列重排
    # --------------------------------------------------------
    def rearrange(self, y, non_mask_intervals, mask_intervals):
        """将音频序列重排为非掩码段和掩码段交替的形式
        
        原始序列: [a, b, c, d, e, f] (假设 b-c 被掩码)
        重排后: [a, EOG, d, e, f, EOG, b, c, EOG]
        即: 非掩码段在前, 掩码段在后, 每段末尾加 EOG
        
        Args:
            y: [B, K, T] 音频 token
            non_mask_intervals: 非掩码区间
            mask_intervals: 掩码区间
            
        Returns:
            rearranged_y: list of list of tensors，每个样本重排后的各段
        """
        rearranged_y = []
        for i in range(len(y)):
            if self.eos_value > 0:
                # 有 EOS token 的情况
                cur_y = (
                    [y[i, :, item[0]: item[1]] for item in non_mask_intervals[i][:-1]]
                    + [torch.cat([y[i, :, non_mask_intervals[i][-1][0]: non_mask_intervals[i][-1][1]], self.eos], dim=-1)]
                    + [torch.cat([y[i, :, item[0]: item[1]], self.eog], dim=-1) for item in mask_intervals[i]]
                )
            else:
                # 无 EOS token，每段末尾加 EOG
                cur_y = (
                    [torch.cat([y[i, :, item[0]: item[1]], self.eog], dim=-1) for item in non_mask_intervals[i]]
                    + [torch.cat([y[i, :, item[0]: item[1]], self.eog], dim=-1) for item in mask_intervals[i]]
                )
            rearranged_y.append(cur_y)
        return rearranged_y

    # --------------------------------------------------------
    # 3. 模式偏移（应用延迟模式）
    # --------------------------------------------------------
    def shift(self, rearranged_y):
        """对每个片段应用延迟模式（DelayedPattern）
        
        将并行的多码本序列转换为交错排列的延迟序列。
        
        Args:
            rearranged_y: 重排后的各段
            
        Returns:
            shifted_y: 应用模式偏移后的各段
            patterns: 各段对应的 Pattern 对象
        """
        shifted_y = []
        patterns = []
        for i in range(len(rearranged_y)):
            cur_patterns = [self.pattern.get_pattern(cur_y.shape[1]) for cur_y in rearranged_y[i]]
            out = [
                cur_pattern.build_pattern_sequence(
                    z=cur_y.unsqueeze(0).contiguous(),
                    special_token=self.empty_token,
                    keep_only_valid_steps=False
                )
                for cur_pattern, cur_y in zip(cur_patterns, rearranged_y[i])
            ]
            shifted_y.append([item[0].squeeze(0) for item in out])
            patterns.append(cur_patterns)
        return shifted_y, patterns

    # --------------------------------------------------------
    # 4. 插入 Mask Token
    # --------------------------------------------------------
    def insert_mask(self, shifted_y):
        """在非掩码段和掩码段之间插入 mask token 占位符
        
        实际使用 EOG token 作为占位符，后续在 embed_y 中替换为真正的 mask embedding。
        
        Args:
            shifted_y: 模式偏移后的各段
            
        Returns:
            inserted_y: 插入 mask 后的各段
            mask_position: 每个样本中 mask token 的位置
            mask_value: 每个 mask 使用的 embedding 索引
        """
        inserted_y = []
        mask_position = []
        mask_value = []
        for i in range(len(shifted_y)):
            num_masks = (len(shifted_y[i]) - 1) // 2
            assert num_masks == (len(shifted_y[i]) - 1) / 2

            # 随机选择 mask embedding 索引（支持 shuffle）
            emb_inds = list(range(self.max_n_spans))
            if self.shuffle_mask_embedding:
                random.shuffle(emb_inds)
            emb_inds_use = emb_inds[:num_masks]
            emb_inds_use = emb_inds_use + emb_inds_use  # 前后各一个 mask
            mask_value.append(emb_inds_use)

            cur_inserted_y = []
            cur_mask_position = []
            for j in range(len(shifted_y[i]) - 1):
                cur_inserted_y.append(shifted_y[i][j])
                # 记录 mask 插入位置
                cur_mask_position.append(sum([item.shape[1] for item in cur_inserted_y]))
                # 用 EOG token 作为占位符
                cur_inserted_y.append(self.eog)

            cur_inserted_y.append(shifted_y[i][-1])
            inserted_y.append(cur_inserted_y)
            mask_position.append(cur_mask_position)

        return inserted_y, mask_position, mask_value

    # --------------------------------------------------------
    # 5. 拼接序列
    # --------------------------------------------------------
    def cat_y(self, inserted_y, mask_position, y_lens):
        """将每个样本的所有片段拼接为一个序列，并进行 padding
        
        Args:
            inserted_y: 插入 mask 后的各段
            mask_position: mask 位置
            y_lens: 原始长度
            
        Returns:
            cated_y: [K, T, B] 拼接并 padding 后的序列
            new_y_lens: [B] 每个样本的新长度
        """
        cated_y = []
        new_y_lens = []
        for i in range(len(inserted_y)):
            cur_cated_y = torch.cat(inserted_y[i], dim=1)  # [K, S]
            cur_cated_y = cur_cated_y.transpose(1, 0)  # [S, K]
            cur_cated_y_len = cur_cated_y.shape[0]
            new_y_lens.append(cur_cated_y_len)
            cated_y.append(cur_cated_y)

        cated_y = torch.nn.utils.rnn.pad_sequence(
            cated_y, batch_first=False, padding_value=self.audio_pad_token
        )
        cated_y = cated_y.permute(2, 0, 1)  # [T, B, K] -> [K, T, B]
        return cated_y, torch.LongTensor(new_y_lens).to(cated_y.device)

    # --------------------------------------------------------
    # 6. 嵌入音频 token 和 mask embedding
    # --------------------------------------------------------
    def embed_y(self, cated_y, mask_position, mask_value):
        """嵌入音频 token，并将 mask 位置替换为可学习的 mask embedding
        
        Args:
            cated_y: [K, T, B] 拼接后的 token 序列
            mask_position: 每个样本的 mask 位置
            mask_value: 每个 mask 的 embedding 索引
            
        Returns:
            embedded_y: [B, T, D] 嵌入后的序列
        """
        # 对每个码本分别嵌入，然后求和
        embedded_y = torch.stack(
            [self.audio_embedding[k](cated_y[k]) for k in range(self.n_codebooks)],
            dim=0
        )  # [K, T, B, D]
        embedded_y = embedded_y.sum(dim=0)  # [T, B, D]
        embedded_y = embedded_y.transpose(1, 0)  # [B, T, D]

        # 将 mask 位置替换为 mask embedding
        for i in range(len(embedded_y)):
            if len(mask_position[i]) > 0:
                embedded_y[i, mask_position[i]] = self.mask_embedding[mask_value[i]]

        return embedded_y

    # --------------------------------------------------------
    # 7. 训练时的完整输入目标准备
    # --------------------------------------------------------
    def prepare_input_target(self, y, y_lens):
        """训练时准备输入和目标（完整的掩码填充流程）
        
        完整流程：
        1. 采样掩码区间
        2. 重排序列
        3. 应用延迟模式
        4. 插入 mask token
        5. 拼接并嵌入
        6. 生成 attention mask 和 padding mask
        
        Args:
            y: [B, K, T] 音频 token 序列
            y_lens: [B] 每个样本的长度
            
        Returns:
            y_input: [B, S, D] 模型输入
            new_y_lens: [B] 新长度
            targets: 目标序列（用于计算 loss）
            y_padding_mask: padding mask
            y_attention_mask: 因果注意力 mask
            mask_position: mask 位置
            patterns: 模式对象（用于还原）
        """
        assert y.shape[1] == self.n_codebooks

        # Step 1: 采样掩码区间
        mask_intervals, non_mask_intervals = self.prepare_mask_intervals(y_lens)

        # Step 2: 重排序列（同时作为 targets）
        rearranged_y = self.rearrange(y, non_mask_intervals, mask_intervals)
        targets = rearranged_y

        # Step 3: 应用延迟模式
        shifted_y, patterns = self.shift(rearranged_y)

        # Step 4: 插入 mask token
        inserted_y, mask_position, mask_value = self.insert_mask(shifted_y)

        # Step 5: 拼接序列
        cated_y, new_y_lens = self.cat_y(inserted_y, mask_position, y_lens)

        # Step 6: 嵌入
        embedded_y = self.embed_y(cated_y, mask_position, mask_value)  # [B, T, D]

        # Step 7: 生成 mask
        y_padding_mask = make_pad_mask(new_y_lens).to(y.device)
        y_attention_mask = torch.triu(
            torch.ones(embedded_y.shape[1], embedded_y.shape[1]), diagonal=1
        ).bool().to(y_padding_mask.device)

        return embedded_y, new_y_lens, targets, y_padding_mask, y_attention_mask, mask_position, patterns

    # --------------------------------------------------------
    # 8. 移除 mask 并还原模式
    # --------------------------------------------------------
    def remove_mask(self, logits, mask_position, new_y_lens):
        """从模型输出 logits 中移除 mask token 位置
        
        Args:
            logits: [B, K, S, card] 模型输出 logits
            mask_position: mask 位置
            new_y_lens: 序列长度
            
        Returns:
            logits_use: list of list of tensors，每段的 logits
        """
        logits_use = []
        for i in range(len(logits)):
            non_mask_positions = [-1] + mask_position[i] + [new_y_lens[i]]
            non_mask_intervals = [
                [non_mask_positions[j] + 1, non_mask_positions[j + 1]]
                for j in range(len(non_mask_positions) - 1)
            ]
            cur_logits_use = [logits[i, :, l:r] for l, r in non_mask_intervals]
            logits_use.append(cur_logits_use)

        return logits_use

    def revert_pattern(self, patterns, logits_use):
        """将延迟模式的 logits 还原回原始并行排列
        
        Args:
            patterns: 各段对应的 Pattern 对象
            logits_use: 各段的 logits
            
        Returns:
            logits_final: 还原后的 logits
            logit_masks: 有效位置 mask
        """
        logits_final = []
        logit_masks = []
        for i in range(len(logits_use)):
            cur_logits = [
                item.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
                for item in logits_use[i]
            ]  # [1, card, K, S]
            cur_logits_final = [
                cur_pattern.revert_pattern_logits(
                    item, 0, keep_only_valid_steps=False
                )
                for cur_pattern, item in zip(patterns[i], cur_logits)
            ]
            cur_logits_final_ret = [
                item[0].permute(0, 2, 3, 1).squeeze(0)
                for item in cur_logits_final
            ]  # [K, T, card]
            logits_final.append(cur_logits_final_ret)
            logit_masks.append([item[2] for item in cur_logits_final])

        return logits_final, logit_masks


# ============================================================
# 辅助函数：Top-K / Top-P 采样
# ============================================================

def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """使用 top-k 和/或 nucleus (top-p) 过滤 logits 分布"""
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def topk_sampling(logits, top_k=10, top_p=1.0, temperature=1.0):
    """Top-K / Top-P 采样"""
    if temperature != 1.0:
        logits = logits / temperature
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
    return token


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    # 示例：演示掩码填充流程
    print("=" * 60)
    print("VoiceCraft 掩码填充（Mask Filling）示例")
    print("=" * 60)

    # 参数设置
    batch_size = 2
    n_codebooks = 4
    audio_vocab_size = 2048
    d_model = 128
    seq_len = 100

    # 创建模型
    model = VoiceCraftMaskFilling(
        n_codebooks=n_codebooks,
        audio_vocab_size=audio_vocab_size,
        d_model=d_model,
        max_n_spans=3,
        mask_sample_dist="uniform",
        mask_len_min=10,
        mask_len_max=30,
        min_gap=10,
    )

    # 模拟输入
    y = torch.randint(0, audio_vocab_size, (batch_size, n_codebooks, seq_len))
    y_lens = torch.tensor([seq_len, seq_len - 5])

    print(f"\n输入形状: y={y.shape}, y_lens={y_lens.tolist()}")

    # 执行掩码填充准备
    y_input, new_y_lens, targets, y_padding_mask, y_attention_mask, mask_position, patterns = \
        model.prepare_input_target(y, y_lens)

    print(f"输出形状: y_input={y_input.shape}")
    print(f"新序列长度: {new_y_lens.tolist()}")
    print(f"每个样本的 mask 数量: {[len(pos)//2 for pos in mask_position]}")
    print(f"mask 位置: {[pos.tolist() if isinstance(pos, torch.Tensor) else pos for pos in mask_position]}")

    print("\n" + "=" * 60)
    print("掩码填充流程说明:")
    print("=" * 60)
    print("1. prepare_mask_intervals: 随机采样掩码区间")
    print("2. rearrange: 重排序列 [非掩码段, 掩码段...]")
    print("3. shift: 应用 DelayedPattern 延迟模式")
    print("4. insert_mask: 插入 mask token 占位符")
    print("5. cat_y: 拼接并 padding")
    print("6. embed_y: 嵌入 token + mask embedding")
    print("7. 模型自回归生成掩码部分")
    print("8. remove_mask + revert_pattern: 还原序列")
