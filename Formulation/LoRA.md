# LoRA

## 什么是 LoRA？

**LoRA 微调（Low-Rank Adaptation，低秩适应）** 是一种参数高效微调方法：不更新预训练模型的大矩阵权重，而是冻结原模型，只在部分层旁边注入一对小的低秩矩阵来学习“权重增量”。这样可训练参数极少，显存和存储成本大幅下降，适合把大模型适配到新任务、新风格或新数据。

### 数学原理

假设某层原权重是 $W_0 \in \mathbb{R}^{d \times k}$。全量微调会直接更新 $W_0$，参数量很大。LoRA 改为：

$$
W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} BA
$$

其中：
- $W_0$ 冻结，不训练；
- $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，是可训练的低秩矩阵；
- 秩 $r \ll \min(d,k)$，所以 $BA$ 参数量远小于 $W_0$；
- $\alpha$ 是缩放超参数，常按 $\alpha/r$ 缩放。

*注：线性代数中的 低秩分解。*

前向传播变成：

$$
h = W_0 x + \frac{\alpha}{r} BAx
$$

初始化时，$A$ 用随机高斯，$B$ 全零，因此训练开始时 $\Delta W=0$，模型行为与原模型一致，训练稳定。

- **训练时**：只更新 $A$、$B$，原模型权重冻结。优化器状态、梯度、显存占用都大幅减少。
- **推理时**：可以把 $BA$ 合并回原权重：
  $$
  W' = W_0 + \frac{\alpha}{r}BA
  $$
  合并后推理速度与原来几乎一样，没有额外延迟。
- **多任务**：基座模型共享，每个任务只保存一个很小的 LoRA adapter，可插拔切换。

### 主要优点

1. **显存低**：可训练参数常降到原模型的 0.01%～1%。
2. **存储小**：一个 adapter 可能只有几 MB 到几百 MB。
3. **不易灾难性遗忘**：原模型能力基本保留。
4. **部署灵活**：可合并，也可动态加载多个 adapter。
5. **适合大模型**：LLM、扩散模型、语音模型、TTS 都常用。

### 缺点与注意点

- 秩 $r$ 需要调：太小欠拟合，太大接近全量微调、失去省参优势。
- 容量有限，某些复杂任务可能不如全量微调。
- 通常加在注意力层的 Q/K/V/O 投影或 FFN 线性层上，具体加哪些层要实验。
- 多个 LoRA 合并或叠加时可能相互干扰。
- 如果基座模型也更新，就不是纯 LoRA 了。



如果要把 AuK 适配到自己的编辑任务，典型做法是：冻结 MLLM、VAE 和大部分 Transformer 骨干，只在骨干的注意力/FFN 线性层加 LoRA，用你的“指令 + 输入音频 + 目标音频”数据训练。这样只需少量 GPU 显存，就能让 AuK 学会新的编辑风格或领域数据，同时保留原有生成、编辑、增强和分离能力。


## LoRA in AuK：

$$
W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} BA
$$

选取了 FFN 和 注意力注入。

>   FFN in AuK:

    FFN（Feed-Forward Network），前馈神经网络。

### 流程

1. 准备数据集：

- （筛选，去除语气词等简短语句）
- ASR 撰写，结果转录，生成输入对

> 为什么需要 ASR？训练不是无转录的吗？

    没有给参考音频使用 ASR。

2. 使用 LoRA 脚本进行训练

3. 权重保存与合并




### 代码

- `lora.py` ：定义 LoRA
- `lora_train.py` ：训练
- `merge_lora.py` ：合并权重

> Flux2Edit？

    AuK backbone 的具体网络架构名称，源自 Diffusion Transformer (DiT) 家族，专为音频编辑设计。


### lora.py

```python
class LoRALinear(nn.Linear):
    def __init__(self, in_features, out_features, r=32, alpha=32, ...):
        super().__init__(in_features, out_features, bias=bias)
        self.register_buffer("lora_scaling", torch.tensor(alpha / r, dtype=torch.float32))
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.weight.requires_grad_(False)   # 冻结原权重
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)          # B=0 → 初始等效原权重

    def forward(self, x):
        out = super().forward(x)             # 冻结的 W x + b
        out = out + (x @ self.lora_A.t() @ self.lora_B.t()) * self.lora_scaling
        return out
```

### lora_train.py

> EMA ？

    EMA（Exponential Moving Average，指数移动平均）。对历史权重的平滑平均，泛化性能通常优于当前权重。

    音色微调对 EMA 收益不大，微调靠 LoRA 增量。

### merge_lora.py

### debug

> ZeRO-3 是什么？

    ZeRO-3（Zero Redundancy Optimizer stage 3），DeepSpeed 提出的分布式训练优化技术。

    核心思想：分片存储（sharding）——把同一个参数张量切分成 N 份，每块 GPU 只存自己的那份，而不是像 DDP 那样所有 GPU 都存全量权重。

    不用这个根本塞不到这几个超算卡里面，目测以后还会用到。
    

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| 1 | `mat1 and mat2 dtype mismatch (Float vs BFloat16)` | ZeRO-3 把 Linear 换成 `LinearFunctionForZeroStage3`，直接 `addmm` 不做 autocast 类型提升；VAE latent/timestep 是 fp32，权重是 bf16 | [cfm_edit.py](file:///e:/IcekyPrime/AIspeech/Code/AuK/src/auk/model/cfm_edit.py) `forward` 入口把 `inp/ref_latent/x0/time/text_embeds` 显式 `.to(backbone dtype)` |
| 2 | `CUDA OOM`（23.57G 全满） | ZeRO-3 只分片权重、不分片激活；32 层 transformer 前向激活常驻 | `checkpoint_activations=True` |
| 3 | `Invalid mt19937 state` → NCCL 600s 超时 → SIGABRT | accelerate 每 epoch 重迭代 DataLoader 时广播 RNG 状态，多卡+ZeRO-3 下偶尔得到非法 mt19937 状态，部分 rank 崩、其余等 collective 超时 | `rng_types=[]` 关闭 RNG 同步（不影响收敛） |
| 4 | rank 2 领先其他 rank ~58 个 collective（`_ALLGATHER_BASE` vs `ALLREDUCE` 错位） | `even_batches=False`，`len(dataset)` 不能被 `num_proc×bsz` 整除时各 rank batch 数差 1 | `even_batches=True` |
| 5 | 每 epoch 只有 14 update，`total_updates=1880` 虚高，lr 永远不 decay | `total_updates` 用 prepare 前的单机 batch 数，没除 `num_processes`；`warmup_updates` 还多乘了 7 | 改为 `ceil(len / grad_acc / num_processes) × epochs`，warmup 用 `warmup_steps` |
| 6 | val 采样阶段 rank 0 卡 `_ALLGATHER_BASE`，其余 rank 空闲 | `synthesize_and_save` 开头 `if not is_main_process: return`，只有 rank 0 跑 `sample()`，但 ZeRO-3 前向是跨 rank collective | 重写为所有 rank 跑前向，仅 main 保存音频，且恢复 `model.train()` |
| 7 | 训练结束后保存时 rank 0 卡 gather（NumelIn=6） | `save_checkpoint` 里 `get_state_dict` 放在 `is_main_process` 之后，只有 rank 0 参与 gather | 把 `get_state_dict` 移到判断前，所有 rank 执行 |
| 8 | 推理 OOM（单卡 24G） | 推理端 `model.to(torch.float32)` 把 Qwen(3B)+DiT(1.5B)+VAE 全 fp32 常驻单卡 | 加 `--cpu_offload`（Qwen/DiT 动态 CPU↔GPU） |
| 9 | 生成语速偏快 | 未指定时长时 `get_gen_duration` 退化为「参考音频时长」，短参考音频把长文本压进短时长 | `--gen_seconds` 显式对齐，或 `--ref_text`+`--gen_text` 按字数比例估算 |



> 前 7 个 bug 的共性根因是 **ZeRO-3 改变了三个默认假设**：
> 1. **类型**：ZeRO-3 线性层绕过 autocast，需手动对齐 dtype（bug #1）。
> 2. **collective 语义**：ZeRO-3 的 `get_state_dict`、`sample()`、前向   gather 都是**跨 rank collective**，任何「只有 main process 执行」的分支都会死锁（bug #6、#7）。
> 3. **batch 对齐**：多卡下 batch 数必须严格一致（bug #4），且 schedule 的步数要除以进程数（bug #5）。

> 在 ZeRO-3 + 多卡下，凡是涉及「主进程单独做某件事」的写法（保存、采样、RNG 同步），都要先确认该操作是不是 collective，是的话必须所有 rank 同步参与。

### 对比结果