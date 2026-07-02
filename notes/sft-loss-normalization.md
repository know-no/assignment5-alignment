# SFT loss 的作用域、掩码归一化 + microbatch 训练步

> 锚点代码:
>
> - [cs336_alignment/utils/tokenizer_utils.py:21](../cs336_alignment/utils/tokenizer_utils.py#L21) `tokenize_prompt_and_output`(生产 `response_mask`)
> - [cs336_alignment/utils/compute_entropy.py:94](../cs336_alignment/utils/compute_entropy.py#L94) `masked_normalize`
> - [cs336_alignment/utils/sft_utils.py:5](../cs336_alignment/utils/sft_utils.py#L5) `sft_microbatch_train_step`
>
> 主线:**作用域**(§0 谁进 loss)→ **尺度**(§1 用什么归一化)→ **形状**(§2 `dim=None` 语义)→ **反传**(§3 microbatch + 梯度累积)。这一篇覆盖 SFT 从 mask 到 backward 的整条链路。

## 0. 作用域:SFT 只对 response token 求 loss,预训练对全部 token 求

### 目标函数的差别

预训练把整段文本当成"想让模型学的分布":

$$
\mathcal{L}_{\text{PT}} = -\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})
$$

**每个位置都是有效监督信号**。

SFT 只想学条件分布 $p_\theta(y \mid x)$——给定 prompt,生成什么样的 response:

$$
\mathcal{L}_{\text{SFT}} = -\sum_{t \in \text{response}} \log p_\theta(y_t \mid x, y_{<t})
$$

**prompt 位置的 token 概率不算 loss**,由 `response_mask` 挡掉。

### 如果 SFT 也算 prompt 会怎样

三个具体问题:

1. **目标函数与实际使用不匹配**:prompt 是外部给的,推理时不需要模型"生成 prompt"。让 loss 覆盖 prompt = 强迫模型学一个你根本不 care 的分布,浪费容量。
2. **信号稀释**:prompt 常常远长于 response(GSM8K/MATH 里几百 token vs 几十 token),loss 平均到全部 token 会把真正想学的 response 梯度权重摊薄。
3. **过拟合模板**:prompt 里有大量固定套路(`"You are a helpful assistant..."`),让模型花梯度记这些固定串,切换模板时鲁棒性变差。

### 作用域 vs 尺度:两个正交的设计

`masked_normalize` 里两个参数刚好对应两个不同问题:

| 参数 | 处理什么 | 表现 | 见 |
| --- | --- | --- | --- |
| `mask`(`response_mask`) | **作用域** —— 谁进 loss | prompt 长时 response 信号是否被稀释 | 本节 |
| `normalize_constant` | **尺度** —— 怎么归一化 | 响应长度不同时单 token 权重是否公平 | §1 |

两者叠加,才让 SFT 的 loss 对"prompt 长短"和"response 长短"都不敏感。

### 代码落点

```python
# tokenize_prompt_and_output 里,mask 默认全 0,只在 response 位置置 1
mask = [0 for _ in range(max_len - 1)]
for j in range(output_start_idx - 1, output_start_idx - 1 + output_size):
    mask[j] = 1
```

配合 `masked_normalize(policy_log_probs, response_mask, normalize_constant=C, dim=None)`,就同时解决了作用域和尺度两个问题。

### 反直觉例外:prefix LM / instruction pre-training

有些工作(UL2、FLAN 的一部分)**故意也算 prompt 的 loss**——这不是 SFT 而是**继续预训练**:目标是保留通用语言建模能力,顺便见到指令数据。判断依据是**目标**:

- "让模型学会条件生成" → mask prompt(SFT)
- "让模型继续学语言分布,顺便见指令" → 不 mask(instruction pre-training / prefix LM)

## 1. `normalize_constant` 是固定常数,不是 `mask.sum()`

### 两种归一化对比

| 方式 | 表达式 | 单 token 梯度大小 |
|---|---|---|
| per-token mean(常见但有偏) | `(tensor * mask).sum() / mask.sum()` | **取决于该样本有效 token 数** |
| 固定常数(Dr.GRPO / 本作业要求) | `(tensor * mask).sum() / normalize_constant` | **与样本长度无关** |

### 为什么要用固定常数

用 `mask.sum()` 当分母时,长序列里每个 token 的贡献被稀释——分母大、单 token 梯度小,模型实际上隐式地**更看重短序列的 token**。这就是 Dr.GRPO 等工作指出的**长度偏差(length bias)**。

固定常数(典型取 `max_response_length`,或干脆 `1.0`)让每个 token 贡献的梯度大小**不依赖它所属序列的长度**,行为更可控、和"每个 token 一票"的直觉一致。

### 在 SFT pipeline 里的位置

```
per-token NLL  →  masked_normalize(.., dim=None) / batch_size / grad_accum_steps  →  backward()
```

- `normalize_constant` 处理**序列长度方向**的偏差
- 再除以 `batch_size`、`grad_accum_steps` 控制**梯度噪声大小**和**有效 batch size**

## 2. `dim=None` 的语义:为什么不是"只有最后一维有数据"

### 容易出错的直觉

> "tensor 形状是 `(B, L)`,batch 维只是索引,数据都在 L 那一维,所以 `dim=None` 全维度求和 ≈ `dim=-1` 求和?"

**错。** 每个 `(b, l)` 位置上都是一个**真实的标量损失**(第 b 条样本第 l 个 token 的 -log_prob)。batch 维不是元信息,它的每个 slice 都是完整序列。

### 形状对照

假设 `tensor.shape = (B, L) = (4, 100)`,mask 里共 250 个 1:

```python
masked_normalize(tensor, mask, C, dim=None)
# (tensor * mask).sum() → 标量,把全部 4×100=400 位置里 mask=1 的 250 个值加起来
# / C → 标量
# 用法:整 batch 的 total loss(配 batch_size 一起做平均)

masked_normalize(tensor, mask, C, dim=-1)
# (tensor * mask).sum(dim=-1) → (B,) = (4,),每条样本一个 sum
# / C → (4,)
# 用法:逐样本 loss(后续可能要按样本加权、reward 加权、再做 reduction)
```

### 选哪个

- **SFT 简单形态**:`dim=None`,直接给优化器一个标量
- **逐样本加权**(RLHF / DPO / 重要性采样 / 长度归一化研究):`dim=-1`,先得到 `(B,)`,再决定怎么 reduce

## 3. `sft_microbatch_train_step`:从 log_probs 到 backward 的完整一步

### SFT 目标函数

最大化响应 token 的对数似然,等价于最小化其负值:

$$
\mathcal{L}_{\text{SFT}}(\theta) = -\frac{1}{C} \sum_{(x, y)} \sum_{t \in \text{response}} \log p_\theta(y_t \mid x, y_{<t})
$$

其中 $C$ 就是 `normalize_constant`。上游 `get_response_log_probs` 已经把 `log p_θ(y_t | ...)` 算好,microbatch step 负责**加负号、按 mask 聚合、除梯度累积步数、反传**。

### 为什么必须先除 `gradient_accumulation_steps` 再 backward

**核心事实**:PyTorch 里 `.backward()` 是**累加**梯度到 `.grad`,不是覆盖。

**关键恒等式**:对 K 个 microbatch 累积等价于对 K 份拼成的大 batch 做一次 `mean().backward()`:

$$
\nabla \Big( \frac{1}{K}\sum_k L_k \Big) = \sum_k \nabla \Big( \frac{L_k}{K} \Big)
$$

所以在函数**内部**先除以 `gradient_accumulation_steps` 再反传;若在外层再除,梯度尺度会错 K 倍。

### 参考实现

```python
def sft_microbatch_train_step(
    policy_log_probs,     # (B, L) 每个位置真实 token 的 log p_θ
    response_mask,        # (B, L) 1=response, 0=prompt/pad
    gradient_accumulation_steps,
    normalize_constant=1.0,
):
    # 1) per-token NLL,mask 求和,除常数 → 标量
    neg_log_lik = -masked_normalize(
        policy_log_probs, response_mask,
        normalize_constant=normalize_constant, dim=None,
    )
    # 2) 除梯度累积步数,让 K 个 microbatch 累加 == 大 batch mean
    loss = neg_log_lik / gradient_accumulation_steps
    # 3) 就地反传,grad 累加到参数 .grad
    loss.backward()
    metadata = {
        "unscaled_loss": neg_log_lik.detach(),
        "n_response_tokens": response_mask.sum().detach(),
    }
    return loss, metadata
```

### 外层训练循环模板

```python
optimizer.zero_grad()
for k in range(gradient_accumulation_steps):
    mb = next(data_iter)
    out = get_response_log_probs(model, mb.input_ids, mb.labels)
    loss, meta = sft_microbatch_train_step(
        out["log_probs"], mb.response_mask,
        gradient_accumulation_steps=gradient_accumulation_steps,
        normalize_constant=normalize_constant,
    )   # 函数内已 backward,不要重复调
optimizer.step()      # 只在累完 K 次后 step 一次
optimizer.zero_grad()
```

### 容易踩的四个坑

| 坑 | 现象 | 正确做法 |
| --- | --- | --- |
| 忘记加负号 | loss 一直为负、越训越负 | `-masked_normalize(...)` |
| 用 `mask.sum()` 当分母 | 长响应权重被稀释,长度偏差 | 用固定 `normalize_constant`(见 §1) |
| 在函数外再除以 K | 梯度尺度错 K 倍,不等价于大 batch | 函数内 `loss / K` 后 backward |
| microbatch 循环内调 `optimizer.step()` | 退化成 K 次小 batch 更新,训练动态完全变 | `step()` 只在 K 次累加完调一次 |

## 外延拓展方向

按主题分组,可任选切入。已读完在前面打勾。

### A. Loss 归一化与长度偏差(§1 的延伸)

- [ ] **Dr.GRPO 论文** `Understanding R1-Zero-Like Training` (Liu et al. 2025):重点看它对 GRPO 长度偏差和"标准差归一化偏差"的剖析——为什么 GRPO 原版用 `1/|response|` 分母和 group-relative std 分母都会引入长度偏好。这是本作业选 fixed `normalize_constant` 的直接依据。
- [ ] **DPO / IPO / KTO 的 loss 归一化**:对比它们处理"每条样本长度不同"的方式,理解 length-normalized log prob vs sum log prob 的取舍。DPO 原文默认 sum,SimPO 改为 length-normalized 并给出理由。
- [ ] **PPO/GRPO 里 token-level vs sequence-level 优势**:同样是"每 token 一票 vs 每序列一票"。看 HuggingFace TRL 的 `PPOTrainer` / verl / OpenRLHF 的 loss 实现。
- [ ] **长度偏差的实证现象**:训练时长响应被隐式惩罚 / 模型生成越来越短(reward hacking 的一种)。可对照 Open-RLHF、verl、TRL 的相关 issue。
- [ ] **`F.cross_entropy(reduction=...)`** 与 `masked_normalize` 的对应关系:`"sum"` / `"mean"` / `"none"` 本质就是这里 `dim=None` 配不同分母的特例;`ignore_index` 就是内建版本的 mask。
- [ ] **PyTorch reduction 的形状规则**:`sum(dim=None)` / `sum(dim=-1)` / `sum(dim=(0,1))` 三种写法结果差别;`keepdim=True` 什么时候必须要(比如广播归一化)。

### B. Autograd 与梯度累积机制(§3 的延伸)

- [ ] **`loss.backward()` 累加而非覆盖**:PyTorch autograd 文档里 `Tensor.backward` 与 `torch.autograd.grad` 的语义差异;`retain_graph=True` / `create_graph=True` 的场景;为什么每个 `optimizer.step()` 前需要 `zero_grad(set_to_none=True)`。
- [ ] **梯度累积的等价性到底成立到什么程度**:BatchNorm / dropout / weight decay(尤其 AdamW 的 decoupled decay)/ 学习率 warmup 计数——哪些操作让"K 个 microbatch 累积" ≠ "1 个大 batch",哪些完全等价。
- [ ] **梯度累积 vs DDP `no_sync`**:分布式训练里前 K-1 次 `no_sync()` 上下文内不做 all-reduce、最后一次才同步,省掉 K-1 次通信。看 PyTorch DDP 文档和 accelerate 的 `accumulate()` context。
- [ ] **梯度裁剪(`clip_grad_norm_`)在梯度累积里的正确位置**:必须在所有 `backward()` 完成后、`optimizer.step()` 之前调用;裁剪对象是累加后的总梯度,而不是每个 microbatch 各裁一次。
- [ ] **AMP / bf16 与梯度累积的交互**:`GradScaler` 只在 fp16 需要,bf16 通常关掉;累积多步时 scaler 的 `unscale_` / `step` 时机;bf16 累积时数值稳定性对 `normalize_constant` 尺度的敏感度。
- [ ] **有效 batch size 的三个杠杆**:`per_device_batch × grad_accum_steps × world_size`。同一有效 batch 下,增大 grad_accum 换显存但增加计算时间;学习率 scaling law(linear / sqrt)如何跟随。

### C. 本作业上下文(动手验证)

- [ ] **grep 所有 caller**:在 [cs336_alignment/](../cs336_alignment/) 里 `codegraph_explore` 一下 `masked_normalize` 和 `sft_microbatch_train_step` 的所有调用点,确认 SFT 与 GRPO 两条路径里的具体形状和 `normalize_constant` 取值,验证正文理解。
- [ ] **对拍 `F.cross_entropy`**:用一个小样本手写 `F.cross_entropy(logits, labels, reduction='none') * response_mask` 然后 sum,对比 `-masked_normalize(policy_log_probs, response_mask, 1.0)`,数值应完全一致。
- [ ] **构造长度偏差实验**:同一 batch 里放一条 10-token 响应和一条 100-token 响应,分别用 `mask.sum()` 归一化和固定常数归一化,观察两条样本对同一参数的梯度范数比。这是把 §1 的抽象结论变成手感的最快方式。
