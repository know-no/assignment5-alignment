# SFT loss 的作用域、掩码归一化 + microbatch 训练步

> 锚点代码:
>
> - [cs336_alignment/utils/tokenizer_utils.py:21](../cs336_alignment/utils/tokenizer_utils.py#L21) `tokenize_prompt_and_output`(生产 `response_mask`)
> - [cs336_alignment/utils/compute_entropy.py:96](../cs336_alignment/utils/compute_entropy.py#L96) `masked_normalize`
> - [cs336_alignment/utils/sft_utils.py:14](../cs336_alignment/utils/sft_utils.py#L14) `sft_microbatch_train_step`
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

`response_mask` 排除 prompt 和 padding 位置的直接损失，`normalize_constant` 决定有效 token 的统一缩放。固定常数下，较长回答会贡献更多有效 token，其整条回答的损失仍可能更大。

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
| 整批有效 token 平均 | `(tensor * mask).sum() / mask.sum()` | 统一除以整批有效 token 总数 |
| 固定常数（本作业此处采用） | `(tensor * mask).sum() / normalize_constant` | 统一除以固定的 `C` |

### 为什么要用固定常数

对整个 `(B, L)` 张量使用 `mask.sum()`，会让这批所有有效 token 共享同一个分母。若先对每条回答除以各自长度、再对样本平均，短回答中单个 token 的系数会更大。这两种归一化沿不同的轴进行，需要结合实际 reduction 判断权重。

本作业此处使用固定的 `C`，每个有效 token 对 loss 的系数相同。回答越长，有效 token 通常越多，因此整条回答的总权重仍可能更大。下面所说的统一系数指 loss 对输入 log-prob 的导数，模型参数梯度还取决于上游计算图。

### 在 SFT pipeline 里的位置

```
per-token NLL  →  masked_normalize(.., dim=None) / batch_size / grad_accum_steps  →  backward()
```

- `normalize_constant` 对有效 token 的求和结果进行固定缩放。
- 除以 `batch_size` 对当前 microbatch 的样本取平均，再除以 `grad_accum_steps`，使等大小 microbatch 的累积梯度对应整个有效 batch 的平均损失。

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

对当前 microbatch，最大化响应 token 的对数似然，等价于最小化其负值。用于反传的标量损失为

$$
\mathcal{L}_{\text{micro}}(\theta) = -\frac{1}{BCK} \sum_{b=1}^{B} \sum_{t=1}^{L} m_{b,t}\log p_\theta(y_{b,t} \mid x_b, y_{b,<t})
$$

其中 $B$ 是当前 microbatch 的样本数，$C$ 是 `normalize_constant`，$K$ 是 `gradient_accumulation_steps`，$m$ 是 `response_mask`。上游 `get_response_log_probs` 已经计算并选出了标准答案 token 的 log-prob，当前函数负责取负、按 mask 求和、除以 $C$、对样本平均、除以 $K$ 并反传。

### 为什么交叉熵只剩负的 log-prob

多分类交叉熵（Categorical Cross-Entropy，CCE，通常简称 CE）为 $-\sum_i q_i\log p_i$。每个位置的标准答案给出一个 one-hot 目标分布 $q$，正确 token 对应项为 1，其余为 0，因此求和简化为 $-\log p_{\text{gold}}$。标签 ID 用来选取对应项。

正确 token 的概率从 `0.2` 提高到 `0.8` 时，$-\log p$ 从约 `1.609` 降到 `0.223`。负号使最小化损失与提高正确 token 的概率一致。BCE 是二分类的交叉熵，标签为 1 时同样得到 $-\log p$，标签为 0 时得到 $-\log(1-p)$。MSE（Mean Squared Error）则计算数值预测误差的平方平均，常用于回归任务。

### 用一个 microbatch 检查数值和梯度

设 `policy_log_probs = [[-1, -2], [-3, -4]]`，`response_mask = [[1, 1], [1, 0]]`，$C=1$、$B=2$、$K=2$。两条回答的负对数似然都是 `3`，样本平均 `unscaled_loss` 为 `3`，除以 $K$ 后返回的 `loss` 为 `1.5`。

对任一输入 log-prob 求导，得到 $\partial\mathcal{L}_{\text{micro}}/\partial\log p_{b,t}=-m_{b,t}/(BCK)$。有效位置的导数为 $-1/(2\times1\times2)=-0.25$，被遮住的位置为 `0`。例如把第一个值从 `-1` 提高到 `-0.99`，loss 会减少 `0.0025`。这些梯度随后由 autograd 沿上游计算图传播到模型参数。

### 为什么必须先除 `gradient_accumulation_steps` 再 backward

PyTorch 的 `.backward()` 将本次梯度累加到参数 `.grad`。

设各个 microbatch 大小相同，$L_k$ 已对第 $k$ 个 microbatch 的样本取平均。在各份损失计算一致的前提下，累积梯度满足

$$
\nabla \Big( \frac{1}{K}\sum_k L_k \Big) = \sum_k \nabla \Big( \frac{L_k}{K} \Big)
$$

因此，这个函数在 `backward()` 前除以 `gradient_accumulation_steps`。在反传后仅对返回的 loss 再做除法，不会改变已经累积的梯度。若各 microbatch 样本数不同，需要按样本数加权才能对应整个有效 batch 的样本平均。

### 参考实现

```python
def sft_microbatch_train_step(
    policy_log_probs,     # (B, L) 每个位置真实 token 的 log p_θ
    response_mask,        # (B, L) 1=response, 0=prompt/pad
    gradient_accumulation_steps,
    normalize_constant=1.0,
):
    batch_size = policy_log_probs.shape[0]
    # 1) 取负得到 NLL，按 mask 求和并除以固定常数 C。
    neg_log_lik = -masked_normalize(
        policy_log_probs, response_mask,
        normalize_constant=normalize_constant, dim=None,
    )
    # 2) 对 B 条样本平均，得到除 K 之前用于日志的 microbatch loss。
    unscaled_loss = neg_log_lik / batch_size
    # 3) 除以 K 后反传，将本次梯度累加到参数 .grad。
    loss = unscaled_loss / gradient_accumulation_steps
    loss.backward()
    metadata = {
        "unscaled_loss": unscaled_loss.detach(),
        "n_response_tokens": response_mask.sum().detach(),
    }
    return loss, metadata
```

`unscaled_loss` 已包含负号、$C$ 归一化和样本平均，表示除以 $K$ 前的损失。`.detach()` 保留日志数值并切断它与计算图的连接。`loss` 则是本次已经用于反传、包含 $1/K$ 缩放的值。

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
| 忘记加负号 | loss 和梯度符号反转，优化方向降低标准答案的概率 | `-masked_normalize(...)` |
| 混淆固定常数和有效 token 平均 | 改变整批 loss 的分母和梯度尺度 | 按此函数约定使用固定 `normalize_constant`（见 §1） |
| 反传后只对返回 loss 除以 K | 参数中已经累积的梯度仍大 K 倍 | 函数内先 `loss / K` 再 `backward()` |
| microbatch 循环内调 `optimizer.step()` | 退化成 K 次小 batch 更新,训练动态完全变 | `step()` 只在 K 次累加完调一次 |

### 快照更新与普通验证

`UPDATE_SNAPSHOTS` 由仓库的 `NumpySnapshot.assert_match` 读取。值为 `1` 时，当前输出数组会写入对应 `.npz` 文件，随后返回，跳过与旧快照的数值比较。显式设置 `force_update=True` 或快照文件缺失时，也会写入当前输出。它用于在确认新的预期结果后刷新参考数据。若当前实现有误，更新会把错误输出存成后续比较的参考值。[tests/conftest.py:71](../tests/conftest.py#L71)

这个开关在本仓库中有明确的本机精度适配用途。2026-06-30 的提交 `37b5cfb` 写明它用于“方便在 Mac (bf16) 上重新生成基线”，同次提交实现了 `get_response_log_probs` 并讨论了显式 FP32 加载。模型前向使用 BF16 时，输出已经包含该精度下的舍入误差，随后对 logits 调用 `.float()` 会保留这些误差，并让后续计算使用 FP32。

2026-11-13 在当前 Mac 的 CPU 上，使用相同模型、测试输入、SDPA 后端以及 `torch 2.5.1`、`transformers 5.12.1`，对照官方 `test_get_response_log_probs.npz` 得到

| 模型前向精度 | log-prob 最大绝对偏差 | entropy 最大绝对偏差 | 原快照比较 |
|---|---:|---:|---|
| 默认 BF16 | `0.239779` | `0.167585` | 失败 |
| 显式 FP32 | `0.00009537` | `0.00072396` | 通过 |

本次 CPU 对照中，显式 FP32 前向后通过了原快照比较。当前 `tests/conftest.py` 的 model fixture 已显式使用 `dtype=torch.float32`，下方命令按此配置重建 FP32 本地参考。若要建立 BF16 基线，需要先将该 fixture 的加载精度显式改为 `dtype=torch.bfloat16`，独立确认数值正确后，再仅对 `test_get_response_log_probs` 更新，并记录模型、dtype 和运行环境。

按当前 FP32 配置重建这项测试的本地基线时，运行

```bash
UPDATE_SNAPSHOTS=1 .venv/bin/python -m pytest tests/test_sft.py::test_get_response_log_probs -q
```

该命令会覆盖 `tests/_snapshots/test_get_response_log_probs.npz`，并跳过与旧参考结果的数值比较。随后保持同一模型前向配置，移除更新开关，运行同一测试，检查新本地基线能否复现

```bash
env -u UPDATE_SNAPSHOTS .venv/bin/python -m pytest tests/test_sft.py::test_get_response_log_probs -q
```

更新后通过快照比较，表示输出与该本地基线一致。基线的数值正确性由前面的独立核对建立，常规测试使用已确认的参考结果进行比较。

验证这个训练步时，使用已有快照比较 loss 和输入 log-prob 的梯度。这三个测试直接构造输入张量，可以运行

```bash
env -u UPDATE_SNAPSHOTS .venv/bin/python -m pytest \
  tests/test_sft.py -k sft_microbatch_train_step -q
```

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
