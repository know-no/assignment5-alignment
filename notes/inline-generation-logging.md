# 训练循环内的采样日志:`log_generations` 的设计

> 锚点代码:[cs336_alignment/utils/log_utils.py](../cs336_alignment/utils/log_utils.py) `log_generations`
> 主题:在 SFT/RL 训练里,定期让模型对一批 prompt 生成响应,打分、算熵、按正误分桶——这份 note 讲这一步为什么值得独立成模块,以及三个关键设计选择。

## 为什么要有循环内采样日志

训练时的 loss/grad_norm 只告诉你**优化在进行**,不告诉你**模型在说什么**。

- SFT loss 一直在降但模型开始复读 → loss 看不出来
- GRPO reward 均值涨了但格式全崩 → reward 看不出细节
- Reward hacking(比如响应越写越短)只有看到样本才能识别

一个稳定的循环内采样日志能同时回答:
- **样本级**:这条 prompt 具体生成了什么、拿到多少奖励、熵多高
- **聚合级**:整批的平均 reward、平均长度、正确响应 vs 错误响应的长度差

后者是**长度偏差、reward hacking、模式崩塌**的早期信号。

## 三个关键设计选择

### 1. 生成路径解耦:`generate_fn` 作为回调

```python
GenerateFn = Callable[[list[str]], list[str]]
```

不直接接 `LLM` 或 `PreTrainedModel`,而是接一个"prompts 进、responses 出"的函数。vllm 的 `llm.generate(...)` 和 HF 的 `model.generate(...)` 都能包成同一签名。

好处:

- **同一日志模块给 SFT 和 GRPO 两条路径复用** —— SFT 里可能是 HF 采样,GRPO 里通常是 vllm
- **单测不需要真加载模型** —— 传个返回固定字符串的桩函数就能测
- **生成参数与日志逻辑正交** —— temperature/top_p/max_tokens 由调用方在 `generate_fn` 里封装

调用回调时，日志函数会进入 `torch.no_grad()`，并将传入的 `policy_model` 临时设为 eval。正常返回或抛出异常后，各子模块的 `training` 标志都恢复到调用前的值。若回调使用独立的 vLLM 实例或另一个 HF 模型，回调负责那个模型自己的推理状态。

### 2. 熵按 `mask.sum(-1)` 逐样本平均,而不是训练用的固定常数

这是 [sft-loss-normalization.md](sft-loss-normalization.md) 的对照场景。**同一个 mask+entropy 张量,两处用途、两种归一化**:

| 场景 | 分母 | 目的 |
| --- | --- | --- |
| 训练 loss(§1) | 固定 `normalize_constant`，再对样本平均并缩放梯度累积 | 有效 token 采用统一系数，较长回答仍可能贡献更多总损失 |
| **日志展示(本篇)** | **`mask.sum(-1)`(每条真实响应长度)** | **给出可比较的 nats/token,人眼可读** |

关键洞察:**归一化选谁,取决于你要什么样的可比性**:

- 训练要"每 token 权重一致"——所以除固定常数
- 日志要"每条样本的平均不确定度"——所以除各自长度

令 `R` 为具有响应 token 的行数，这些行一起送入模型。逐行聚合的计算为

```python
# token_entropy 与 mask 的形状均为 (R, L)，每行至少有一个响应 token。
per_sample_sum = (token_entropy * mask).sum(dim=-1)
per_sample_count = mask.sum(dim=-1)
per_sample_mean = per_sample_sum / per_sample_count  # (R,)，单位为 nats/token
```

这里 `dim=-1` 是精确匹配 [sft-loss-normalization.md](sft-loss-normalization.md) §2 里"逐样本 loss"那条路径的实战案例。

没有响应 token 的样本，其 `mean_token_entropy` 为 `None`，整批平均熵排除这些空值。整批响应都为空时，熵 helper 直接返回 `None` 列表并跳过模型前向。其余响应保持整批前向的计算方式，结果按原输入顺序返回。

### 3. 长度按 token 数,分桶按 `answer_reward`

传入 `tokenizer` 时，函数使用 `tokenizer.encode(response, add_special_tokens=False)` 统计响应字符串的 token 数。正式记录 token 长度时，调用者应传入与评分模型匹配的 tokenizer。省略 tokenizer 时，现有 API 保留 `len(response.split())` 作为近似长度，并继续存入 `response_len_tokens` 字段。

正确响应按 `answer_reward == 1` 分桶。当前 `r1_zero_reward_fn` 的返回规则为

| 响应情况 | `format_reward` | `answer_reward` | `reward` |
|---|---:|---:|---:|
| 格式正确且答案正确 | 1 | 1 | 1 |
| 格式正确且答案错误 | 1 | 0 | 0 |
| 格式错误 | 0 | 0 | 0 |

因此，这个 grader 中 `answer_reward` 与 `reward` 始终相同，格式错误也进入失败桶。即使响应字符串写出了正确数字，格式错误时这两个奖励仍为 0。`mean_correct_len` 和 `mean_wrong_len` 按当前奖励函数认可的成功与失败统计。[drgrpo_grader.py:1008](../cs336_alignment/drgrpo_grader.py#L1008)

## 记录什么、聚合什么

**每条 record**(适合事后 grep / 用 pandas 分析):

- `prompt` / `response` / `ground_truth`
- `format_reward` / `answer_reward` / `reward`
- `response_len_tokens`
- `mean_token_entropy`，未启用熵计算或响应没有 token 时为 `None`
- `step`

**每批 summary**(适合直接扔进 wandb / print):

- `n`, `step`
- `mean_reward` / `mean_format_reward` / `mean_answer_reward`
- `mean_response_len` / `mean_correct_len` / `mean_wrong_len`
- `mean_token_entropy`

JSONL 以追加方式保存 `records`，每条记录一行。`summary` 返回给调用者，用于 wandb 或其他聚合指标记录。

`extra` 可以附加 `epoch`、`lr` 等字段。若键名与 `n`、`step` 或任一核心均值字段冲突，函数会在生成和写日志前抛出 `ValueError`。空批次仍返回完整的核心 summary 字段，`n=0`，各均值为 `None`，并保留不冲突的 `extra`。

## 训练循环里的位置

```python
if global_step % log_every == 0:
    log = log_generations(
        prompts=val_prompts, ground_truths=val_gts,
        generate_fn=sample_with_vllm, reward_fn=r1_zero_reward_fn,
        policy_model=policy, tokenizer=tokenizer,
        log_path="outputs/gen_log.jsonl", step=global_step,
    )
    wandb.log(log["summary"], step=global_step)
```

采样频率的取舍:

- 太密:vllm 采样 + reward 计算不便宜,吃训练时间
- 太稀:看不到 hacking / 崩塌的早期信号
- 经验值:SFT 每 200~500 步一次,GRPO 每次评估轮(通常就是 rollout batch 边界)一次

## 需要留意的三个坑

1. 生成与熵前向都在 `policy_model.eval()` 和 `torch.no_grad()` 下运行。进入时保存所有子模块的 `training` 标志，退出时逐个恢复，因此原先处于 eval 的子模块也能保持自己的状态。
2. 某批可能全对或全错，对应空桶由 `_safe_mean` 返回 `None`。空响应的熵也为 `None`，其余有效样本照常参与整批熵均值。
3. 长度统计以回调返回的响应字符串为准。HF 回调应从生成序列中取出响应部分，特殊 token 的保留方式需要与日志口径一致。已从字符串中移除的 EOS 无法由重新编码自动恢复。

`compute_entropy.py` 已在 shifted 域使用 `shifted - log(sum_exp)` 计算 log-prob，修复 FP32 下共同大偏移引起的舍入问题。`test_compute_entropy_large_common_offset` 用 `[1e8, 1e8, 1e8]` 验证熵为 `log(3)`，当前日志调用复用这个实现。

## 外延拓展方向

### A. 与训练指标的对照

- [ ] **response length 随训练的走势**:SFT 中通常先降再稳,RL 中可能上升(思考链变长)或崩塌(hacking 到短响应)。画 `mean_correct_len` vs `mean_wrong_len` 差值曲线是判断 hacking 的常用手段。
- [ ] **平均 token 熵作为探索度指标**:RL 训练中熵持续下降到接近 0 → 策略退化 / mode collapse;PPO/GRPO 里的 entropy bonus 就是干预这个信号。
- [ ] **format_reward vs answer_reward 的分解**:两条曲线的时间差反映"模型先学会格式还是先学会答题"。r1_zero 里前者应该几步就到 1,后者慢慢爬。

### B. 采样与评估基础设施

- [ ] **vllm 的 `SamplingParams`**:temperature/top_p/top_k/`n`(rollouts per prompt)/`stop`。RL 里 `n>1` 就是 GRPO 的 group;评估时通常 `n=1, temperature=0` 做 greedy。
- [ ] **HF `model.generate` 的坑**:`pad_token_id` 未设 → warning + 结果错位;`max_new_tokens` vs `max_length` 语义不同;beam search 与 sampling 的开销差异。
- [ ] **wandb Table 记录样本**:除了 summary,把 `records` 里几条采样样本作为 `wandb.Table` 上传,可以在 UI 里直接翻。适合定性检查。
- [ ] **异步日志**:大 batch 采样阻塞训练;可用后台线程 / 独立 rollout worker 与训练解耦(vllm 官方推荐)。

### C. 本作业相关

- [ ] **接入 SFT 训练脚本**:在 `cs336_alignment/` 里找 SFT 训练入口,每 N 步调 `log_generations`,用一小份 GSM8K/MATH 验证集做采样。观察 `mean_response_len` 和 `mean_answer_reward` 的关系。
- [ ] **接入 GRPO rollout**:GRPO 里 rollout batch 天然就是"prompt+采样",复用 `log_generations` 的 record 结构可以省掉一套日志代码——直接把 rollout 数据格式对齐即可。
- [ ] **分析已有 log**:训练一段时间后,用 pandas 读 `gen_log.jsonl`,按 `step` 分组画 reward/长度/熵三条曲线,找 hacking / 崩塌拐点。
