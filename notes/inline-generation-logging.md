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

### 2. 熵按 `mask.sum(-1)` 逐样本平均,而不是训练用的固定常数

这是 [sft-loss-normalization.md](sft-loss-normalization.md) 的对照场景。**同一个 mask+entropy 张量,两处用途、两种归一化**:

| 场景 | 分母 | 目的 |
| --- | --- | --- |
| 训练 loss(§1) | 固定 `normalize_constant` | 消除长度偏差,让每 token 一票 |
| **日志展示(本篇)** | **`mask.sum(-1)`(每条真实响应长度)** | **给出可比较的 nats/token,人眼可读** |

关键洞察:**归一化选谁,取决于你要什么样的可比性**:

- 训练要"每 token 权重一致"——所以除固定常数
- 日志要"每条样本的平均不确定度"——所以除各自长度

实现片段:

```python
# token_entropy: (B, L),mask: (B, L)
per_sample_sum = (token_entropy * mask).sum(dim=-1)      # (B,)
per_sample_count = mask.sum(dim=-1).clamp_min(1.0)       # 防 0 除
per_sample_mean = per_sample_sum / per_sample_count      # (B,) 单位是 nats/token
```

这里 `dim=-1` 是精确匹配 [sft-loss-normalization.md](sft-loss-normalization.md) §2 里"逐样本 loss"那条路径的实战案例。

### 3. 长度按 token 数,分桶按 `answer_reward`

**长度用 token 数,不是字符数**:与训练时 mask 分母的口径一致,才能和 loss 里的"序列长度"直接对得上。

**正确/错误分桶用 `answer_reward==1`,不用 `reward==1`**:

r1_zero 的 reward 规则里(见 [drgrpo_grader.py:1008](../cs336_alignment/drgrpo_grader.py#L1008) `r1_zero_reward_fn`):

```python
if 格式对 and 答案对: reward = 1
if 格式对 and 答案错: reward = 0   # ← 格式对没奖励,防 hacking
if 格式错:            reward = 0
```

所以 `reward==0` 里混着"格式错"和"答案错"两类。用 `answer_reward==1` 分桶能拆出**答对了没**这个更纯的信号,`mean_correct_len` / `mean_wrong_len` 才有解释力。

## 记录什么、聚合什么

**每条 record**(适合事后 grep / 用 pandas 分析):

- `prompt` / `response` / `ground_truth`
- `format_reward` / `answer_reward` / `reward`
- `response_len_tokens`
- `mean_token_entropy`(可选)
- `step`

**每批 summary**(适合直接扔进 wandb / print):

- `n`, `step`
- `mean_reward` / `mean_format_reward` / `mean_answer_reward`
- `mean_response_len` / `mean_correct_len` / `mean_wrong_len`
- `mean_token_entropy`

**落盘用 JSONL 追加**:一条记录一行,不用担心训练崩了丢文件,后续用 `jq` 或 `pandas.read_json(..., lines=True)` 都很顺手。

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

1. **`policy_model.eval()` + `torch.no_grad()`**:算熵的前向不能影响 dropout/BN 统计,也不需要建图。函数里用 try/finally 恢复原状态,避免污染训练模式。
2. **empty list**:某批可能全对或全错,`fmean([])` 会抛异常;用 `_safe_mean` 返回 `None`,让下游(wandb)自己处理缺失。
3. **响应里的特殊 token**:vllm 通常已经去了 EOS,但 HF `generate` 输出可能带。落盘前若要展示,记得 `tokenizer.decode(..., skip_special_tokens=True)`;算长度时反过来——**要包含 EOS**,因为 EOS 也是模型学出来的 token,和 loss 口径一致。

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
