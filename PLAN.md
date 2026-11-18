# Assignment 5 (Alignment) 个人自学计划

## 目标与验收口径

当前目标是理解数学推理训练中的 SFT、Expert Iteration、GRPO，以及指令与安全对齐中的偏好数据、DPO 和评估。当前没有可用训练卡，已用本机 CPU/MPS 验证训练机制。

当前安排的动手实现已完成。剩余的 DPO、Safety/RLHF 和主作业实验统一转入指导学习，用讲义、小例和公开实践理解机制、实验设计、曲线解读与结论边界。完整训练、评估和复现可按兴趣另行选择。

正式课程的 26 题要求保留为对照。当前按整题交付口径，#2–7、#10–15、#22 共 **13 题完成**，另 **13 题尚未完整交付**。其中 #16 的基础训练循环已经实现，正式 MATH 训练奖励曲线和生成样例仍缺。个人自学按下文的机制理解、代码实践和分析能力验收，完整课程复现留作以后可选的目标。

## 当前进度与资源

- 当前是 `feature/my-work-2025` 分支。GRPO 同批复用与可选 KL 的提交为 `b5dcd3d`。
- SFT 基础题 #2–7 已完成。已有 `sft_optimizer_step`、多 epoch 训练循环，以及支持模型/数据路径、batch、microbatch、学习率、设备和 FP32/BF16 的 CLI。训练返回 history，入口打印配置和逐步指标。
- GRPO 基础题 #10–15 已完成。#16 已具备生成回答、整组评分与优势、编码、缓存 old log-prob、微批累积、梯度裁剪、参数更新和 history 的衔接。#22 已支持同批多 epoch 和分训练批更新。
- 最新七文件离线回归 **77 项通过**，涵盖 SFT、日志、GRPO 与 DPO。正式 MATH 实验状态单独记录。
- 本地产物仍只有 GSM8K 的 10 条 baseline 调试样本。正式 MATH baseline、SFT/EI/GRPO 训练实验、checkpoint、实验曲线和 `writeup.pdf` 尚未交付。EI 尚无完整流程。正式 MATH 错误分析已暂缓。
- `log_generations` 已实现并有训练衔接测试，普通 SFT/GRPO 循环尚未接入训练内采样日志、周期验证及保存恢复。是否补这些工程功能，按后续本地学习需要选择。
- Safety/RLHF 补充题 `dpo_loss` 已完成并接入测试。tiny GPT-2 上验证了初始 loss、一次 `optimizer.step()` 后的 loss 下降，以及参考模型无梯度、参数固定。CPU/MPS 交叉放置也通过了单独验证。HH 数据处理、正式 DPO 训练和评估仍待学习，补充题的其余 4 个 adapter 未实现。[DPO 实现](cs336_alignment/dpo_utils.py#L12) · [DPO 测试](tests/test_dpo.py#L8)

| 验证文件 | 最新通过数 | 主要覆盖 |
|---|---|---|
| `tests/test_sft.py` | 11 | SFT 基础张量运算、评分、熵与微批反传 |
| `tests/test_logging.py` | 13 | 采样日志、空响应、模式恢复及 tiny FP32/BF16 训练衔接 |
| `tests/test_sft_optimizer_step.py` | 4 | 微批累积与整批对照、裁剪及丢尾 |
| `tests/test_train_sft.py` | 12 | 离线 CLI、多 epoch、FP32/BF16 更新及输入检查 |
| `tests/test_grpo.py` | 15 | GRPO 基础损失、组内优势、mask、微批训练步和 KL 数值与梯度 |
| `tests/test_train_grpo.py` | 21 | 同批多轮更新、参数校验、MPS、tiny HF generate 和可选 KL |
| `tests/test_dpo.py` | 1 | 单条 DPO loss 数值、真实参数更新及固定参考模型 |

## 26 题索引与个人处理方式

下表保留主讲义题号。实现完成、正式实验交付和个人学习安排分别记录，避免用测试用例数量代替课程题目数量。

### #1–9　Baseline、SFT 与 Expert Iteration

| # | Problem | 当前状态 | 个人处理方式 |
|---|---|---|---|
| 1 | `math_baseline` | 评估/分桶工具已实现，已有 GSM8K 10 条调试，正式 MATH 未交付 | 指导学习评估口径、三桶、准确率和错误分析方法。全量运行及正式错误案例报告暂缓 |
| 2 | `tokenize_prompt_and_output` | 完成 | 保留本地实现与测试，作为后续训练输入基础 |
| 3 | `compute_entropy` | 完成 | 保留数值边界回归与熵的解释 |
| 4 | `get_response_log_probs` | 完成 | 保留逐 token 评分、labels 对齐与测试 |
| 5 | `masked_normalize` | 完成 | 在长度归一化学习中复用固定常数与 mask 的知识 |
| 6 | `sft_microbatch_train_step` | 完成 | 保留微批归约、反传与梯度累积验证 |
| 7 | `log_generations` | 完成 | 保留日志工具与 tiny 训练衔接验证，普通循环接入按需选做 |
| 8 | `sft_experiment` | 训练 CLI 与 loop 已有，正式规模/过滤实验未做 | 指导学习数据规模、质量过滤和验证曲线，无需亲跑完整矩阵 |
| 9 | `expert_iteration_experiment` | 尚无完整 EI 流程与实验 | 指导学习采样、判分过滤、继续 SFT 和重新采样，以及每轮数据的变化 |

### #10–16　GRPO 基础与训练循环

| # | Problem | 当前状态 | 个人处理方式 |
|---|---|---|---|
| 10 | `compute_group_normalized_rewards` | 完成 | 保留完整回答组计算 reward/advantage 的实现与测试 |
| 11 | `compute_naive_policy_gradient_loss` | 完成 | 保留固定回答权重向 token 广播的实现与测试 |
| 12 | `compute_grpo_clip_loss` | 完成，已讨论 min/clamp、优势符号与梯度 | 作为用户指定的原理复习项，结合已有图与数值例子继续巩固 |
| 13 | `compute_policy_gradient_loss` | 完成 | 保留不同 loss 分支选择的实现与测试 |
| 14 | `masked_mean` | 完成 | 保留 token 轴与 batch 轴归约的实现与测试 |
| 15 | `grpo_microbatch_train_step` | 完成 | 保留每条回答 token 均值→batch 均值→除以 K 的反传验证 |
| 16 | `grpo_train_loop` | 基础循环已实现，正式 MATH reward 曲线和样例未交付 | 指导分析训练 reward、验证表现与样例之间的关系 |

### #17–26　实验、消融、Off-policy 与 Leaderboard

| # | Problem | 当前状态 | 个人处理方式 |
|---|---|---|---|
| 17 | `grpo_learning_rate` | 正式实验未做 | 指导学习学习率对照、训练稳定性和验证曲线 |
| 18 | `grpo_baselines` | 损失分支已有，正式对比实验未做 | 指导学习 baseline 的作用、估计噪声与公平对比方法 |
| 19 | `think_about_length_normalization` | 思考与分析尚待系统整理 | 指导推导长度权重，用短/长回答小例解释目标差异和条件 |
| 20 | `grpo_length_normalization` | 正式归一化对比实验未做 | 指导比较不同归约的梯度尺度、长度权重及曲线读法 |
| 21 | `grpo_group_standard_deviation` | 开关已有，正式消融实验未做 | 指导学习组内标准差归一化的机制与消融分析 |
| 22 | `grpo_off_policy` | 已实现同批多 epoch、分训练批更新，通过本地测试 | 结合小模型验证复习 old 缓存和参数更新边界 |
| 23 | `grpo_off_policy_sweep` | 正式 sweep 未做 | 指导比较 rollout 数据复用、更新次数、耗时和验证表现 |
| 24 | `grpo_off_policy_clip_ablation` | 正式消融未做 | 指导学习 Clip 与 No-Clip 的公平对照，区分策略比裁剪和梯度范数裁剪 |
| 25 | `grpo_prompt_ablation` | 正式 prompt 消融未做 | 指导学习 prompt、格式奖励和答案奖励怎样分别影响结论 |
| 26 | `leaderboard` | 未做 | 指导学习固定训练预算下的评估协议和实验取舍，竞榜留作选做 |

正式讲义的 #26 要求在 2 张 H100、4 小时训练约束下报告结果，是综合前面算法、在固定预算内提高 MATH 成绩的挑战。个人自学以理解与代码实践验收，竞榜、排名和官方硬件预算列为选做。以后如主动选择完整复现，再单独安排对应实验。

主讲义 25 个标分项合计 66 分，#22 未单独标分。讲义标注的实验预算合计 52 H100 参考小时，含 Leaderboard 的 16 小时，属于课程背景，不作为当前执行前提，也不换算成本机或其他硬件的实测工时。

## #22 的本地实现与验证

训练循环每次生成一批回答，再按训练批遍历指定的 epoch 数。[grpo_train_loop](cs336_alignment/train_grpo.py#L199)、[grpo_optimizer_step](cs336_alignment/train_grpo.py#L76)

- 默认 `epochs_per_rollout_batch=1`，`train_batch_size=None` 表示使用整批回答，两项都取默认值时每批只更新一次。
- `num_steps` 表示总 optimizer 更新次数。新增复用层后区分 rollout 批次计数与更新计数，达到总更新数就结束。
- 每批回答只生成、评分、计算优势并缓存 old log-prob 一次。同批复用期间回答、reward、advantage 和 old 分数固定。
- 每次更新重新计算带图的 current log-prob，清梯度，完成 K 个微批反传后再裁剪和 step。step 继续放在微批循环之外。
- 本地测试核对了 `rollout=8`、`train_batch=4`、`microbatch=2`、`epochs=2` 时的切片顺序、旧分数对齐、总更新预算和 history，也验证了默认配置。
- 可选 KL 默认关闭。开启后用固定 reference 为每批回答评分，逐 token 的估计式、旧样本复用边界和实现见 [GRPO 学习笔记](notes/grpo-clip-single-update.md) 第 3 节。

验收时应能指出哪些量每个 rollout 刷新、哪些量每次更新刷新，并解释首次 ratio 接近 1 和后续可能偏离的条件。小例通过后再决定是否增加保存恢复、训练内采样日志或普通 GRPO CLI。

## 后续指导学习路线

先从已经跑通的单条 DPO 更新往下学，再回到数学推理训练与消融。每个单元用一个小例讲机制，随后读公开实测并分析它能支持什么结论。#12 继续作为 Clip 原理复习项。

| 顺序 | 学习单元 | 机制、分析与验收 |
|---|---|---|
| 1 | **DPO 剩余内容** · `look_at_hh`、`dpo_training` | 走一条 HH 记录，解释为什么要对齐同一 prompt、拆 `chosen/rejected`、过滤多轮和检查偏好噪声。把已有四分数和一步更新延伸到梯度累积、冻结参考模型、验证集与 checkpoint 选择。区分模型原始偏好分差、减去 reference 后的分差和生成偏好胜率。讨论 `beta`、数据来源与回答长度可能怎样影响结果。 |
| 2 | **指令与安全对齐** · Safety/RLHF 补充 | 书中第 7 章的同类指令 SFT 以复习为主，补样本 packing、批迭代、评估 parser 和 `SFT → DPO → 评估` 的衔接。用 AlpacaEval、SimpleSafetyTests、MMLU、GSM8K 区分指令遵循、安全拒答与能力保留。讨论红队评估、过度拒答、裁判偏差和 alignment tax。 |
| 3 | **评估、SFT 与 EI** · #1、#8–9 | 用小样本核对格式与答案奖励、分桶和可复现的评估配置。以讲义中的 `128/256/512/1024` 条和全量数据为例，设计规模和质量过滤的对照，区分样本量、训练步数和数据质量。追踪 EI 两轮的模型、采样、筛选和 SFT，解释 rollout 数及 SFT epoch 的作用。 |
| 4 | **GRPO 训练与消融** · #12、#16–21 | 从已有代码追到奖励、优势、Clip、归一化和更新。用短长回答对照 token 均值、回答均值和常数归一化，再看 baseline、组标准差和学习率各改变什么。结合 reward、验证准确率、熵、长度和采样案例分析训练行为。 |
| 5 | **Off-policy 与 Prompt** · #22–25 | 复习同批多个 epoch 中 old/current 的边界。比较按采样数、更新数和墙钟时间对齐的实验，分析复用与新旧策略偏离的权衡。Clip/No-Clip 对照中保留新旧概率比，只改变裁剪。做 prompt 对照时，把格式奖励的变化与答案正确率的变化分开看。 |
| 6 | **固定预算与 2026 拓展** · #26、MaxRL、GSPO | 理解 leaderboard 的训练预算和验证协议。沿一组 rollout 对比 MaxRL 的优势归一化与 GSPO 的序列 ratio，说明它们分别改动训练链路的哪一步。对照 [2026 主讲义](https://github.com/stanford-cs336/assignment5-alignment/blob/main/cs336_spring2026_assignment5_alignment.pdf) 和原论文。 |

### 业界案例与证据

| 公开资料 | 后续具体看什么 |
|---|---|
| [Anthropic HH 数据](https://github.com/anthropics/hh-rlhf)及[论文](https://arxiv.org/abs/2204.05862)、[DPO 原论文](https://arxiv.org/abs/2305.18290) | 从同 prompt 的偏好对出发，对照显式 reward model 加 RL 与直接做 DPO 的数据、训练和评估流程。 |
| [Meta Llama 3 技术报告](https://arxiv.org/html/2407.21783#S4)、[Hugging Face Zephyr 模型卡](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)、[SimpleSafetyTests](https://arxiv.org/abs/2311.08370) | 看 Llama 3 怎样刷新偏好数据、Zephyr 怎样用合成偏好及记录分差。比较指令遵循、安全表现及能力评估，核对能否把结果归因到某个训练阶段。 |
| [STaR](https://arxiv.org/abs/2203.14465)、[DeepSeekMath](https://arxiv.org/abs/2402.03300) | 对照筛出正确轨迹再训练和在 rollout 上做策略更新。核对奖励信号、模型起点和评估条件。 |
| [Dr. GRPO](https://arxiv.org/abs/2503.20783)、[DAPO](https://arxiv.org/abs/2503.14476)、[MaxRL](https://arxiv.org/abs/2602.02710)、[GSPO](https://arxiv.org/abs/2507.18071) | 从公开消融里拆出长度权重、组内归一化、采样和重要性比值的作用，区分单因素结论与整套配方的结果。 |

每个案例记录**问题、方法、对照、观测和适用范围**。引用公开曲线时记下模型、数据、训练和推理预算、评估口径，明确区分论文实测、公式推论、本地小例三种证据。正式 MATH 全量错误分析继续暂缓，用小样本学习分析方法即可。

## 本地环境与验证方式

现有项目 `.venv` 用于本地代码和测试，按改动范围运行定向检查。最近一次七文件回归命令如下。

```sh
env -u UPDATE_SNAPSHOTS HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest \
  tests/test_sft.py tests/test_logging.py tests/test_sft_optimizer_step.py \
  tests/test_train_sft.py tests/test_grpo.py tests/test_train_grpo.py tests/test_dpo.py -q
```

历史推理调试使用独立的 `~/.venv-vllm-metal/` 环境。当前 baseline 入口仍配置 GSM8K 10 条与 Mac 调试模型，正式 MATH 入口尚待替换。[math_baseline.py](cs336_alignment/math_baseline.py#L24)

如后续恢复训练卡资源，再按实际设备核对依赖、模型、数据、显存和持久化存储。先做短生成、评分、有限 loss/梯度与参数更新验证，再按需补 checkpoint 保存恢复和所选正式实验，不预设卡型、吞吐或排期。

## 个人检查清单

- [x] SFT 基础 #2–7 完成，训练 CLI、多 epoch 与 FP32/BF16 tiny 更新验证已具备。
- [x] GRPO 基础 #10–15 和本地训练循环已完成。
- [x] #22 同批多 epoch 与分训练批已实现，已验证缓存、更新次数与梯度累积边界。
- [x] 补充题 `dpo_loss` 和 tiny GPT-2 单步更新已验证。
- [ ] 指导学习 HH 偏好数据、DPO 训练流程、安全评估和公开案例。
- [ ] 指导学习 #1、#8–9 的评估设计和复现条件。
- [ ] 指导学习 #16–21、#23–26 的曲线分析、消融和预算取舍，结合已有 #12、#22 复习并延伸到 2026 的 MaxRL、GSPO。

个人计划接下来按上面的课程和案例推进。正式交付状态继续按主讲义记录，Safety/RLHF 是独立的选做补充。
