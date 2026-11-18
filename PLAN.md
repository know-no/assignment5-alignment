# Assignment 5 (Alignment) 个人自学计划

## 目标与验收口径

当前目标是通过本地代码实践和指导学习，理解 MATH 上的 SFT、Expert Iteration、GRPO 及其消融方法。当前没有可用训练卡，使用本机 CPU/MPS 做小模型验证。

个人验收分为两条路线。代码部分由用户自己实现，以张量小例、定向测试和 tiny 模型的真实生成、反传、参数更新验证机制。实验部分通过讲义、可核实的已发表结论和案例指导，学习实验设计、曲线解读与结论边界，无需亲跑全部正式实验。

正式课程的 26 题要求保留为对照。当前按整题交付口径，#2–7、#10–15、#22 共 **13 题完成**，另 **13 题尚未完整交付**。其中 #16 的基础训练循环已经实现，正式 MATH 训练奖励曲线和生成样例仍缺。个人自学按下文的机制理解、代码实践和分析能力验收，完整课程复现留作以后可选的目标。

## 当前进度与资源

- 当前代码基线为 `5fca333`，分支为 `feature/my-work-2025`。
- SFT 基础题 #2–7 已完成。已有 `sft_optimizer_step`、多 epoch 训练循环，以及支持模型/数据路径、batch、microbatch、学习率、设备和 FP32/BF16 的 CLI。训练返回 history，入口打印配置和逐步指标。
- GRPO 基础题 #10–15 已完成。#16 已具备生成回答、整组评分与优势、编码、缓存 old log-prob、微批累积、梯度裁剪、参数更新和 history 的衔接。#22 已支持同批多 epoch 和分训练批更新。
- 前次六文件离线回归 **61 项通过**，含本地 Qwen 的 FP32 评分、CPU FP32/BF16 tiny 模型训练、MPS 迁移和真实 HF generate。包含 #22 和可选 KL 的 GRPO 定向回归 **36 项通过**。正式 MATH 实验状态单独记录。
- 本地产物仍只有 GSM8K 的 10 条 baseline 调试样本。正式 MATH baseline、SFT/EI/GRPO 训练实验、checkpoint、实验曲线和 `writeup.pdf` 尚未交付。EI 尚无完整流程。正式 MATH 错误分析已暂缓。
- `log_generations` 已实现并有训练衔接测试，普通 SFT/GRPO 循环尚未接入训练内采样日志、周期验证及保存恢复。是否补这些工程功能，按后续本地学习需要选择。
- Safety/RLHF 补充支线的 5 个 adapter 尚未实现，对应 7 个可选用例，作为独立选学内容记录。

| 验证文件 | 最新通过数 | 主要覆盖 |
|---|---|---|
| `tests/test_sft.py` | 11 | SFT 基础张量运算、评分、熵与微批反传 |
| `tests/test_logging.py` | 13 | 采样日志、空响应、模式恢复及 tiny FP32/BF16 训练衔接 |
| `tests/test_sft_optimizer_step.py` | 4 | 微批累积与整批对照、裁剪及丢尾 |
| `tests/test_train_sft.py` | 12 | 离线 CLI、多 epoch、FP32/BF16 更新及输入检查 |
| `tests/test_grpo.py` | 14 | GRPO 基础损失、组内优势、mask 与微批训练步 |
| `tests/test_train_grpo.py` | 7 | 三种 loss 的两轮更新、空回答保护、MPS 和 tiny HF generate |

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
| 9 | `expert_iteration_experiment` | 尚无完整 EI 流程与实验 | 指导学习采样→判分过滤→SFT→重新采样的迭代机制与分析方法，本地流程实践后续可选 |

### #10–16　GRPO 基础与训练循环

| # | Problem | 当前状态 | 个人处理方式 |
|---|---|---|---|
| 10 | `compute_group_normalized_rewards` | 完成 | 保留完整回答组计算 reward/advantage 的实现与测试 |
| 11 | `compute_naive_policy_gradient_loss` | 完成 | 保留固定回答权重向 token 广播的实现与测试 |
| 12 | `compute_grpo_clip_loss` | 完成，已讨论 min/clamp、优势符号与梯度 | 作为用户指定的原理复习项，结合已有图与数值例子继续巩固 |
| 13 | `compute_policy_gradient_loss` | 完成 | 保留不同 loss 分支选择的实现与测试 |
| 14 | `masked_mean` | 完成 | 保留 token 轴与 batch 轴归约的实现与测试 |
| 15 | `grpo_microbatch_train_step` | 完成 | 保留每条回答 token 均值→batch 均值→除以 K 的反传验证 |
| 16 | `grpo_train_loop` | 基础循环与 7 项训练回归已实现，正式 MATH reward 曲线和样例未交付 | 本地代码成果已具备，接下来指导分析训练 reward、验证表现与样例之间的关系 |

### #17–26　实验、消融、Off-policy 与 Leaderboard

| # | Problem | 当前状态 | 个人处理方式 |
|---|---|---|---|
| 17 | `grpo_learning_rate` | 正式实验未做 | 后续待选，学习范围或实测范围按后续目标确定 |
| 18 | `grpo_baselines` | 损失分支已有，正式对比实验未做 | 指导学习 baseline 的作用、估计噪声与公平对比方法 |
| 19 | `think_about_length_normalization` | 思考与分析尚待系统整理 | 指导推导长度权重，用短/长回答小例解释目标差异和条件 |
| 20 | `grpo_length_normalization` | 正式归一化对比实验未做 | 指导比较不同归约的梯度尺度、长度权重及曲线读法 |
| 21 | `grpo_group_standard_deviation` | 开关已有，正式消融实验未做 | 指导学习组内标准差归一化的机制与消融分析 |
| 22 | `grpo_off_policy` | 已实现同批多 epoch、分训练批更新，通过本地测试 | 结合小模型验证复习 old 缓存和参数更新边界 |
| 23 | `grpo_off_policy_sweep` | 正式 sweep 未做 | 后续待选，完成 #22 后再决定学习或实践范围 |
| 24 | `grpo_off_policy_clip_ablation` | 正式消融未做 | 后续待选，学习重点是 Clip 与 No-Clip 的对照设计 |
| 25 | `grpo_prompt_ablation` | 正式 prompt 消融未做 | 后续待选，学习重点是 prompt 与配套奖励设置的对照 |
| 26 | `leaderboard` | 未做 | 正式讲义属于主作业，个人自学选做，当前保留挑战内容与评估协议的理解 |

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

## 指导学习实验的路线与验收

用户指定的 #1、8、9、16、12、18、20、21、19 按下表衔接。#12 保持已实现、已学习状态，承担原理复习，不替换成其他题号。

| 学习单元 | 要回答的机制问题 | 可验收内容 |
|---|---|---|
| #1 Baseline | 如何定义一次评估，格式与答案奖励分别衡量什么 | 能用小例统计三桶和准确率，说明采样设置、数据划分及错误案例的分析方法 |
| #8 SFT | 数据规模与数据质量怎样分别设计对照 | 能设计 `{128,256,512,1024}` 与全量数据、正确性过滤的比较，指出训练步数/epoch、过滤后数据量与验证集等混杂因素 |
| #9 EI | 为什么要重新采样、过滤和训练，每轮哪些数据与模型发生变化 | 能追踪至少两轮模型→回答→过滤数据→新模型，说明 rollout 数、SFT epoch、题目批量会影响什么，理解讲义五轮比较的目的 |
| #16 + #12 训练与 Clip | 完整回答奖励如何变成 token 梯度，单次更新与同批复用有何区别 | 能结合已有代码画更新顺序，计算 A 正负下的 min/clamp 分支与参数更新，分析 reward 曲线和验证准确率不同步的可能原因 |
| #18 Baseline 消融 | 减去组均值改变了什么，如何判断对比是否公平 | 能给出含固定奖励/优势的小例，区分单批梯度与跨采样的估计性质，列出对照需固定的条件 |
| #20 + #19 长度归一化 | token 均值、回答均值和固定常数分别给哪些项加权 | 能推导并计算长短回答的权重，读长度/熵/奖励曲线时识别归约和梯度尺度的影响，不笼统断言某一归约完全消除长度偏差 |
| #21 标准差消融 | 除以组内标准差怎样改变组间信号尺度 | 能比较同奖励差、不同组标准差的小例，说明零方差组与 eps 的作用，提出固定其他变量的比较方法 |

每个单元都应落到机制解释、小例计算、曲线或案例分析以及带条件的结论。使用公开材料时记录模型、数据、训练预算和评估配置，再判断结论能否迁移到本项目。

分析中明确区分三种证据。数学结论由公式与假设推导，公开实测发现必须对应可核实的来源和实验条件，教学假设或示意曲线显式标注。当前没有正式 MATH accuracy 或训练曲线，不把讲义目标、论文数值或教学图当成本项目实测结果。正式 MATH 全量错误分析继续暂缓，不作为后续代码学习的前置。

## 本地环境与验证方式

现有项目 `.venv` 用于本地代码和测试，按改动范围运行定向检查。最近一次 61 项回归的范围为

```sh
env -u UPDATE_SNAPSHOTS HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest \
  tests/test_sft.py tests/test_logging.py tests/test_sft_optimizer_step.py \
  tests/test_train_sft.py tests/test_grpo.py tests/test_train_grpo.py -q
```

历史推理调试使用独立的 `~/.venv-vllm-metal/` 环境。当前 baseline 入口仍配置 GSM8K 10 条与 Mac 调试模型，正式 MATH 入口尚待替换。[math_baseline.py](cs336_alignment/math_baseline.py#L24)

如后续恢复训练卡资源，再按实际设备核对依赖、模型、数据、显存和持久化存储。先做短生成、评分、有限 loss/梯度与参数更新验证，再按需补 checkpoint 保存恢复和所选正式实验，不预设卡型、吞吐或排期。

## 后续待选

#17、23–25 保持待选状态。可先通过指导阅读设计和结果学习，再决定是否亲自复现。后续算力安排服务于主动选择的实验。

Safety/RLHF 是独立的可选补充，共 16 题。可选的本地练习包括 packed dataset、batch 迭代、MMLU/GSM8K parser 与 DPO loss 的 5 个 adapter，当前留待以后选择。

## 个人检查清单

- [x] SFT 基础 #2–7 完成，训练 CLI、多 epoch 与 FP32/BF16 tiny 更新验证已具备。
- [x] GRPO 基础 #10–15 完成，单批单次更新循环与 7 项训练回归已具备。
- [x] 六文件离线回归 61 项通过，正式训练产物与测试证据已区分。
- [x] #22 同批多 epoch 与分训练批已实现，已验证缓存、更新次数与梯度累积边界。
- [ ] 通过指导完成 #1、8、9 的实验设计与结果分析学习，正式错误分析仍暂缓。
- [ ] 结合现有 #12、16 和 #22 的实现，完成 #18、20、21、19 的机制、小例与条件结论学习。

#17、23–25、#26 及 Safety/RLHF 不计入当前个人必做清单，后续可重新选择。正式课程口径的未交付项仍如上表保留，个人自学达成与课程完整交付分别记录。
