# Assignment 5 (Alignment) 个人执行计划

## 目标

完成 CS336 Assignment 5（MATH 数据集上的 SFT / Expert Iteration / GRPO），
**本机 (Mac M2) 继续实现和验证代码，使用现有的 1 张 H20 分阶段开展训练与评估。**

主作业全部 26 个 Problem 均纳入完成范围，包括 Off-policy GRPO 和 Leaderboard。Safety/RLHF 补充作业作为可开展的支线，优先完成 MATH 主线，再按资源安排补充实验。

## 当前进度与资源

- 基础工具基线为 `45e7df3`。SFT 六个基础题（题号 2-7）已完成，本机 SFT 与日志相关测试共 **24 项通过**，包括真实微型 HF causal LM 的 FP32/BF16 生成、反传、梯度累积、日志与参数更新衔接。
- `sft_optimizer_step` 已实现，一次调用完成有效 batch 的微批累积、梯度裁剪和参数更新。新增 4 项 CPU FP32 整批对照测试，与原 3 项微批测试组成的定向回归共 **7 项通过**，完整多 step 训练入口仍待实现。
- 主作业仍有 **20 项未完整交付**，包括正式 MATH baseline、完整训练入口、训练实验、消融及书面结果。现有 baseline 产物是 GSM8K 的 10 条调试样本。
- GRPO 的 14 项测试因基础函数尚未实现而失败。另有 7 项失败属于可选 Safety/RLHF 补充作业，对应 5 个 adapter。
- GPU 资源为 **1 × H20，尚未启动**。显存、驱动、CUDA、PyTorch/vLLM 环境、可用存储和运行性能均待启动后核对。
- 按当前阶段需要的功能推进即可先开展 baseline 和 SFT，GRPO 与可选题的实现继续在本机完成。

---

## 策略总览

| 阶段 | 在哪里做 | 目标 | 验收依据 |
|---|---|---|---|
| 阶段 1：本机继续实现 | Mac | 补完整训练 loop、MATH 数据读取及后续 GRPO 基础函数 | 当前阶段的定向测试与微型模型验证 |
| 阶段 2：H20 环境小验证 | 单张 H20 | 核对环境，运行短生成、评分及少量训练步骤 | 实际设备信息、有限 loss/梯度、checkpoint 保存与恢复 |
| 阶段 3：主线实验 | 单张 H20 | MATH baseline → SFT → EI → GRPO/消融/off-policy → Leaderboard 实验 | 完整验证集指标、曲线、配置与硬件记录 |
| 阶段 4：结果与交付 | Mac | 分析结果、画图、写报告并整理代码 | `writeup.pdf`、代码及可追溯实验产物 |

---

## 任务拆解（按 PDF 中的 Problem 列出）

主讲义共 26 个 Problem，其中 25 个标分项合计 66 分，`grpo_off_policy` 未单独标分。表中的 `H100 hrs` 保留讲义参考预算，H20 的实际用时在运行后记录。

### Part 1: Zero-shot MATH Baseline（§3）

| # | Problem | 分值 | GPU? | Mac 可做? |
|---|---|---|---|---|
| 1 | `math_baseline` | 4 | 推理 | 已跑通 GSM8K 10 条调试，正式 MATH 评估待补 |

产出物包括正式 MATH 验证集评估脚本、样本与分数、三种奖励桶的数量以及准确率。对格式错误、格式正确但答案错误这两类，各分析至少 10 个案例，并展示三类响应的代表例子。

### Part 2: SFT (§4 / Supervised Finetuning)

| # | Problem | 分值 | 实现位置 |
|---|---|---|---|
| 2 | `tokenize_prompt_and_output` | 2 | `tests/adapters.py` |
| 3 | `compute_entropy` | 1 | `tests/adapters.py` |
| 4 | `get_response_log_probs` | 2 | `tests/adapters.py` |
| 5 | `masked_normalize` | 1 | `tests/adapters.py` |
| 6 | `sft_microbatch_train_step` | 3 | `tests/adapters.py` |
| 7 | `log_generations` | 1 | `cs336_alignment/utils/log_utils.py` |
| 8 | `sft_experiment` | 2 | **2 H100 hrs** — GPU 跑 SFT 完整训练 |

题号 2-7 已完成并接入测试。下一步在本机补完整 SFT 训练入口，再到单张 H20 做短运行和正式实验。题号 8 需要比较数据规模 `{128, 256, 512, 1024}` 与全量数据，目标是全量 SFT 验证准确率至少 15%，并补做仅保留答案正确样本的全量过滤实验，报告过滤后数据量及验证曲线。

### Part 3: Expert Iteration (§5)

| # | Problem | 分值 | GPU? |
|---|---|---|---|
| 9 | `expert_iteration_experiment` | 2 | **6 H100 hrs** |

在本机实现采样、过滤和迭代脚本，单张 H20 上顺序运行生成与训练。按讲义做 5 轮 EI，比较至少两种 rollout 数和 SFT epoch 数，并比较每轮题目批量 `{512, 1024, 2048}` 中的配置，保留验证曲线与响应熵曲线，目标验证准确率至少 15%。

### Part 4: GRPO 基础 (§7)

| # | Problem | 分值 | 实现位置 |
|---|---|---|---|
| 10 | `compute_group_normalized_rewards` | 2 | `tests/adapters.py` |
| 11 | `compute_naive_policy_gradient_loss` | 1 | `tests/adapters.py` |
| 12 | `compute_grpo_clip_loss` | 2 | `tests/adapters.py` |
| 13 | `compute_policy_gradient_loss` | 1 | `tests/adapters.py` |
| 14 | `masked_mean` | 1 | `tests/adapters.py` |
| 15 | `grpo_microbatch_train_step` | 3 | `tests/adapters.py` |
| 16 | `grpo_train_loop` | 5 | 训练 loop 脚本（不只是 adapter） |

在本机完成题号 10-15 的 6 个基础函数及题号 16 的训练 loop，先验证对应的 14 项单测，再在 H20 上检查短训练的奖励、梯度及生成样本。

### Part 5: GRPO 实验与消融 (§8)

| # | Problem | 分值 | GPU? |
|---|---|---|---|
| 17 | `grpo_learning_rate` | 2 | **6 H100 hrs** |
| 18 | `grpo_baselines` | 2 | **2 H100 hrs** |
| 19 | `think_about_length_normalization` | 1 | 纯思考题 |
| 20 | `grpo_length_normalization` | 2 | **2 H100 hrs** |
| 21 | `grpo_group_standard_deviation` | 2 | **2 H100 hrs** |

### Part 6: Off-policy GRPO（§8 续）

| # | Problem | 分值 | GPU? |
|---|---|---|---|
| 22 | `grpo_off_policy`（实现） | 未单独标分 | Mac 写代码，H20 验证 |
| 23 | `grpo_off_policy_sweep` | 4 | **12 H100 hrs** |
| 24 | `grpo_off_policy_clip_ablation` | 2 | **2 H100 hrs** |
| 25 | `grpo_prompt_ablation` | 2 | **2 H100 hrs** |

### Part 7: Leaderboard（§9）

| # | Problem | 分值 | GPU? |
|---|---|---|---|
| 26 | `leaderboard` | 16 | **16 H100 hrs** |

正式讲义要求报告 **2 张 H100 上、4 小时训练内**获得的结果，并在整个 MATH 验证集（5000 题）上按规定的 R1-Zero prompt、`temperature=1.0`、`max_tokens=1024` 和原始 `r1_zero_reward_fn` 评估。当前可先用单张 H20 开展算法实验，报告实际硬件与耗时。正式双 H100 预算下的结果需要另行安排资源验证。

---

### 算力预算汇总

| 类型 | 分值与范围 | 讲义 H100 参考小时 |
|---|---|---|
| 代码与训练 loop | **25 分**，题号 2-7、10-16 | 代码可在本机实现，loop 仍需运行验证 |
| 评估与实验 | 40 分，题号 1、8、9、17、18、20、21、23-26 | **52 H100 小时**，按有标注的题目合计，含 Leaderboard 的 16 小时 |
| 纯思考题 | 1 分，题号 19 | 无训练预算标注 |

H100 小时是讲义参考预算。单张 H20 的吞吐、显存需求和实际工时，需要在启动后用当前脚本测量，再决定 batch size 和实验排期。

题号 22-26 均属于主作业。主讲义第 33 页明确将 Leaderboard 称为 mandatory assignment 的最后一部分。分阶段推进时可以后做，完整交付仍需包含这些题目。

### Safety/RLHF 可选支线

单独的补充讲义包含 16 个 Problem，学习顺序为

```text
Llama 3.1 8B Base 四项基线
  → UltraChat-200K + SafetyTunedLlamas 指令 SFT
  → 四项重评与 red teaming
  → Anthropic HH chosen/rejected 偏好对上的 DPO
  → 再评估
```

四项评估为 MMLU、GSM8K、AlpacaEval 和 SimpleSafetyTests。传统 reward model、PPO/RLHF 作为背景，补充作业实际要求实现和运行的偏好训练算法是 DPO。

- 本机可先完成 packed SFT dataset、batch 迭代、MMLU parser、GSM8K parser 和 DPO loss 这 5 个 adapter，对应当前未实现的 7 项测试。
- 补充训练使用 8B 模型。DPO 需要训练 policy 并使用冻结 reference，先在启动后的 H20 上用小 microbatch 验证实际显存与反传，再安排完整实验。
- AlpacaEval 和 SimpleSafetyTests 的讲义原方案使用 Llama 3.3 70B Instruct 评审，要求 2 张 GPU、每张显存大于 80GB。这个评审环节的资源或方案需单独安排。
- 补充讲义的 SFT 实验标注 24 H100 小时参考预算，独立于主线的 52 H100 小时。各项 H20 实际用时均在运行后记录。

---

## 环境配置

### 本机（Mac，已配置好）

```sh
uv sync          # 装核心依赖，跳过 GPU-only 包
```

### 单张 H20（启动后核对）

启动后先记录设备型号和显存、驱动/CUDA、Python、PyTorch、vLLM、FlashAttention 版本，以及模型和数据所在位置、可持久化存储路径。当前 GPU 还未启动，这些环境信息均待确认。

项目的 `gpu` 可选依赖组提供 `vllm==0.7.2` 与 `flash-attn==2.7.4.post1` 作为环境配置起点。先核对 H20 的实际软件环境与依赖兼容性，再安装所需依赖并做短运行验证。

---

## Mac 与 GPU 调试环境

- 本机项目 `.venv` 用于代码和单元测试，独立的 `~/.venv-vllm-metal/` 用于 Mac 上的 vLLM 推理调试。本次已验证后者可以导入 `LLM`、`SamplingParams` 并激活 Metal 插件。
- 既有 Mac baseline 采用 `mlx-community/Qwen2.5-Math-1.5B-bf16` 和 GSM8K 10 条样本。正式迁移时改为 MATH 数据及 `Qwen/Qwen2.5-Math-1.5B`，核对采样配置与奖励函数。
- `cs336_alignment.llm_backend.build_llm()` 已提供平台和模型名映射。当前 `math_baseline.py` 仍直接构造 `LLM` 并使用调试模型名，迁移时需要实际接好入口。
- macOS 的多进程入口保持 `if __name__ == "__main__":` 保护。H20 的模型加载、推理和训练兼容性在启动后分别验证。

本机推理调试入口为

```sh
~/.venv-vllm-metal/bin/python -m cs336_alignment.math_baseline
```

---

## 工作流程

### 阶段 1：本机继续实现与验证

先补完整 SFT loop、MATH 数据入口、配置、保存与恢复，以及训练中调用 `log_generations` 的位置。当前 tiny 模型测试已经在 CPU 上覆盖 FP32/BF16 的真实生成、两次反传之间的日志调用、梯度保留和 optimizer 更新。

当前基础实现的验证命令为

```sh
env -u UPDATE_SNAPSHOTS HF_HUB_OFFLINE=1 .venv/bin/python -m pytest \
  tests/test_sft.py tests/test_logging.py tests/test_sft_optimizer_step.py -q
```

后续按对应阶段补 GRPO 的 6 个基础函数和训练 loop。可选支线的 5 个 adapter 与 7 项测试单独推进。

### 阶段 2：H20 环境与小验证

1. 启动 H20，记录设备、显存、软件版本、代码提交和可持久化目录。
2. 准备讲义所需 MATH `train.jsonl`、`validation.jsonl`、`sft.jsonl` 与 Qwen2.5-Math-1.5B Base，确认字段和模型版本。
3. 跑小批量生成、奖励计算与评分，再跑少量 SFT 步骤，检查 loss/梯度有限且参数确实更新。
4. 验证 checkpoint 保存、重载及训练继续，确认实际显存峰值与用时后再调 batch size。

### 阶段 3：单张 H20 主线实验

1. **正式 MATH baseline**。替换 GSM8K 10 条调试入口，跑完 5000 条验证集，保存原始响应与奖励，统计三桶、分析两类错误各至少 10 例，并报告准确率。
2. **SFT**。跑数据规模比较与正确答案过滤实验，记录每组配置、验证曲线、过滤后数据量和 checkpoint。
3. **Expert Iteration**。先完成一轮采样、过滤、SFT 和评估的衔接，再按题目要求做 5 轮和配置比较。
4. **GRPO、消融与 off-policy**。完成主循环短运行后，依次推进学习率、baseline、长度归一化、标准差归一化、off-policy sweep、去 clip 与 prompt 消融。
5. **Leaderboard 实验**。在 H20 上记录算法表现、硬件和实际 wall-clock。正式讲义的双 H100 / 4 小时约束按 Part 7 单独核对并安排验证。

单张卡先按阶段顺序复用生成、训练和评估资源。可在阶段边界保存 checkpoint，释放当前阶段的 GPU 资源，再加载需要的模型继续下一阶段。EI/GRPO 生成时使用当前策略的权重，保存优化器状态和步数以支持训练继续。脚本先跑通，再依据实际显存和耗时调整 batch。

日志可先写本地 JSONL 再生成曲线，wandb 按需接入。实验持续保存配置、代码版本、采样设置、loss、reward、响应长度、熵、梯度范数、显存峰值与耗时。checkpoint 写入已确认的持久化目录，保留模型、优化器及适用的 scheduler/RNG 状态，并在优化器更新边界验证恢复。远程运行可使用 `tmux` 保持会话。

### 阶段 4：本机收尾

- 汇总日志、验证结果、配置及 checkpoint metadata，确认曲线对应准确的代码和模型版本。
- 按主讲义逐题完成 `writeup.pdf`，报告实验对比与硬件、时间预算。
- 整理代码及所需产物，保存原始结果以便复核和恢复。

---

## 风险与备选

- GPU 依赖兼容性在启动后通过实际导入、短生成及反传确认，记录最终版本组合。
- 出现 OOM 时，先根据显存峰值减小生成或训练 microbatch，再评估梯度累积、gradient checkpointing 等配置。
- 同卡阶段切换时核对策略权重版本与资源释放，保存可恢复的训练状态。
- 8B 补充训练与 70B 评审单独评估资源需求，分别记录可运行条件与待安排环节。

---

## 检查清单

已有本机验证

- [x] SFT 六个基础题实现并接入测试。
- [x] SFT 与 logging 定向测试 24 项通过。
- [x] tiny HF causal LM 的 CPU FP32/BF16 生成、反传、日志及 optimizer 更新衔接通过。
- [x] 单个有效 batch 的优化器步实现，新增 4 项 CPU FP32 整批对照测试通过。

H20 启动后及主线下一步

- [ ] 核对单张 H20 的实际显存、驱动、CUDA、依赖和持久化存储。
- [ ] 正式 MATH 数据与模型已准备，baseline 入口完成替换并验证。
- [ ] 小批量生成、评分及少量 SFT 步骤通过，记录显存峰值与耗时。
- [ ] checkpoint 保存、重载、评估及训练恢复通过。
- [ ] MATH 全量 baseline 的三桶、错误案例和准确率已交付。
- [ ] SFT 数据规模与过滤实验完成，随后推进 EI、GRPO 及消融。
- [ ] 为 Leaderboard 的正式双 H100 时间预算验证另行安排资源。

可选支线

- [ ] 本机完成 5 个可选 adapter 及对应 7 项测试。
- [ ] 单张 H20 上核对 8B SFT 与 DPO 的实际可运行配置。
- [ ] 单独安排 70B 评审所需资源或确认可用方案，再补齐相关评估。
