# Assignment 5 (Alignment) 个人执行计划

## 目标

完成 CS336 Assignment 5（MATH 数据集上的 SFT / Expert Iteration / GRPO），
**本机 (Mac M2) 写完所有代码 + 单测全过**，最后**租 1-2 天 GPU 集中跑训练实验**。

不做 Safety RLHF 可选部分（除非时间富余）。

---

## 策略总览

| 阶段 | 在哪里做 | 目标 | 预计时间 |
|---|---|---|---|
| 阶段 1：本机写代码 | Mac | `tests/adapters.py` 16 个 stub 全实现，所有 pytest 通过 | 主要工作量 |
| 阶段 2：本机冒烟测试 | Mac (MPS, Qwen2.5-0.5B) | 训练 loop 跑 5 步，loss 下降，checkpoint OK | 半天 |
| 阶段 3：租 GPU 跑实验 | Vast.ai / RunPod A100 80GB | 跑完 SFT / Expert Iteration / GRPO + 评估 | 20-25 GPU 小时 |
| 阶段 4：本机收尾 | Mac | 分析结果、画图、写 report | 1-2 天 |

---

## 任务拆解（按 PDF 中的 Problem 列出）

PDF 共 26 个 Problem。点数 = 评分项；标 `H100 hrs` 的是需要训练算力的实验题。

### Part 1: Zero-shot MATH Baseline（§3）

| # | Problem | 分值 | GPU? | Mac 可做? |
|---|---|---|---|---|
| 1 | `math_baseline` | 4 | 推理 | ✅ 写脚本；本机用 vllm-metal + MLX 模型抽样跑通 |

产出物：`cs336_alignment/math_baseline.py`（评估脚本 + 4 段错误案例分析 + 准确率数字）

### Part 2: SFT (§4 / Supervised Finetuning)

| # | Problem | 分值 | 实现位置 |
|---|---|---|---|
| 2 | `tokenize_prompt_and_output` | 2 | `tests/adapters.py` |
| 3 | `compute_entropy` | 1 | `tests/adapters.py` |
| 4 | `get_response_log_probs` | 2 | `tests/adapters.py` |
| 5 | `masked_normalize` | 1 | `tests/adapters.py` |
| 6 | `sft_microbatch_train_step` | 3 | `tests/adapters.py` |
| 7 | `log_generations` | 1 | `tests/adapters.py` |
| 8 | `sft_experiment` | 2 | **2 H100 hrs** — GPU 跑 SFT 完整训练 |

Mac 上做 2-7；8 上 GPU。

### Part 3: Expert Iteration (§5)

| # | Problem | 分值 | GPU? |
|---|---|---|---|
| 9 | `expert_iteration_experiment` | 2 | **6 H100 hrs** |

Mac 上写采样 / 过滤 / 迭代脚本，GPU 上跑。

### Part 4: GRPO 基础 (§6)

| # | Problem | 分值 | 实现位置 |
|---|---|---|---|
| 10 | `compute_group_normalized_rewards` | 2 | `tests/adapters.py` |
| 11 | `compute_naive_policy_gradient_loss` | 1 | `tests/adapters.py` |
| 12 | `compute_grpo_clip_loss` | 2 | `tests/adapters.py` |
| 13 | `compute_policy_gradient_loss` | 1 | `tests/adapters.py` |
| 14 | `masked_mean` | 1 | `tests/adapters.py` |
| 15 | `grpo_microbatch_train_step` | 3 | `tests/adapters.py` |
| 16 | `grpo_train_loop` | 5 | 训练 loop 脚本（不只是 adapter） |

Mac 上做 10-16；16 的实际跑通见 17-19。

### Part 5: GRPO 实验与消融 (§6 续)

| # | Problem | 分值 | GPU? |
|---|---|---|---|
| 17 | `grpo_learning_rate` | 2 | **6 H100 hrs** |
| 18 | `grpo_baselines` | 2 | **2 H100 hrs** |
| 19 | `think_about_length_normalization` | 1 | 纯思考题 |
| 20 | `grpo_length_normalization` | 2 | **2 H100 hrs** |
| 21 | `grpo_group_standard_deviation` | 2 | **2 H100 hrs** |

### Part 6: Off-policy GRPO（§7）

| # | Problem | 分值 | GPU? |
|---|---|---|---|
| 22 | `grpo_off_policy`（实现） | 待查 | Mac 写代码 |
| 23 | `grpo_off_policy_sweep` | 4 | **GPU 长** |
| 24 | `grpo_off_policy_clip_ablation` | 2 | **2 H100 hrs** |
| 25 | `grpo_prompt_ablation` | 2 | **2 H100 hrs** |

### Part 7: Leaderboard（§8）

| # | Problem | 分值 | GPU? |
|---|---|---|---|
| 26 | `leaderboard` | 16 | **16 H100 hrs** |

---

### 算力预算汇总

| 类型 | 总点数 | 总 GPU 小时 |
|---|---|---|
| 代码 + 单测题（Mac 全做） | 15 点（题号 2-7, 10-16）+ 4 点（math_baseline 写脚本部分） | 0 |
| 实验题（必须 GPU） | 36 点（题号 8, 9, 17-26） | **~46 H100 小时**（按 PDF 给的标注加总） |
| 纯思考题 | 1 点（题号 19） | 0 |

> 注：H100 小时是 PDF 中给出的参考值。A100 上一般 ×1.5-2 倍时间，所以租 A100 80GB 实际可能要 ~70-90 小时；预算从原来的 ~$30-50 上调到 **~$70-150**。

> 注：`leaderboard` 是开放性比赛题，16 H100 小时是上限，可以少投入。如果只做必修部分，跳过 22-26 也能拿到大头分数。

---

## 环境配置

### 本机（Mac，已配置好）

```sh
uv sync          # 装核心依赖，跳过 GPU-only 包
```

### GPU 机器

```sh
uv sync --extra gpu          # 装齐 flash-attn + vllm
```

`flash-attn` 和 `vllm` 已放进 `[project.optional-dependencies].gpu`，
本机不会再尝试编译。

---

## vLLM 跨平台方案（已落地）

为了让本机调试的评估 / 采样代码能直接复用到 GPU 机器，采用以下结构：

### 1. 安装

- **本机 (Mac M2)**: 安装 `vllm-metal`，独立于项目 `.venv`，装在 `~/.venv-vllm-metal/`

  ```sh
  curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
  ```

  当前装的版本：`vllm-metal 0.3.0.dev` / `vllm 0.23.0` / Python 3.12 arm64。

- **GPU 机器**: 项目自带的 `[tool.uv]` 配置，`uv sync --extra gpu` 装 `vllm==0.7.2 + flash-attn`。

### 2. 后端抽象

代码统一通过 `cs336_alignment.llm_backend.build_llm()` 构造 `LLM`：

```python
from cs336_alignment.llm_backend import build_llm
from vllm import SamplingParams

llm = build_llm("qwen2.5-math-1.5b")  # 自动按平台挑模型 + 参数
outputs = llm.generate(prompts, SamplingParams(temperature=0.7, max_tokens=1024))
```

模块负责：
- 平台检测（Darwin+arm64 → Mac，否则 → GPU）
- 模型名映射：Mac 用 `mlx-community/Qwen2.5-Math-1.5B-bf16`，GPU 用 `Qwen/Qwen2.5-Math-1.5B`
- 平台专属构造参数（GPU 上的 `gpu_memory_utilization` 等）

`LLM.generate()` API 在 vllm-metal 0.23 和 vllm 0.7.2 之间保持兼容（基础离线批量推理），所以业务代码两边完全一致。

### 3. 运行方式

```sh
# 本机用 vllm-metal 的 Python 跑（不是项目 .venv，因为 vllm-metal 不在里面）
~/.venv-vllm-metal/bin/python -m cs336_alignment.math_baseline ...

# 或者临时把 vllm-metal 加入 PATH:
export PATH="$HOME/.venv-vllm-metal/bin:$PATH"
```

### 4. 已知约束

- macOS 上 multiprocessing 用 spawn，所有调用 `LLM(...)` 的入口必须包在 `if __name__ == "__main__":` 里。
- 评估大规模数据集（MATH 5000 题）在 M2 上慢，**仅用于调试 pipeline**，正式评估上 GPU。
- `flash-attn` 无 Metal 版本，训练任务（SFT/EI/GRPO）必须 GPU，本机仅能 MPS 跑通极小冒烟。

---

## 工作流程

### 阶段 1：本机写代码（现在）

按以下顺序填 `tests/adapters.py`，每填一组立刻跑单测：

```sh
uv run pytest tests/test_metrics.py -v
uv run pytest tests/test_sft.py -v
uv run pytest tests/test_data.py -v
uv run pytest tests/test_grpo.py -v
# 可选
uv run pytest tests/test_dpo.py -v
```

**目标**：16 个 stub 全实现，所有 pytest 通过。

### 阶段 2：本机冒烟测试

用极小模型验证训练 pipeline：

```python
model_name = "Qwen/Qwen2.5-0.5B"     # 不是 1.5B-Math
device = "mps"
batch_size = 1
max_steps = 5
# 关闭 flash-attn，用 SDPA
attn_implementation = "sdpa"
```

确认：loss 下降、checkpoint 能存能读、wandb 上传 OK。

### 阶段 3：GPU 实验

**机器选择**（按性价比 → 省心度排序，价格随市场波动，更新于 2026-06）：

| 平台 | A100 80GB 单价 | H100 80GB 单价 | 优点 | 缺点 |
|---|---|---|---|---|
| **Vast.ai** | ~$0.8-1.5/h | ~$1.8-3.0/h | 最便宜，机型选择多 | P2P 市场，机器质量参差；要挑可靠性 >98%、带宽 >500Mbps |
| **RunPod Community** | ~$1.7/h | ~$2.5/h | UI 友好；Network Volume 方便；预装 PyTorch 模板 | 比 Vast 贵 50-80% |
| **RunPod Secure** | ~$2.7/h | ~$3.4/h | 同 Community + 不被抢占 | 贵 |
| **Lambda Labs** | ~$1.29/h | ~$2.49/h | 官方运营，稳定，网速好 | 经常缺货，要排队 |
| **Modal** | 按秒计费 ~$2-4/h | ~$4-6/h | 适合短任务、评估、做实验 | 不适合长训练；按秒计费贵 |
| **Together AI / Fireworks** | 不直接卖 GPU | — | 适合纯推理 | 不能跑训练 |
| **Colab Pro+** | A100 40GB only | — | 浏览器即用 | 12h 会话限制；磁盘小；性价比低 |

**给本作业的具体推荐**：

1. **首选 Vast.ai 1× A100 80GB**（~$1/h）
   - 写完代码、冒烟测试通过后再租
   - 挑可靠性 >98%、带宽 >500Mbps、剩余磁盘 >100GB
   - 跑完所有必修实验估 ~30-50 小时 → **~$30-50**
   - 风险：可能被抢占。**对策**：训练每 N 步保存 checkpoint 到 Network Volume

2. **省心备选 RunPod A100 80GB Community**（~$1.7/h）
   - 比 Vast 贵 70% 但折腾少
   - 同样工作量 → **~$50-100**

3. **如果想跑得快用 H100**
   - Lambda Labs H100 $2.49/h（缺货）
   - Vast.ai H100 $1.8-3.0/h（按 PDF 标注的 H100 小时基本 1:1）
   - 必修实验 ~14 H100 小时 → **~$25-45**

**注册建议**：
- Vast.ai 需要预充值（最低 $10）
- RunPod 需要绑卡 + 预充值
- Lambda Labs 注册后要等审核（KYC）
- 都不需要学校邮箱

**开机脚本**：
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <repo> && cd assignment5-alignment
uv sync --extra gpu
# 提前下模型避免训练时阻塞
huggingface-cli download Qwen/Qwen2.5-Math-1.5B
```

**实验顺序**（按依赖关系）：
1. MATH baseline 评估（~2h）
2. SFT 训练（~3h）→ 保存 checkpoint
3. SFT 后评估（~1h）
4. Expert Iteration（~6h）
5. GRPO 训练（~10h）→ 中途看 wandb 曲线，loss 炸了立刻 kill
6. 最终评估对比（~2h）

**总计**：~24 GPU 小时，预算 ~$30-50（Vast.ai）或 ~$60-100（RunPod）。

**注意事项**：
- 全程用 `tmux` 避免 SSH 断了训练丢
- checkpoint 写到 persistent volume，关机不丢
- 训练 wandb 监控，曲线异常立刻 kill 省钱
- 跑完所有数据/checkpoint 拉回本机再关机

### 阶段 4：本机收尾

- 拉回 wandb 数据 / checkpoint metadata
- 画对比图、写 report
- 整理交付

---

## 风险与备选

- **`flash-attn` 在 GPU 机器上仍可能编译失败**：分两步装 `uv sync` → `uv sync --extra gpu`，或手动 `uv pip install flash-attn --no-build-isolation`
- **vllm 0.7.2 与 transformers 版本可能冲突**：现已装 transformers 5.12.1，作业仅要求 `>=4.50.0`，如遇 API 不兼容 pin 一个具体版本
- **Expert Iteration / GRPO OOM**：降 batch size、开 gradient checkpointing、或换 A100 80GB
- **GPU 机器跑到一半被抢占**（Vast 低价机有这个风险）：勤存 checkpoint，挑可靠性 >98% 的机器

---

## 检查清单

本机阶段完成标志：
- [ ] `tests/adapters.py` 全部 16 个 stub 已实现
- [ ] `uv run pytest` 全绿
- [ ] Qwen2.5-0.5B 在 MPS 上跑通 5 步 SFT，loss 下降
- [ ] wandb 账户已配置，本地 dry-run 能上传

上 GPU 前的最终检查：
- [ ] 训练脚本的所有路径、超参已确认
- [ ] checkpoint 保存路径写到 persistent volume
- [ ] 评估脚本能从 checkpoint 加载并跑通
