# CS336 Spring 2025 Assignment 5: Alignment

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment5_alignment.pdf](./cs336_spring2025_assignment5_alignment.pdf)

We include a supplemental (and completely optional) assignment on safety alignment, instruction tuning, and RLHF at [cs336_spring2025_assignment5_supplement_safety_rlhf.pdf](./cs336_spring2025_assignment5_supplement_safety_rlhf.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

As in previous assignments, we use `uv` to manage dependencies.

### GPU 机器（Linux + CUDA，跑训练/推理用）

安装包含 `flash-attn` 和 `vllm` 的完整依赖：

```sh
uv sync --extra gpu
```

如果 `flash-attn` 编译报 `ModuleNotFoundError: No module named 'setuptools'`，分两步装：

```sh
uv sync                    # 先装核心依赖（torch、setuptools 等就位）
uv sync --extra gpu        # 再装 flash-attn / vllm
```

### 本机开发（macOS / 无 GPU，仅写代码和跑单元测试）

`flash-attn` 和 `vllm` 已经放在可选依赖组 `gpu` 里，本机直接：

```sh
uv sync
```

即可装齐核心依赖（`torch` 是 CPU 版）。本机可做：

- 写代码、改 prompt
- 跑单元测试 (`uv run pytest`)
- 小批量数据 debug（用 `device="cpu"` 或 `device="mps"`）
- 数据处理、评估的非 GPU 部分

涉及到训练、vLLM 推理等需要 CUDA 的内容，统一在 GPU 机器上跑。

### 运行单元测试

```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

