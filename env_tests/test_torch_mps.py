"""PyTorch + MPS 模型加载与推理冒烟测试（不走 vllm）。

用项目 .venv 的 Python 直接运行：
    uv run python env_tests/test_torch_mps.py

验证：
- 项目 .venv（CPU 版 torch + transformers）能加载本地缓存的 HF 模型
- MPS 后端可用、能跑前向
- 简单文本生成正确

不依赖 vllm-metal，所以哪怕没装 vllm-metal 也能跑——这是开发 SFT/GRPO
代码时的最常用环境（要算 logprob / loss / 反向传播都走这条路径）。
"""

from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    device = pick_device()
    print(f"device: {device}")
    print(f"torch: {torch.__version__}")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.float32,  # MPS 上 float32 最稳；bf16 在部分算子还不支持
        attn_implementation="eager",  # SDPA 在 MPS 上对 Qwen2.5 有 shape bug
    ).to(device)
    model.eval()
    print(f"load time: {time.time() - t0:.2f}s")

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    t1 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    dt = time.time() - t1
    new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"PROMPT: {prompt}")
    print(f"OUTPUT: {text}")
    print(f"gen time: {dt:.2f}s ({new_tokens / dt:.2f} tok/s)")
    print("OK")


if __name__ == "__main__":
    main()
