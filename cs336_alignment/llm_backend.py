"""LLM backend abstraction for Mac (vllm-metal) and GPU (vllm) platforms.

In an Apple Silicon Mac, this module assumes vllm-metal is installed in
~/.venv-vllm-metal (see PLAN.md for setup). Models from `mlx-community` are
used. On a CUDA Linux GPU machine, the standard `Qwen/...` HF-format models
are used with the official vllm.

Usage:
    from cs336_alignment.llm_backend import build_llm, SamplingParams

    llm = build_llm("qwen2.5-math-1.5b")
    outputs = llm.generate(prompts, SamplingParams(temperature=0.7, max_tokens=1024))
"""

from __future__ import annotations

import platform


MODEL_MAP: dict[str, dict[str, str]] = {
    "qwen2.5-math-1.5b": {
        "gpu": "Qwen/Qwen2.5-Math-1.5B",
        "mac": "mlx-community/Qwen2.5-Math-1.5B-bf16",
    },
    "qwen2.5-0.5b": {
        "gpu": "Qwen/Qwen2.5-0.5B-Instruct",
        "mac": "mlx-community/Qwen2.5-0.5B-Instruct-bf16",
    },
}


def get_platform() -> str:
    """Return 'mac' for Apple Silicon, 'gpu' for CUDA Linux assumed elsewhere."""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mac"
    return "gpu"


def resolve_model_name(model_key: str, plat: str | None = None) -> str:
    plat = plat or get_platform()
    if model_key not in MODEL_MAP:
        # Allow passing a raw HF / mlx-community name straight through.
        return model_key
    return MODEL_MAP[model_key][plat]


def build_llm(model_key: str, **kwargs):
    """Construct a vllm.LLM, picking the right model + platform-specific args.

    Both vllm-metal (Mac) and vllm 0.7.2 (GPU) expose the same `LLM` /
    `SamplingParams` API for basic offline batch inference, so callers can
    treat the return value uniformly.
    """
    from vllm import LLM  # type: ignore

    plat = get_platform()
    model_name = resolve_model_name(model_key, plat)

    if plat == "mac":
        # vllm-metal: omit CUDA-only args like gpu_memory_utilization.
        return LLM(model=model_name, **kwargs)

    # GPU defaults — caller may override.
    kwargs.setdefault("gpu_memory_utilization", 0.85)
    kwargs.setdefault("dtype", "bfloat16")
    return LLM(model=model_name, **kwargs)
