"""vLLM 评估工具。

evaluate_vllm: 批量送 prompts 给 vllm 推理 → 对每条调用 reward_fn 判分 →
按 Problem (math_baseline)(b) 要求分桶（format + answer 都对 / format 对 answer 错 /
format 错） → 序列化结果到 disk。
"""

import json
from collections import Counter
from pathlib import Path
from typing import Callable, List, Optional

from vllm import LLM, SamplingParams


def load_vllm_model(model: str) -> LLM:
    return LLM(model)


def generate_sample_params(
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 1024,
) -> SamplingParams:
    return SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )


def _bucket(reward: dict) -> str:
    """Map a reward dict to one of three categories per PDF Problem (math_baseline)(b)."""
    f = reward.get("format_reward", 0.0)
    a = reward.get("answer_reward", 0.0)
    if f == 1.0 and a == 1.0:
        return "format_1_answer_1"
    if f == 1.0 and a == 0.0:
        return "format_1_answer_0"
    return "format_0_answer_0"


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict],
    prompts: List[str],
    ground_truths: List[str],
    eval_sampling_params: SamplingParams,
    output_path: Optional[str] = None,
) -> dict:
    """
    Args:
        vllm_model: vllm LLM 实例
        reward_fn: (response_text, ground_truth) -> {"format_reward", "answer_reward", "reward"}
        prompts: 已经渲染好的 prompt 列表
        ground_truths: 与 prompts 对齐，每个是该题的金标答案字符串
        eval_sampling_params: SamplingParams
        output_path: 若给定，把每条 (prompt, output, gt, reward) 写到 jsonl

    Returns: 评估摘要 dict，包含分桶计数和准确率。
    """
    assert len(prompts) == len(ground_truths), "prompts 和 ground_truths 长度不一致"

    outputs = vllm_model.generate(prompts, eval_sampling_params)

    records = []
    bucket_counts = Counter()
    for prompt, gt, o in zip(prompts, ground_truths, outputs):
        text = o.outputs[0].text
        reward = reward_fn(text, gt)
        bucket = _bucket(reward)
        bucket_counts[bucket] += 1
        records.append({
            "prompt": prompt,
            "output": text,
            "ground_truth": gt,
            "format_reward": reward.get("format_reward", 0.0),
            "answer_reward": reward.get("answer_reward", 0.0),
            "reward": reward.get("reward", 0.0),
            "bucket": bucket,
        })

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(records)
    summary = {
        "n": n,
        "format_1_answer_1": bucket_counts["format_1_answer_1"],
        "format_1_answer_0": bucket_counts["format_1_answer_0"],
        "format_0_answer_0": bucket_counts["format_0_answer_0"],
        "accuracy": bucket_counts["format_1_answer_1"] / n if n else 0.0,
        "format_rate": (bucket_counts["format_1_answer_1"]
                       + bucket_counts["format_1_answer_0"]) / n if n else 0.0,
    }
    print("=== eval summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return summary
