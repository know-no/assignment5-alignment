"""训练循环内的采样日志工具。

`log_generations` 拿一批 prompt + ground_truth,让模型生成响应,算奖励和 token
熵,然后落盘成 JSONL,顺便返回一份聚合统计供 wandb/print 使用。

设计要点:
- 生成路径解耦:调用方传 `generate_fn: list[str] -> list[str]`,vllm 的 llm.generate
  和 HF 的 model.generate 都能包成同一签名。这样这个模块不绑定具体后端。
- token 熵走 policy 分支(HF PreTrainedModel + tokenizer),因为 vllm 不方便直接取
  逐 token 熵。若不需要,传 None 跳过即可。
- 长度按"response token 数"算,不是字符数——和训练时 loss 的分母口径一致。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from statistics import fmean
from typing import Any, Callable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from cs336_alignment.utils.compute_entropy import get_response_log_probs
from cs336_alignment.utils.tokenizer_utils import tokenize_prompt_and_output


GenerateFn = Callable[[list[str]], list[str]]
RewardFn = Callable[[str, Any], dict[str, float]]


def _safe_mean(xs: list[float]) -> float | None:
    """空 list 时返回 None,避免 statistics.mean 抛异常污染日志。"""
    return fmean(xs) if xs else None


def _per_sample_response_entropy(
    policy_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    responses: list[str],
    device: torch.device,
) -> list[float]:
    """对每条 (prompt, response),返回响应 token 的平均熵。

    做法:走一遍训练时的 tokenize/labels/mask 流程,取 token_entropy,按 response_mask
    做 dim=-1 求和后除以每条样本的 mask.sum(),得到 (B,) 的 per-sample 平均熵。
    """
    batch = tokenize_prompt_and_output(prompts, responses, tokenizer)
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    mask = batch["response_mask"].to(device).float()  # (B, L)

    was_training = policy_model.training
    policy_model.eval()
    try:
        with torch.no_grad():
            out = get_response_log_probs(
                policy_model, input_ids, labels, return_token_entropy=True
            )
    finally:
        if was_training:
            policy_model.train()

    token_entropy = out["token_entropy"]  # (B, L)
    # 逐样本平均:对 response 位置求和 / response token 数;这里必须用 mask.sum(-1)
    # 当分母(每条真实响应长度),而不是训练时的 fixed normalize_constant——因为这里
    # 目的是"日志可读的平均熵",不是反传的 loss。
    per_sample_sum = (token_entropy * mask).sum(dim=-1)  # (B,)
    per_sample_count = mask.sum(dim=-1).clamp_min(1.0)  # 防 0 除
    per_sample_mean = (per_sample_sum / per_sample_count).detach().cpu().tolist()
    return per_sample_mean


def log_generations(
    prompts: list[str],
    ground_truths: list[Any],
    generate_fn: GenerateFn,
    reward_fn: RewardFn,
    *,
    policy_model: PreTrainedModel | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
    device: torch.device | None = None,
    log_path: str | os.PathLike | None = None,
    step: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """在训练循环里对一批 prompt 做采样并记录。

    Args:
        prompts: 送给模型的 prompt(通常已经套好模板)。
        ground_truths: 与 prompts 一一对应的真值答案;类型由 reward_fn 决定。
        generate_fn: `list[str] -> list[str]`,返回每条 prompt 对应的完整响应字符串。
        reward_fn: `(response, ground_truth) -> {"format_reward","answer_reward","reward"}`。
            例如 `cs336_alignment.drgrpo_grader.r1_zero_reward_fn`。
        policy_model, tokenizer: 若同时提供,则计算响应 token 的平均熵;否则跳过。
        device: policy_model 所在设备;不填时自动从模型参数推断。
        log_path: JSONL 追加写入的路径;None 表示不落盘。
        step: 训练步数,写进每条记录方便事后按 step 过滤。
        extra: 会合并到 summary 里,常用于附加 epoch/lr 等元信息。

    Returns:
        {
            "records": [ {prompt, response, ground_truth, reward, format_reward,
                          answer_reward, response_len_tokens, mean_token_entropy}, ... ],
            "summary": {
                "n": int,
                "mean_reward", "mean_format_reward", "mean_answer_reward",
                "mean_response_len", "mean_correct_len", "mean_wrong_len",
                "mean_token_entropy",  # 可能为 None
                "step": ...,
            },
        }
    """
    assert len(prompts) == len(ground_truths), "prompts 与 ground_truths 长度必须一致"
    if not prompts:
        return {"records": [], "summary": {"n": 0, "step": step}}

    responses = generate_fn(prompts)
    assert len(responses) == len(prompts), (
        f"generate_fn 返回数量不匹配: got {len(responses)}, expected {len(prompts)}"
    )

    # 逐条算 reward
    reward_dicts = [reward_fn(r, gt) for r, gt in zip(responses, ground_truths)]

    # 响应 token 数(不是字符数):可选用 tokenizer;没 tokenizer 时退化成词数
    if tokenizer is not None:
        response_lens = [
            len(tokenizer.encode(r, add_special_tokens=False)) for r in responses
        ]
    else:
        response_lens = [len(r.split()) for r in responses]

    # 平均 token 熵(可选)
    mean_entropies: list[float | None]
    if policy_model is not None and tokenizer is not None:
        dev = device or next(policy_model.parameters()).device
        mean_entropies = _per_sample_response_entropy(  # type: ignore[assignment]
            policy_model, tokenizer, prompts, responses, dev
        )
    else:
        mean_entropies = [None] * len(prompts)

    # 打包 records
    records: list[dict[str, Any]] = []
    for prompt, resp, gt, rd, rlen, ent in zip(
        prompts, responses, ground_truths, reward_dicts, response_lens, mean_entropies
    ):
        records.append(
            {
                "step": step,
                "prompt": prompt,
                "response": resp,
                "ground_truth": gt,
                "format_reward": rd.get("format_reward"),
                "answer_reward": rd.get("answer_reward"),
                "reward": rd.get("reward"),
                "response_len_tokens": rlen,
                "mean_token_entropy": ent,
            }
        )

    # 长度分桶:按 answer_reward==1 判定"正确"。这里用 answer_reward 而不是 reward
    # 是因为 r1_zero 里 reward 会因为格式错而清零,但 answer_reward 更直接反映"答对没"。
    correct_lens = [
        r["response_len_tokens"] for r in records if r["answer_reward"] == 1.0
    ]
    wrong_lens = [
        r["response_len_tokens"] for r in records if r["answer_reward"] != 1.0
    ]

    summary: dict[str, Any] = {
        "n": len(records),
        "step": step,
        "mean_reward": _safe_mean([r["reward"] for r in records]),
        "mean_format_reward": _safe_mean([r["format_reward"] for r in records]),
        "mean_answer_reward": _safe_mean([r["answer_reward"] for r in records]),
        "mean_response_len": _safe_mean(response_lens),
        "mean_correct_len": _safe_mean(correct_lens),
        "mean_wrong_len": _safe_mean(wrong_lens),
        "mean_token_entropy": _safe_mean([e for e in mean_entropies if e is not None]),
    }
    if extra:
        summary.update(extra)

    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return {"records": records, "summary": summary}
