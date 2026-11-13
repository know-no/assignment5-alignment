"""训练循环内的采样日志工具。

`log_generations` 拿一批 prompt + ground_truth,让模型生成响应,算奖励和 token
熵,然后落盘成 JSONL,顺便返回一份聚合统计供 wandb/print 使用。

设计要点:
- 生成路径解耦:调用方传 `generate_fn: list[str] -> list[str]`,vllm 的 llm.generate
  和 HF 的 model.generate 都能包成同一签名。这样这个模块不绑定具体后端。
- token 熵走 policy 分支(HF PreTrainedModel + tokenizer),因为 vllm 不方便直接取
  逐 token 熵。若不需要,传 None 跳过即可。
- 传入 tokenizer 时，长度统计采用 response token 数。省略时保留按空格分词的近似长度。
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Iterator

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from cs336_alignment.utils.compute_entropy import get_response_log_probs
from cs336_alignment.utils.tokenizer_utils import tokenize_prompt_and_output


GenerateFn = Callable[[list[str]], list[str]]
RewardFn = Callable[[str, Any], dict[str, float]]


def _safe_mean(xs: list[float]) -> float | None:
    """空 list 时返回 None,避免 statistics.mean 抛异常污染日志。"""
    return fmean(xs) if xs else None


@contextmanager
def _evaluation_context(model: PreTrainedModel | None) -> Iterator[None]:
    """在 no_grad 下临时评估，并恢复模型各子模块原有的 training 标志。"""
    training_states: list[tuple[torch.nn.Module, bool]] = []
    if model is not None:
        training_states = [(module, module.training) for module in model.modules()]
    try:
        if model is not None:
            model.eval()
        with torch.no_grad():
            yield
    finally:
        for module, was_training in training_states:
            module.training = was_training


def _per_sample_response_entropy(
    policy_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    responses: list[str],
    device: torch.device | None = None,
) -> list[float | None]:
    """对每条 (prompt, response) 返回响应 token 的平均熵，空响应返回 None。

    做法:走一遍训练时的 tokenize/labels/mask 流程,取 token_entropy,按 response_mask
    做 dim=-1 求和后除以每条样本的 mask.sum()。有响应 token 的行一起送入模型，
    结果映射回原批次顺序。整批响应都为空时直接返回空值列表。
    """
    batch = tokenize_prompt_and_output(prompts, responses, tokenizer)
    has_response = batch["response_mask"].any(dim=-1)
    response_rows = has_response.nonzero(as_tuple=True)[0].tolist()
    mean_entropies: list[float | None] = [None] * len(prompts)
    if not response_rows:
        return mean_entropies

    device = device or next(policy_model.parameters()).device
    input_ids = batch["input_ids"][has_response].to(device)
    labels = batch["labels"][has_response].to(device)
    mask = batch["response_mask"][has_response].to(device).float()
    with _evaluation_context(policy_model):
        out = get_response_log_probs(
            policy_model, input_ids, labels, return_token_entropy=True
        )

    token_entropy = out["token_entropy"]  # (R, L)，R 为有响应 token 的样本数
    # 每条响应的 token 熵求和，再除以该行的响应 token 数，得到 nats/token。
    per_sample_sum = (token_entropy * mask).sum(dim=-1)  # (R,)
    per_sample_count = mask.sum(dim=-1)  # 参与前向的每一行至少有一个响应 token
    per_sample_mean = (per_sample_sum / per_sample_count).detach().cpu().tolist()
    for row, entropy in zip(response_rows, per_sample_mean):
        mean_entropies[row] = entropy
    return mean_entropies


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
            回调在 no_grad 下运行，传入的 policy_model 会临时设为 eval。
            若回调使用另一个独立模型，回调负责该模型自己的推理状态。
        reward_fn: `(response, ground_truth) -> {"format_reward","answer_reward","reward"}`。
            例如 `cs336_alignment.drgrpo_grader.r1_zero_reward_fn`。
        policy_model, tokenizer: 若同时提供,则计算响应 token 的平均熵;否则跳过。
            空响应的熵为 None，整批平均熵排除这些空值。
        tokenizer: 传入时用 add_special_tokens=False 统计真实响应 token 数。
            省略时 response_len_tokens 保留按空格分词的近似长度。
        device: policy_model 所在设备;不填时自动从模型参数推断。
        log_path: JSONL 追加写入的路径;None 表示不落盘。
        step: 训练步数,写进每条记录方便事后按 step 过滤。
        extra: 会合并到 summary 里，常用于附加 epoch/lr 等元信息。
            与核心统计字段重名时抛出 ValueError。

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
    summary: dict[str, Any] = {
        "n": len(prompts),
        "step": step,
        "mean_reward": None,
        "mean_format_reward": None,
        "mean_answer_reward": None,
        "mean_response_len": None,
        "mean_correct_len": None,
        "mean_wrong_len": None,
        "mean_token_entropy": None,
    }
    if extra:
        conflicting_keys = summary.keys() & extra.keys()
        if conflicting_keys:
            raise ValueError(
                f"extra 不能覆盖 summary 核心字段 {', '.join(sorted(conflicting_keys))}"
            )
        summary.update(extra)
    if not prompts:
        return {"records": [], "summary": summary}

    with _evaluation_context(policy_model):
        responses = generate_fn(prompts)
    assert len(responses) == len(prompts), (
        f"generate_fn 返回数量不匹配: got {len(responses)}, expected {len(prompts)}"
    )

    # 逐条算 reward
    reward_dicts = [reward_fn(r, gt) for r, gt in zip(responses, ground_truths)]

    # 传入 tokenizer 时统计真实 token 数，省略时保留按空格分词的近似长度。
    if tokenizer is not None:
        response_lens = [
            len(tokenizer.encode(r, add_special_tokens=False)) for r in responses
        ]
    else:
        response_lens = [len(r.split()) for r in responses]

    # 平均 token 熵(可选)
    mean_entropies: list[float | None]
    if policy_model is not None and tokenizer is not None:
        mean_entropies = _per_sample_response_entropy(
            policy_model, tokenizer, prompts, responses, device
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

    # 按奖励函数的 answer_reward 判定成功。当前 r1_zero_reward_fn 中，
    # answer_reward 与 reward 始终相同，格式错误的回答也进入失败桶。
    correct_lens = [
        r["response_len_tokens"] for r in records if r["answer_reward"] == 1.0
    ]
    wrong_lens = [
        r["response_len_tokens"] for r in records if r["answer_reward"] != 1.0
    ]

    summary.update(
        {
            "mean_reward": _safe_mean([r["reward"] for r in records]),
            "mean_format_reward": _safe_mean([r["format_reward"] for r in records]),
            "mean_answer_reward": _safe_mean([r["answer_reward"] for r in records]),
            "mean_response_len": _safe_mean(response_lens),
            "mean_correct_len": _safe_mean(correct_lens),
            "mean_wrong_len": _safe_mean(wrong_lens),
            "mean_token_entropy": _safe_mean([e for e in mean_entropies if e is not None]),
        }
    )

    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return {"records": records, "summary": summary}
