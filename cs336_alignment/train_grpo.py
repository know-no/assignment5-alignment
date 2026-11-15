"""GRPO 基础训练入口，用同一个 HF 模型衔接生成与训练。

当前每批新生成的回答只完成一次参数更新，没有同批回答的多次更新循环。
单次更新下的概率比裁剪、梯度路径和模型模式详见
notes/grpo-clip-single-update.md。
"""

from __future__ import annotations

import random
from typing import Callable, Literal

import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from cs336_alignment.utils.log_utils import _evaluation_context
from cs336_alignment.grpo_utils import grpo_microbatch_train_step, compute_group_normalized_rewards
from cs336_alignment.utils.compute_entropy import get_response_log_probs
from cs336_alignment.grpo_utils import compute_policy_gradient_loss
from cs336_alignment.utils.tokenizer_utils import tokenize_prompt_and_output

def generate_rollouts(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt_strs: list[str],
    ground_truths: list[str],
    group_size: int,
    *,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
) -> tuple[list[str], list[str], list[str]]:
    """使用当前训练模型的权重，为每道题抽样生成一组回答。

    输入包含 P 道题，每题生成 G=group_size 条回答，总数 N=P*G。
    同一道题的 G 条回答连续排列，三个返回列表按相同行顺序对齐。
    返回顺序为 repeated_prompts、repeated_ground_truths、rollout_responses，
    长度均为 N；rollout_responses 只包含新生成的文本。

    拟调用同一个 model.generate 进行抽样，使用 temperature 和
    max_new_tokens 控制生成。生成时临时切换到 eval 并使用 torch.no_grad()，
    正常结束或发生异常后恢复各模块原来的训练模式。
    """
    device = next(model.parameters()).device
    rollout_resps = []
    with _evaluation_context(model):
        for i in range(0,len(prompt_strs)):
            prompt = prompt_strs[i]
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device) #(1, prompt_length)
            resps = []
            generated_ids = model.generate(input_ids,
                                    do_sample = True,
                                    temperature= temperature, max_new_tokens= max_new_tokens,
                                    num_return_sequences = group_size, return_dict_in_generate=False,
                                    )
            response_ids = generated_ids[:, input_ids.shape[1]:]
            responses = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
            rollout_resps.extend(responses)

    ps = [
    prompt
    for prompt in prompt_strs
    for _ in range(group_size)
    ]
    rs = [
    truth
    for truth in ground_truths
    for _ in range(group_size)
    ]
    return ps,rs , rollout_resps



def grpo_optimizer_step(
    model: PreTrainedModel,
    optimizer: torch.optim.Optimizer,
    input_ids: Tensor,
    labels: Tensor,
    response_mask: Tensor,
    raw_rewards: Tensor,
    advantages: Tensor,
    old_log_probs: Tensor | None,
    microbatch_size: int,
    *,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = "grpo_clip",
    cliprange: float = 0.2,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    """将整批 N 条回答拆成微批，完成一次梯度累积、裁剪和参数更新。

    input_ids、labels、response_mask 的形状均为 (N, L)，token 位置完全对齐。
    labels 中参与回答损失的 token 来自本轮 rollout；SFT 则使用数据集提供的标准回答。
    GRPO 的 ground_truths 交给奖励函数判分，奖励或优势决定如何调整采样回答的概率。
    raw_rewards 和 advantages 均为 (N, 1)，每条回答的权重沿 token 轴广播。
    grpo_clip 使用固定的 old_log_probs，其形状为 (N, L)，需要在更新前
    缓存并保持无梯度，与当前整批标签逐位置对齐。其余损失模式可传 None。

    N 为正且能够被 microbatch_size 整除，累积步数 K=N/microbatch_size。
    各输入仅按第 0 维切微批，保留整批编码时的 token 轴和 padding 对齐。
    复用现有 get_response_log_probs 计算当前策略的 log-prob，随后调用
    grpo_microbatch_train_step，由其完成每个微批的损失聚合与反传。
    全部 K 次反传完成后，进行一次梯度裁剪和 optimizer.step()。

    模式选择：基础版可用 eval() 并保留梯度，也可用 train() 并关闭 Dropout。
    这是可选设计说明：当前实现仍调用 train()，且不会自动关闭 Dropout；
    若采用 eval() 方案，需调整下方模式设置，仅在外层调用 eval() 会被覆盖。
    生成、旧策略评分与当前策略评分需统一 Dropout 等影响概率的前向行为，
    避免权重未变时仅因模式切换产生不同的 log-prob。
    _evaluation_context 同时包含 no_grad()，可用于生成和缓存旧分数，
    不能包住需要保留计算图的当前策略训练前向。

    返回 loss 为本批次各个已按 K 缩放的微批损失之和，
    grad_norm_before_clip 为裁剪前的全局梯度范数。
    可选 clip_fraction 按有效 response token 统计实际裁剪比例，
    分母使用整批 response_mask 的有效 token 数。
    """
    assert input_ids.shape[0] > 0 and microbatch_size > 0 and input_ids.shape[0]%microbatch_size  == 0
    losses = []
    K: int=input_ids.shape[0]//microbatch_size
    # 旧分数在 eval 下缓存；这里仍切回 train，若启用 Dropout，评分可能发生变化。
    model.train()
    optimizer.zero_grad(set_to_none=True)
    # K 个微批只累积梯度，参数在整个循环结束后的 optimizer.step() 才更新。
    for i in range(0, input_ids.shape[0], microbatch_size):
        if i + microbatch_size > input_ids.shape[0]:
            break
        micro_inputs = input_ids[i:i + microbatch_size, :]
        micro_labels = labels[i: i+ microbatch_size, :]
        micro_response_mask = response_mask[i: i + microbatch_size, :]
        micro_old_log_probs = (
            old_log_probs[i:i + microbatch_size, :]
            if old_log_probs is not None
            else None
        )
        # 当前策略的 log-prob 需要保留计算图，供微批 loss 反向传播。
        # train/eval 控制 Dropout 等模块的前向行为，eval 下也能计算梯度；
        # 此处不能放在 torch.no_grad() 中。
        _res : dict[str, torch.Tensor] = get_response_log_probs(
            model=model,
            input_ids=micro_inputs,
            labels=micro_labels,
            return_token_entropy=True,
        )
        micro_policy_log_probs = _res['log_probs'] # (B, L)
        micro_token_entropy = _res['token_entropy']

        batched_masked_loss, matadata = grpo_microbatch_train_step(
            policy_log_probs=micro_policy_log_probs,
            response_mask=micro_response_mask,
            gradient_accumulation_steps=K,
            loss_type=loss_type,
            raw_rewards=raw_rewards[i: i + microbatch_size, :],
            advantages=advantages[i: i + microbatch_size, :],
            old_log_probs=micro_old_log_probs,
            cliprange=cliprange,
        )
        losses.append(batched_masked_loss)
    grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm, norm_type=2.0)
    optimizer.step()
    return {
            "loss": sum(loss.item() for loss in losses), "grad_norm_before_clip": grad_norm_before_clip.item()
        }






def grpo_train_loop(
    model: PreTrainedModel,
    optimizer: torch.optim.Optimizer,
    tokenizer: PreTrainedTokenizerBase,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompt_strs: list[str],
    ground_truths: list[str],
    *,
    num_steps: int,
    n_prompts_per_step: int,
    group_size: int,
    microbatch_size: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = "grpo_clip",
    advantage_eps: float = 1e-6,
    normalize_by_std: bool = True,
    cliprange: float = 0.2,
    max_grad_norm: float = 1.0,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    seed: int = 42,
) -> list[dict[str, float]]:
    """抽题、生成、计算组内优势并更新模型，返回按更新顺序排列的 history。

    num_steps 表示优化器更新次数。本地初版每轮抽取 P=n_prompts_per_step
    道题，用 generate_rollouts 得到 N=P*group_size 条回答，然后执行一次
    grpo_optimizer_step。下一轮使用更新后的同一个模型生成新回答。
    后续可以在这一流程上扩展同一批 rollout 的多次复用。

    当前批次的旧分数和当前分数在参数更新前计算；若前向行为一致，
    ratio 接近 1，概率比裁剪通常不改变目标，但 ratio 的计算图仍用于反传。
    这不代表 clip_grad_norm_ 无效，它裁剪的是另一对象：累积后的参数梯度。
    详细公式、反传示例与复用扩展见 notes/grpo-clip-single-update.md。

    每轮先对完整回答组调用 compute_group_normalized_rewards。
    组内 reward/advantage 全部计算完成后，将一维权重整理成 (N, 1)，
    再用 tokenize_prompt_and_output 一次编码整批，得到对齐的 (N, L) 张量。
    在参数更新前，以临时 eval 和 torch.no_grad() 缓存旧策略的 log-prob，
    恢复模型模式后，将同一批编码、权重和旧分数交给优化器步。

    seed 用于题目抽样与生成的可复现设置。每轮 N 需满足优化器步的微批条件。
    history 每项记录该次更新的 loss、grad_norm_before_clip、原始奖励均值等
    Python float 指标，可附带按有效回答 token 统计的 clip_fraction。
    """
    assert len(prompt_strs) == len(ground_truths)
    assert n_prompts_per_step * group_size % microbatch_size == 0
    assert group_size >= 2
    rng = random.Random(seed)
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    history = []

    for step in range(num_steps):
        indices:list[int] = rng.sample(range(len(prompt_strs)), n_prompts_per_step)
        ps = [prompt_strs[i] for i in indices]
        ts = [ground_truths[i] for i in indices]
        repeated_prompts, repeated_ground_truths, rollout_responses = generate_rollouts(
            model=model, tokenizer=tokenizer,prompt_strs=ps,ground_truths=ts,
            group_size=group_size,max_new_tokens=max_new_tokens, temperature=temperature
        )

        advantages_tensor, raw_rewards_tensor, metadata =  compute_group_normalized_rewards(
            reward_fn=reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=normalize_by_std,
        )
        _res = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
        # 任一回答没有有效 token 时，masked_mean 的分母会为 0。
        # 在旧策略评分和参数更新前明确报错，避免 NaN 污染模型参数。
        if not _res["response_mask"].any(dim=-1).all().item():
            raise ValueError("rollout 回答必须至少包含一个有效 token，不能使用空回答")

        # 分词和奖励计算默认产生 CPU 张量；评分及训练均需与模型在同一设备。
        response_mask = _res["response_mask"].to(device)
        input_ids = _res["input_ids"].to(device)
        labels = _res["labels"].to(device)
        raw_rewards_tensor = raw_rewards_tensor.unsqueeze(-1).to(device)
        advantages_tensor = advantages_tensor.unsqueeze(-1).to(device)
        old_log_probs = None
        if loss_type == "grpo_clip":
            # 此时的 model 扮演旧策略；缓存数值固定，不随后续参数更新而变化。
            # _evaluation_context 同时启用 eval 和 no_grad，仅包住旧分数评分。
            with _evaluation_context(model):
                old_log_probs = get_response_log_probs(
                    model = model,
                    input_ids=input_ids,
                    labels = labels
                )['log_probs'] # (N,L), 没有梯度

        # 本轮只更新一次；下一轮会重新生成回答，并重新缓存对应的旧分数。
        _step_res = grpo_optimizer_step(
            model=model,
            optimizer=optimizer,
            input_ids=input_ids,
            labels=labels,
            response_mask=response_mask,
            raw_rewards=raw_rewards_tensor,
            advantages=advantages_tensor,
            old_log_probs=old_log_probs,
            microbatch_size=microbatch_size,
            loss_type=loss_type,
            cliprange=cliprange,
            max_grad_norm=max_grad_norm
        )
        loss = _step_res["loss"]
        grad_norm_before_clip = _step_res["grad_norm_before_clip"]
        metadata.update(_step_res)
        history.append(metadata)
    return history
