"""GRPO 基础训练入口，用同一个 HF 模型衔接生成与训练。

每批回答按 train_batch_size 切片，每个 epoch 顺序遍历全部回答。
默认整批训练一次，可通过 epochs_per_rollout_batch 复用。
首次更新时的概率比、梯度路径和模型模式详见
notes/grpo-clip-single-update.md。
"""

from __future__ import annotations

import math
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
    reference_log_probs: Tensor | None = None,
    kl_beta: float = 0.0,
) -> dict[str, float]:
    """将整批 N 条回答拆成微批，完成一次梯度累积、裁剪和参数更新。

    input_ids、labels、response_mask 的形状均为 (N, L)，token 位置完全对齐。
    labels 中参与回答损失的 token 来自本轮 rollout；SFT 则使用数据集提供的标准回答。
    GRPO 的 ground_truths 交给奖励函数判分，奖励或优势决定如何调整采样回答的概率。
    raw_rewards 和 advantages 均为 (N, 1)，每条回答的权重沿 token 轴广播。
    grpo_clip 使用固定的 old_log_probs，其形状为 (N, L)，需要在更新前
    缓存并保持无梯度，与当前整批标签逐位置对齐。其余损失模式可传 None。
    reference_log_probs 是独立固定参考策略对相同标签的评分，形状为 (N, L)，
    不承担 old 策略的概率比分母角色。kl_beta 默认 0，完全禁用参考评分项。
    正值时所有 loss 分支均加入 KL，参考分数须已位于输入所在设备。

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
    启用 KL 时 loss 包含加权 KL，另返回未乘 kl_beta 的 kl_loss，
    由各微批已除以 K 的 KL 分量相加得到。
    可选 clip_fraction 按有效 response token 统计实际裁剪比例，
    分母使用整批 response_mask 的有效 token 数。
    """
    if not math.isfinite(kl_beta) or kl_beta < 0:
        raise ValueError("kl_beta 必须为非负有限数")
    if kl_beta > 0:
        if not isinstance(reference_log_probs, Tensor) or reference_log_probs.shape != input_ids.shape:
            raise ValueError("正 kl_beta 需要与 input_ids 完整对齐的 reference_log_probs")
        if reference_log_probs.device != input_ids.device:
            raise ValueError("reference_log_probs 必须与 input_ids 位于同一设备")
    assert input_ids.shape[0] > 0 and microbatch_size > 0 and input_ids.shape[0]%microbatch_size  == 0
    losses = []
    kl_losses = []
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
        micro_reference_log_probs = (
            reference_log_probs[i:i + microbatch_size, :]
            if kl_beta > 0 and reference_log_probs is not None
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
            reference_log_probs=micro_reference_log_probs,
            kl_beta=kl_beta,
        )
        losses.append(batched_masked_loss)
        if kl_beta > 0:
            kl_losses.append(matadata["kl_loss"])
    grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm, norm_type=2.0)
    optimizer.step()
    result = {
        "loss": sum(loss.item() for loss in losses),
        "grad_norm_before_clip": grad_norm_before_clip.item(),
    }
    if kl_beta > 0:
        result["kl_loss"] = sum(loss.item() for loss in kl_losses)
    return result






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
    epochs_per_rollout_batch: int = 1,
    train_batch_size: int | None = None,
    reference_model: PreTrainedModel | None = None,
    kl_beta: float = 0.0,
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

    每批抽取 P=n_prompts_per_step 道题，生成 N=P*group_size 条回答。
    在完整回答组上计算 reward 和 advantage，将权重整理成 (N, 1)，
    再一次性编码整批，得到对齐的 (N, L) 张量。seed 用于题目抽样和生成。

    epochs_per_rollout_batch 控制同一批回答遍历几轮，默认 1。
    每轮从第 0 行顺序切 train_batch_size 条回答，默认 train_batch_size=N。
    每个切片调用一次 grpo_optimizer_step，内部按 microbatch_size 累积反传后更新。
    N 需能被 train_batch_size 整除，train_batch_size 需能被 microbatch_size 整除。
    完整一轮产生 N/train_batch_size 次更新，每个微批均用当前参数重新前向。
    num_steps 是总更新次数，最后一批或最后一轮按剩余额度提前结束。
    下一批回答由更新后的模型重新生成。

    同批复用期间，回答、reward 和 advantage 保持固定。no_baseline 与
    reinforce_with_baseline 在单次更新和同批复用时使用同一个公式，分别用
    reward 或 advantage 加权当前 log-prob，策略项均不使用 current/old 概率比。
    当前策略变化后，这些回答仍来自生成时的策略，属于 off-policy 数据，
    这两个损失分支直接使用这些回答，未对采样分布的变化做概率比修正。

    grpo_clip 在首次更新前用 eval 和 torch.no_grad() 缓存该批的 old_log_probs，
    整个复用期间保持旧分数固定。若新旧评分的前向行为一致，
    首次更新前 ratio 接近 1，仍通过当前分数反传。参数更新后，后续更新的
    ratio 可能偏离 1，裁剪可能开始影响部分 token 的损失。每次优化器步还会
    对累积的参数梯度做范数裁剪。公式见 notes/grpo-clip-single-update.md。

    kl_beta 默认为 0，完全跳过 reference_model。正值时调用方须提供固定、
    参数独立的参考模型，并保证它与当前策略使用相同 token-ID 空间。
    reference 用于 KL，old 用于本批 rollout 的概率比，两种角色分别缓存。
    每批在 reference 自身设备上以临时 eval/no_grad 完整评分一次，随后将
    参考分数移到训练设备，同批所有切片和 epoch 复用，参考模型本身不迁移。
    这里沿用普通 HF 模型所有参数位于单一设备的使用方式。

    history 每项独立记录一个训练切片更新的 loss、grad_norm_before_clip，
    原始奖励统计沿用该切片所属完整 rollout 批次的统计。
    启用 KL 时 loss 是总损失，并额外记录未乘 kl_beta 的 kl_loss。
    """
    assert len(prompt_strs) == len(ground_truths)
    rollout_batch_size = n_prompts_per_step * group_size
    if type(rollout_batch_size) is not int or rollout_batch_size <= 0:
        raise ValueError("rollout_batch_size 必须为正整数")
    if train_batch_size is None:
        train_batch_size = rollout_batch_size
    for name, value in (
        ("epochs_per_rollout_batch", epochs_per_rollout_batch),
        ("microbatch_size", microbatch_size),
        ("train_batch_size", train_batch_size),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} 必须为正整数")
    if rollout_batch_size % train_batch_size != 0:
        raise ValueError("rollout_batch_size 必须能被 train_batch_size 整除")
    if train_batch_size % microbatch_size != 0:
        raise ValueError("train_batch_size 必须能被 microbatch_size 整除")
    if not math.isfinite(kl_beta) or kl_beta < 0:
        raise ValueError("kl_beta 必须为非负有限数")
    if kl_beta > 0:
        if reference_model is None:
            raise ValueError("正 kl_beta 需要独立的 reference_model")
        if reference_model is model:
            raise ValueError("reference_model 必须与当前策略独立，不能使用同一个模型")
        policy_parameter_ids = {id(parameter) for parameter in model.parameters()}
        if any(id(parameter) in policy_parameter_ids for parameter in reference_model.parameters()):
            raise ValueError("reference_model 不能与当前策略共享 Parameter")
    assert group_size >= 2
    rng = random.Random(seed)
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    history = []

    completed_updates = 0
    while completed_updates < num_steps:
        # 还是抽样
        indices:list[int] = rng.sample(range(len(prompt_strs)), n_prompts_per_step)
        ps = [prompt_strs[i] for i in indices]
        ts = [ground_truths[i] for i in indices]
        repeated_prompts, repeated_ground_truths, rollout_responses = generate_rollouts(
            model=model, tokenizer=tokenizer,prompt_strs=ps,ground_truths=ts,
            group_size=group_size,max_new_tokens=max_new_tokens, temperature=temperature
        )
        # 计算出 reward 和 advantage
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

        # reference 对同一批已经生成的 labels 评分一次，跨训练切片与 epoch 复用。
        # old 缓存生成策略的概率比分母，reference 提供固定 KL 参照，两者角色不同。
        reference_log_probs = None
        if kl_beta > 0 and reference_model is not None:
            reference_device = next(reference_model.parameters()).device
            with _evaluation_context(reference_model):
                reference_log_probs = get_response_log_probs(
                    model=reference_model,
                    input_ids=input_ids.to(reference_device),
                    labels=labels.to(reference_device),
                )["log_probs"].to(device)

        # 每个 epoch 都从第 0 行开始切 train batch，复用同一批已生成的回答。
        # 奖励、优势和已经缓存的旧分数、参考分数沿用本批的值，每步用当前参数重新评分。
        # 策略一旦改变，继续用这批旧回答更新就属于 off-policy 训练。
        for _ in range(epochs_per_rollout_batch):
            if completed_updates >= num_steps:
                break
            for start in range(0, input_ids.shape[0], train_batch_size):
                if completed_updates >= num_steps:  # 总步数用完就停，即使当前 epoch 还没走完。
                    break
                end = start + train_batch_size
                rows = slice(start, end)
                _step_res = grpo_optimizer_step(
                    model=model,
                    optimizer=optimizer,
                    input_ids=input_ids[rows],
                    labels=labels[rows],
                    response_mask=response_mask[rows],
                    raw_rewards=raw_rewards_tensor[rows],
                    advantages=advantages_tensor[rows],
                    old_log_probs=old_log_probs[rows] if old_log_probs is not None else None,
                    microbatch_size=microbatch_size,
                    loss_type=loss_type,
                    cliprange=cliprange,
                    max_grad_norm=max_grad_norm,
                    reference_log_probs=(
                        reference_log_probs[rows] if reference_log_probs is not None else None
                    ),
                    kl_beta=kl_beta,
                )
                # 分别留出中间值，方便回看和断点检查。
                loss = _step_res["loss"]
                grad_norm_before_clip = _step_res["grad_norm_before_clip"]
                history.append({**metadata, **_step_res})
                completed_updates += 1
    return history
