import math
from typing import Callable, Literal

import torch
import statistics

def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """对每条完整回答打分，并在同一道题的回答组内计算 advantage。

    reward_fn 接收 response 和 ground_truth，返回含 reward 的评分字典。
    两个输入列表等长，同一道题的 group_size 条回答连续排列。
    normalize_by_std 控制是否在减去组内均值后，再除以组内标准差加
    advantage_eps。

    一条完整回答对应一个 reward 和一个 advantage，返回张量均为一维。
    调用方选出当前微批的 B 条回答后，将权重从 (B,) 变成 (B, 1)，
    使同一回答的权重可以沿 token 轴广播给 (B, L) 的 log-prob。

    Returns:
        advantages: FP32 张量，形状 (rollout_batch_size,)，组内相对优势。
        raw_rewards: FP32 张量，形状 (rollout_batch_size,)，逐条原始奖励。
        metadata: 日志统计字典，各统计值使用 Python float。
    """

    assert len(rollout_responses) == len(repeated_ground_truths)

    raw_rewards = []
    advantages = []
    for i in range(0, len(rollout_responses), group_size):
        rollout = rollout_responses[i:i+group_size]
        truth = repeated_ground_truths[i:i + group_size]
        reward = []
        for j in range(0, group_size):
            rewards: dict[str, float] = reward_fn(rollout[j], truth[j])
            reward.append(rewards['reward'])
        raw_rewards.append(reward)
        avg = sum(reward) / group_size
        norm = (statistics.stdev(reward) + advantage_eps) if normalize_by_std else 1.0;
        advantage = [ (r - avg) / norm for r in reward]
        advantages.append(advantage)


    raw_rewards_tensor = torch.tensor(
        raw_rewards, dtype=torch.float32
    ).flatten()

    advantages_tensor = torch.tensor(
     advantages, dtype=torch.float32
    ).flatten()
    metadata = {
    "reward_mean": raw_rewards_tensor.mean().item(),
    "reward_min": raw_rewards_tensor.min().item(),
    "reward_max": raw_rewards_tensor.max().item(),
    }


    return advantages_tensor, raw_rewards_tensor, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor, #(B, 1)
    policy_log_probs: torch.Tensor, #(B, L)
) -> torch.Tensor:
    """返回每个 token 的策略梯度损失贡献，形状为 (B, L)。

    raw_rewards_or_advantages 为 (B, 1)，每条回答共享一个权重 A；
    policy_log_probs 为 (B, L)，每个位置保存对应标签 token 的 log-prob，
    后续 response_mask 标记有效回答位置。对同一回答的有效 token 固定 A 时，
    token 项求和为 -A * sum_t(log p_t)，
    也就是 -A * log p(完整回答 | prompt)。当前微批训练步随后按
    response_mask 选取有效位置，并对每条回答的有效长度取平均。
    """
    # log_probs = policy_log_probs.sum(dim = -1, keepdim=True)
    # loss = - raw_rewards_or_advantages * log_probs # (B,1)
    # return loss
    # 同一回答权重固定时，token 项求和可用上面的序列写法。
    # 若使用上面的序列求和写法，应在 sum 前先应用 response_mask；当前接口返回 token 项，后续再 mask 和平均。

    return - raw_rewards_or_advantages * policy_log_probs

def compute_grpo_clip_loss(
    advantages: torch.Tensor,        # (B, 1)
    policy_log_probs: torch.Tensor,  # (B, L)
    old_log_probs: torch.Tensor,     # (B, L)
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """计算逐 token 的 GRPO-Clip 损失贡献，返回形状为 (B, L) 的张量。

    新旧 log-prob 都按相同 prompt 和回答前缀计算。p = exp(new-old)
    是每个 token 的概率比，形状为 (B, L)。advantages 为 (B, 1)，
    因而同一完整回答的 A 在该回答的各个 token 上共享。
    每个 token 分别计算 ratio*A 与 clipped_ratio*A，取较小值并加负号。
    后续将这些 token 项聚合，得到每条回答及整个微批的损失。

    这里逐 token 做 ratio/clipping，与先对 log-prob 差求和、
    形成整序列概率比后再 clip，对应不同的目标。

    同批回答的首次更新前，若新旧评分的前向行为一致，ratio 接近 1，
    clamp 通常不改变目标。复用该批时，旧分数保持固定，当前分数随参数
    更新重新计算，后续的 ratio 可能偏离 1 并触发裁剪。
    仍不能将 ratio 直接改为常数 1：exp(new-old) 对 new 的导数为 ratio，
    即使其数值为 1 仍能传递梯度；常数 1 则切断这条策略梯度路径。
    old 必须固定且无梯度，避免 exp(new-new) 的两条求导路径相互抵消。
    详见 notes/grpo-clip-single-update.md。
    """
    # p 保存逐 token 的新旧概率比，每个位置分别进行裁剪。
    p = (policy_log_probs - old_log_probs).exp()

    # 先乘共享的 A，再比较两项，兼顾正 advantage 和负 advantage 的情况。
    u = p * advantages
    c = torch.clamp(p, 1- cliprange, 1+ cliprange) * advantages

    loss = torch.where(u < c, u, c) * -1
    # loss = torch.minimum(u,c) * -1

    metadata = {
        # c < u 表示裁剪项严格更小，裁剪实际改变了该 token 的目标贡献。
        "clipped": c < u
    }
    return loss, metadata

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """按 loss_type 选择逐 token 损失，返回 (B, L) 的贡献和日志 metadata。

    raw_rewards 或 advantages 提供每条回答共享的 (B, 1) 权重，
    返回值交给微批训练步沿 token 轴和 batch 轴聚合。
    """
    loss = None
    metadata = {}
    if loss_type == "no_baseline":
        loss =  compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=raw_rewards,
            policy_log_probs=policy_log_probs,
            )

    elif loss_type == "reinforce_with_baseline":
        loss= compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=advantages,
            policy_log_probs=policy_log_probs,
        )
    elif loss_type == "grpo_clip":
        loss, metadata = compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )
    else:
        raise Exception(f"未知 grpo loss 计算类型: {loss_type}")
    return loss, metadata

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """沿指定维度对 mask 选中的有效元素求平均。

    tensor 和 mask 为 (B, L) 时，dim=-1 只沿 token 轴归一化，
    返回 (B,) 的逐回答均值；dim=None 对全部有效 token 求一个标量均值。
    """
    numerator = (tensor * mask).sum(dim = dim)
    denominator = mask.sum(dim = dim)
    return numerator / denominator

# 这里并没有限定微批次, microbatch_size和group_size的大小，即答案在计算完了优势函数之后，就独立了
def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    *,
    reference_log_probs: torch.Tensor | None = None,
    kl_beta: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """聚合逐 token 贡献，并对当前微批的标量损失执行反向传播。

    B 为当前微批的回答数，K 为 gradient_accumulation_steps，
    T_b 为第 b 条回答的有效 token 数。本函数计算
        L = sum_b[sum_t(mask_bt * token_loss_bt) / T_b] / (B * K)
    即先得到每条回答的平均 token 损失，再对 B 条回答取平均并除以 K。
    外层使用等大微批时，除以 K 后累积对应这些微批的平均损失梯度。

    kl_beta > 0 时，在任一策略损失上加 KL 正则。reference_log_probs 是同一批
    token 的固定参考分数，形状与 policy_log_probs 一致。对采样 token 取
    d=log p_ref-log p_current，KL 估计为 exp(d)-d-1，沿相同的 mask 和 K 归约。
    metadata["kl_loss"] 是未乘 kl_beta 的微批贡献，用于外层汇总日志。

    backward() 沿计算图将梯度累加到模型参数。输入为叶子张量或已调用
    retain_grad() 时，也可在上游 policy_log_probs.grad 查看梯度。
    外层训练循环负责 optimizer 更新。返回的损失已 detach，用于日志记录。
    """

    if not math.isfinite(kl_beta) or kl_beta < 0:
        raise ValueError("kl_beta 必须为有限非负数")

    # 策略梯度权重来自完整回答的评分，底层仍返回 (B, L) 的 token 贡献。
    loss,metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    # 沿 token 轴平均得到 (B,)；例如两条回答的均值为 [3, 6]。 这里是把一条回答的逐token的 loss 都相加了求平均了，得到一条回答的 loss；和之前我遇到的 grpo 计算不同，但是问题不大
    masked_loss = masked_mean(loss, mask=response_mask, dim = -1)
    # 对 batch 取 mean 才得到标量。例中 mean=4.5，K=2 后反传 loss=2.25。
    # 除以 K 只缩放数值，batch 维的归约由 mean 完成。
    batched_masked_loss = masked_loss.mean() / gradient_accumulation_steps
    if kl_beta > 0:
        if reference_log_probs is None or reference_log_probs.shape != policy_log_probs.shape:
            raise ValueError("启用 KL 需要与 policy_log_probs 形状一致的 reference_log_probs")
        # KL 标记，DeepSeekMath 式 (4) 的 k3 估计，d=log(ref/current)，k3=exp(d)-1-d。
        # 在旧 rollout 的 token 上直接加入惩罚，不乘 current/old 概率比。
        # 相同前缀下，对当前策略分布取期望才还原标准 KL(current || reference)。
        # 详见 notes/grpo-clip-single-update.md §3 的 KL 说明。
        log_ratio = reference_log_probs.detach() - policy_log_probs
        # exp 前屏蔽 prompt/padding，避免无效位置溢出后污染归约。
        log_ratio = log_ratio.masked_fill(~response_mask.bool(), 0.0)
        token_kl = torch.expm1(log_ratio) - log_ratio
        kl_loss = masked_mean(token_kl, mask=response_mask, dim=-1).mean() / gradient_accumulation_steps
        batched_masked_loss = batched_masked_loss + kl_beta * kl_loss
        metadata["kl_loss"] = kl_loss.detach()

    batched_masked_loss.backward()
    return batched_masked_loss.detach(), metadata
