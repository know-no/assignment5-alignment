import math

import torch

from cs336_alignment import grpo_utils

from .adapters import (
    run_compute_group_normalized_rewards as compute_group_normalized_rewards,
    run_compute_grpo_clip_loss as compute_grpo_clip_loss,
    run_compute_naive_policy_gradient_loss as compute_naive_policy_gradient_loss,
    run_compute_policy_gradient_loss as compute_policy_gradient_loss,
    run_grpo_microbatch_train_step as grpo_microbatch_train_step,
    run_masked_mean as masked_mean,
)


def test_compute_group_normalized_rewards_normalize_by_std(
    numpy_snapshot,
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    advantage_eps,
    group_size,
):
    normalized_rewards, raw_rewards, metadata = compute_group_normalized_rewards(
        reward_fn=reward_fn,
        rollout_responses=rollout_responses,
        repeated_ground_truths=repeated_ground_truths,
        group_size=group_size,
        advantage_eps=advantage_eps,
        normalize_by_std=True,
    )
    output = {
        "normalized_rewards": normalized_rewards,
        "raw_rewards": raw_rewards,
    }
    numpy_snapshot.assert_match(output)


def test_compute_group_normalized_rewards_no_normalize_by_std(
    numpy_snapshot,
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    advantage_eps,
    group_size,
):
    normalized_rewards, raw_rewards, metadata = compute_group_normalized_rewards(
        reward_fn=reward_fn,
        rollout_responses=rollout_responses,
        repeated_ground_truths=repeated_ground_truths,
        group_size=group_size,
        advantage_eps=advantage_eps,
        normalize_by_std=False,
    )
    output = {
        "normalized_rewards": normalized_rewards,
        "raw_rewards": raw_rewards,
    }
    numpy_snapshot.assert_match(output)




def test_compute_naive_policy_gradient_loss(
    numpy_snapshot,
    raw_rewards_or_advantages,
    policy_log_probs,
):
    output = compute_naive_policy_gradient_loss(
        raw_rewards_or_advantages=raw_rewards_or_advantages,
        policy_log_probs=policy_log_probs,
    )
    numpy_snapshot.assert_match(output)


def test_compute_grpo_clip_loss_large_cliprange(
    numpy_snapshot,
    advantages,
    policy_log_probs,
    old_log_probs,
):
    output, _ = compute_grpo_clip_loss(
        advantages=advantages,
        policy_log_probs=policy_log_probs,
        old_log_probs=old_log_probs,
        cliprange=10.0,
    )
    numpy_snapshot.assert_match(output)


def test_compute_grpo_clip_loss_small_cliprange(
    numpy_snapshot,
    advantages,
    policy_log_probs,
    old_log_probs,
):
    output, _ = compute_grpo_clip_loss(
        advantages=advantages,
        policy_log_probs=policy_log_probs,
        old_log_probs=old_log_probs,
        cliprange=0.1,
    )
    numpy_snapshot.assert_match(output)


def test_compute_policy_gradient_loss_no_baseline(
    numpy_snapshot,
    policy_log_probs,
    raw_rewards,
    advantages,
    old_log_probs,
):
    output, _ = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type="no_baseline",
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=0.5,
    )
    numpy_snapshot.assert_match(output)


def test_compute_policy_gradient_loss_reinforce_with_baseline(
    numpy_snapshot,
    policy_log_probs,
    raw_rewards,
    advantages,
    old_log_probs,
):
    output, _ = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type="reinforce_with_baseline",
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=0.5,
    )
    numpy_snapshot.assert_match(output)


def test_compute_policy_gradient_loss_grpo_clip(
    numpy_snapshot,
    policy_log_probs,
    raw_rewards,
    advantages,
    old_log_probs,
):
    output, _ = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type="grpo_clip",
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=0.5,
    )
    numpy_snapshot.assert_match(output)


def test_masked_mean_dim0(numpy_snapshot, tensor, mask):
    output = masked_mean(
        tensor=tensor,
        mask=mask,
        dim=0,
    )
    numpy_snapshot.assert_match(output)


def test_masked_mean_dim1(numpy_snapshot, tensor, mask):
    output = masked_mean(
        tensor=tensor,
        mask=mask,
        dim=1,
    )
    numpy_snapshot.assert_match(output)


def test_masked_mean_dimlast(numpy_snapshot, tensor, mask):
    output = masked_mean(
        tensor=tensor,
        mask=mask,
        dim=-1,
    )
    numpy_snapshot.assert_match(output)


def test_masked_mean_dimNone(numpy_snapshot, tensor, mask):
    output = masked_mean(
        tensor=tensor,
        mask=mask,
    )
    numpy_snapshot.assert_match(output)


def test_grpo_microbatch_train_step_grpo_clip(
    numpy_snapshot,
    policy_log_probs,
    response_mask,
    gradient_accumulation_steps,
    raw_rewards,
    advantages,
    old_log_probs,
    cliprange,
):
    policy_log_probs.requires_grad = True

    loss, _ = grpo_microbatch_train_step(
        policy_log_probs=policy_log_probs,
        response_mask=response_mask,
        gradient_accumulation_steps=gradient_accumulation_steps,
        loss_type="grpo_clip",
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    output = {"loss": loss, "policy_log_probs_grad": policy_log_probs.grad}
    numpy_snapshot.assert_match(output)


def test_grpo_microbatch_train_step_grpo_clip_10_steps(
    numpy_snapshot,
    policy_log_probs,
    response_mask,
    gradient_accumulation_steps,
    raw_rewards,
    advantages,
    old_log_probs,
    cliprange,
):
    policy_log_probs.requires_grad = True

    loss_list = []
    grad_list = []
    for _ in range(10):
        loss, _ = grpo_microbatch_train_step(
            policy_log_probs=policy_log_probs,
            response_mask=response_mask,
            gradient_accumulation_steps=gradient_accumulation_steps,
            loss_type="grpo_clip",
            raw_rewards=raw_rewards,
            advantages=advantages,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )
        loss_list.append(loss)
        grad_list.append(policy_log_probs.grad)

    output = {
        "loss": torch.stack(loss_list),
        "policy_log_probs_grad": torch.stack(grad_list),
    }
    numpy_snapshot.assert_match(output)


def test_grpo_microbatch_kl_value_and_gradient_with_response_mask():
    log4 = math.log(4)
    policy = torch.tensor([[-log4, -log4, -1000], [-log4, -1000, -1000]], dtype=torch.float64, requires_grad=True)
    reference = torch.tensor([[-math.log(2), -log4, 0], [-math.log(8), 0, 0]], dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    beta = 0.4
    loss, metadata = grpo_utils.grpo_microbatch_train_step(
        policy_log_probs=policy, response_mask=mask, gradient_accumulation_steps=2,
        loss_type="no_baseline", raw_rewards=torch.ones(2, 1),
        reference_log_probs=reference, kl_beta=beta,
    )

    # 有效 token 的 ref/current 比值为 [2, 1] 和 [0.5]。两条回答等权，K=2。
    expected_kl = ((1 - math.log(2)) / 2 + (math.log(2) - 0.5)) / 4
    assert not metadata["kl_loss"].requires_grad
    torch.testing.assert_close(metadata["kl_loss"], policy.new_tensor(expected_kl))
    torch.testing.assert_close(loss, policy.new_tensor(log4 / 2 + beta * expected_kl))
    # d(KL)/d(log p_current) = 1 - p_ref/p_current。
    expected_grad = policy.new_tensor([[(-1 - beta) / 8, -1 / 8, 0], [(-1 + beta / 2) / 4, 0, 0]])
    torch.testing.assert_close(policy.grad, expected_grad)
    assert reference.grad is None
