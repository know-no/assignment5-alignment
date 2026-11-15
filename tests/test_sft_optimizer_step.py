from copy import deepcopy
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from cs336_alignment.utils.sft_utils import sft_optimizer_step
from cs336_alignment.utils.tokenizer_utils import tokenize_prompt_and_output


@pytest.fixture
def tiny_sft_policy():
    backend = Tokenizer(
        WordLevel(
            {"[PAD]": 0, "[UNK]": 1, "hello": 2, "world": 3, "other": 4, "tail": 5},
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]"
    )
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=16,
        n_embd=16,
        n_layer=1,
        n_head=2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=None,
        eos_token_id=None,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        policy = GPT2LMHeadModel(config).to(device="cpu", dtype=torch.float32)
    return tokenizer, policy


@pytest.mark.parametrize("sample_count", [8, 9], ids=["eight_samples", "drop_ninth"])
@pytest.mark.parametrize("max_grad_norm", [0.05, 1e6], ids=["clip", "no_clip"])
def test_sft_optimizer_step_matches_full_batch(tiny_sft_policy, monkeypatch, sample_count, max_grad_norm):
    tokenizer, policy = tiny_sft_policy
    prompts = [
        "hello other", "hello", "other hello", "world other",
        "hello world", "other", "world", "hello other world", "tail tail",
    ][:sample_count]
    responses = [
        "world", "other world", "hello", "world hello other",
        "other", "hello world", "other world hello", "world other", "tail tail tail",
    ][:sample_count]
    microbatch_size = 2
    normalize_constant = 2.0
    learning_rate = 0.1
    retained_count = sample_count // microbatch_size * microbatch_size
    policy.eval()
    reference_policy = deepcopy(policy)
    reference_policy.train()
    initial_parameters = [parameter.detach().clone() for parameter in policy.parameters()]

    # 当前策略按 floor(N / microbatch_size) 丢尾，独立参考也仅使用前 8 条。
    batch = tokenize_prompt_and_output(prompts[:retained_count], responses[:retained_count], tokenizer)
    logits = reference_policy(batch["input_ids"]).logits.float()
    token_losses = F.cross_entropy(
        logits.reshape(-1, len(tokenizer)), batch["labels"].reshape(-1), reduction="none"
    ).reshape_as(batch["labels"])
    # 每条回答先求和，再按保留样本数和 C 缩放，分母采用样本数。
    reference_loss = (token_losses * batch["response_mask"]).sum() / (retained_count * normalize_constant)
    reference_loss.backward()
    reference_gradients = [parameter.grad.detach().clone() for parameter in reference_policy.parameters()]
    reference_norm = torch.sqrt(sum(gradient.double().square().sum() for gradient in reference_gradients)).item()
    scale = min(1.0, max_grad_norm / (reference_norm + 1e-6))
    clipped_reference = [gradient * scale for gradient in reference_gradients]
    expected_parameters = [
        parameter.detach() - learning_rate * gradient
        for parameter, gradient in zip(reference_policy.parameters(), clipped_reference)
    ]
    assert reference_norm > 0.05
    assert reference_norm < 1e6

    # 预填旧梯度，验证 optimizer step 开始时确实清理上一次的累积。
    for parameter in policy.parameters():
        parameter.grad = torch.full_like(parameter, 17.0)
    optimizer = torch.optim.SGD(policy.parameters(), lr=learning_rate)
    zero_grad_spy = Mock(wraps=optimizer.zero_grad)
    monkeypatch.setattr(optimizer, "zero_grad", zero_grad_spy)
    forward_batches = []

    def record_forward(module, args, kwargs):
        assert all(child.training for child in module.modules())
        if not forward_batches:
            assert all(parameter.grad is None or torch.count_nonzero(parameter.grad) == 0 for parameter in module.parameters())
        input_ids = args[0] if args else kwargs["input_ids"]
        forward_batches.append(input_ids.shape[0])

    gradients_before_clip = []
    gradients_after_clip = []
    real_clip = torch.nn.utils.clip_grad_norm_

    def record_clip(parameters, *args, **kwargs):
        parameters = list(parameters)
        gradients_before_clip.extend(parameter.grad.detach().clone() for parameter in parameters)
        norm = real_clip(parameters, *args, **kwargs)
        gradients_after_clip.extend(parameter.grad.detach().clone() for parameter in parameters)
        return norm

    clip_spy = Mock(side_effect=record_clip)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip_spy)
    gradients_at_step = []
    real_step = optimizer.step

    def record_step():
        gradients_at_step.extend(parameter.grad.detach().clone() for parameter in policy.parameters())
        return real_step()

    step_spy = Mock(side_effect=record_step)
    monkeypatch.setattr(optimizer, "step", step_spy)
    hook = policy.register_forward_pre_hook(record_forward, with_kwargs=True)
    try:
        result = sft_optimizer_step(
            policy, optimizer, tokenizer, prompts, responses,
            microbatch_size=microbatch_size,
            normalize_constant=normalize_constant,
            max_grad_norm=max_grad_norm,
        )
    finally:
        hook.remove()

    assert policy.training
    assert forward_batches == [2, 2, 2, 2]
    zero_grad_spy.assert_called_once()
    assert clip_spy.call_count == 1
    step_spy.assert_called_once_with()
    assert result["loss"] == pytest.approx(reference_loss.item(), rel=1e-5, abs=1e-6)
    # clip_grad_norm_ 返回裁剪前的全局 L2 范数，两种阈值都使用该参考。
    assert result["grad_norm_before_clip"] == pytest.approx(reference_norm, rel=1e-4, abs=1e-6)
    for actual, expected in zip(gradients_before_clip, reference_gradients):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
    for actual_gradients in (gradients_after_clip, gradients_at_step):
        for actual, expected in zip(actual_gradients, clipped_reference):
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
    for actual, expected in zip(policy.parameters(), expected_parameters):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
    assert any(
        not torch.equal(parameter, initial)
        for parameter, initial in zip(policy.parameters(), initial_parameters)
    )
