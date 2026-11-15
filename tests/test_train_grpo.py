import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from cs336_alignment import train_grpo


@pytest.fixture(autouse=True)
def preserve_rng_states():
    # loop 内部会设置全局种子，测试退出时恢复各个已使用后端的随机状态。
    cpu_state = torch.get_rng_state()
    mps_state = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


@pytest.fixture
def grpo_tokenizer():
    vocabulary = {
        "[PAD]": 0, "[UNK]": 1, "[EOS]": 2,
        "alpha": 3, "beta": 4, "yes": 5, "no": 6, "tail": 7,
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]",
        eos_token="[EOS]",
    )


class ControlledPolicy(torch.nn.Module):
    """可求导的极小转移模型，固定生成内容便于独立核对组顺序和更新。"""

    def __init__(self, tokenizer, empty_mode=None):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.linspace(-0.3, 0.3, 64).reshape(8, 8))
        self.tokenizer = tokenizer
        self.empty_mode = empty_mode
        self.generated = []
        self.forwards = []
        self.updates = 0

    def forward(self, input_ids):
        assert input_ids.device == self.logits.device
        self.forwards.append({
            "training": self.training, "grad_enabled": torch.is_grad_enabled(),
            "input_ids": input_ids.detach().clone(), "updates": self.updates,
        })
        return SimpleNamespace(logits=self.logits[input_ids])

    def generate(self, input_ids, **kwargs):
        assert not self.training and not torch.is_grad_enabled()
        assert input_ids.device == self.logits.device
        assert kwargs["do_sample"] is True
        assert kwargs["num_return_sequences"] == 2
        prompt = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        # 正常回答长短不同，覆盖 padding、shift 与 response_mask 的对齐。
        continuation = [[5, 2], [6, 7]]
        if self.empty_mode == "no_tokens":
            continuation = [[], []]
        elif self.empty_mode == "special_tokens":
            continuation = [[2, 0], [5, 7]]  # 仅第一条解码为空，不能丢掉这一组成员。
        continuation = torch.tensor(continuation, device=input_ids.device, dtype=torch.long)
        self.generated.append({
            "prompt": prompt,
            "responses": self.tokenizer.batch_decode(continuation, skip_special_tokens=True),
            "parameters": self.logits.detach().clone(), "updates": self.updates,
        })
        return torch.cat((input_ids.repeat(2, 1), continuation), dim=1)


def exercise_two_updates(tokenizer, monkeypatch, loss_type, device="cpu"):
    model = ControlledPolicy(tokenizer).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    initial = model.logits.detach().clone()
    truths = {"alpha tail": "yes", "beta": "no"}
    reward_calls = []
    checked_batches = []
    real_optimizer_step = train_grpo.grpo_optimizer_step
    real_step = optimizer.step

    def reward_fn(response, truth):
        reward_calls.append((response, truth))
        return {"reward": float(response.split()[0] == truth)}

    def record_step(*args, **kwargs):
        # 每次 step 前恰有两个训练微批，而非每个微批执行一次 step。
        current = [event for event in model.forwards if event["updates"] == model.updates]
        assert sum(event["grad_enabled"] for event in current) == 2
        assert torch.isfinite(model.logits.grad).all() and model.logits.grad.norm() > 0
        result = real_step(*args, **kwargs)
        model.updates += 1
        return result

    optimizer.step = Mock(side_effect=record_step)
    optimizer.zero_grad = Mock(wraps=optimizer.zero_grad)

    def check_optimizer_batch(**kwargs):
        round_index = len(checked_batches)
        groups = model.generated[round_index * 2:(round_index + 1) * 2]
        rows = [(group["prompt"], response) for group in groups for response in group["responses"]]
        assert len(rows) == 4
        assert reward_calls[round_index * 4:(round_index + 1) * 4] == [
            (response, truths[prompt]) for prompt, response in rows
        ]
        ids, labels, mask = (kwargs[key] for key in ("input_ids", "labels", "response_mask"))
        assert ids.shape == labels.shape == mask.shape
        assert ids.dtype == labels.dtype == torch.long and mask.dtype == torch.bool
        for row, (prompt, response) in enumerate(rows):
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
            response_ids = tokenizer.encode(response, add_special_tokens=False)
            sequence = torch.tensor(prompt_ids + response_ids, device=device)
            torch.testing.assert_close(ids[row, :len(sequence) - 1], sequence[:-1])
            torch.testing.assert_close(labels[row, :len(sequence) - 1], sequence[1:])
            torch.testing.assert_close(labels[row][mask[row]], sequence[len(prompt_ids):])
            assert not mask[row, :len(prompt_ids) - 1].any()
            assert not mask[row, len(sequence) - 1:].any()

        rewards = torch.tensor(
            [float(response.split()[0] == truths[prompt]) for prompt, response in rows],
            device=device,
        ).unsqueeze(-1)
        # 每组恰好 [1, 0] 或 [0, 1]，样本标准差为 sqrt(1/2)。
        advantages = (rewards - 0.5) / (math.sqrt(0.5) + 1e-6)
        torch.testing.assert_close(kwargs["raw_rewards"], rewards)
        torch.testing.assert_close(kwargs["advantages"], advantages)
        for key in ("input_ids", "labels", "response_mask", "raw_rewards", "advantages"):
            assert kwargs[key].device == model.logits.device
        assert kwargs["raw_rewards"].shape == kwargs["advantages"].shape == (4, 1)
        assert kwargs["raw_rewards"].dtype == kwargs["advantages"].dtype == torch.float32
        old = kwargs["old_log_probs"]
        if loss_type == "grpo_clip":
            assert old.shape == ids.shape and old.device == model.logits.device
            assert not old.requires_grad and old.grad_fn is None
        else:
            assert old is None

        # 用独立整批 CE 求导验证累积结果；首次 ratio=1 的梯度等于固定 A 的 PG。
        before = model.logits.detach().clone()
        reference = before.clone().requires_grad_(True)
        token_nll = F.cross_entropy(
            reference[ids].reshape(-1, 8), labels.flatten(), reduction="none"
        ).reshape_as(labels)
        weights = rewards if loss_type == "no_baseline" else advantages
        reference_loss = (weights.squeeze(-1) * (token_nll * mask).sum(-1) / mask.sum(-1)).mean()
        gradient, = torch.autograd.grad(reference_loss, reference)
        expected_norm = gradient.norm()
        expected_loss = -advantages.mean() if loss_type == "grpo_clip" else reference_loss.detach()
        scale = min(1.0, 0.05 / (expected_norm.item() + 1e-6))
        expected_parameters = before - 0.1 * gradient * scale
        result = real_optimizer_step(**kwargs)
        assert result["loss"] == pytest.approx(expected_loss.item(), abs=2e-6)
        assert result["grad_norm_before_clip"] == pytest.approx(expected_norm.item(), rel=2e-5, abs=1e-6)
        torch.testing.assert_close(model.logits, expected_parameters, rtol=2e-5, atol=2e-6)
        assert not torch.equal(model.logits, before)
        checked_batches.append(result)
        return result

    monkeypatch.setattr(train_grpo, "grpo_optimizer_step", check_optimizer_batch)
    history = train_grpo.grpo_train_loop(
        model, optimizer, tokenizer, reward_fn, list(truths), list(truths.values()),
        num_steps=2, n_prompts_per_step=2, group_size=2, microbatch_size=2,
        loss_type=loss_type, max_new_tokens=2, max_grad_norm=0.05,
    )
    assert len(history) == len(checked_batches) == model.updates == 2
    assert optimizer.zero_grad.call_count == optimizer.step.call_count == 2
    assert len(model.generated) == 4
    for round_index in range(2):
        generated = model.generated[round_index * 2:(round_index + 1) * 2]
        assert all(group["updates"] == round_index for group in generated)
        torch.testing.assert_close(generated[0]["parameters"], generated[1]["parameters"])
        events = [event for event in model.forwards if event["updates"] == round_index]
        assert [(event["training"], event["grad_enabled"]) for event in events] == (
            [(False, False)] if loss_type == "grpo_clip" else []
        ) + [(True, True), (True, True)]
        if loss_type == "grpo_clip":
            torch.testing.assert_close(events[0]["input_ids"], torch.cat([event["input_ids"] for event in events[1:]]))
        assert history[round_index] == {
            "reward_mean": 0.5, "reward_min": 0.0, "reward_max": 1.0,
            **checked_batches[round_index],
        }
        assert all(math.isfinite(value) for value in history[round_index].values())
    assert not torch.equal(model.generated[0]["parameters"], model.generated[2]["parameters"])
    assert torch.isfinite(model.logits).all() and not torch.equal(model.logits, initial)


@pytest.mark.parametrize("loss_type", ["no_baseline", "reinforce_with_baseline", "grpo_clip"])
def test_two_rollout_batches_each_make_one_update(grpo_tokenizer, monkeypatch, loss_type):
    exercise_two_updates(grpo_tokenizer, monkeypatch, loss_type)


@pytest.mark.parametrize("empty_mode", ["no_tokens", "special_tokens"])
def test_empty_response_rejected_before_scoring_or_optimizer(grpo_tokenizer, empty_mode):
    model = ControlledPolicy(grpo_tokenizer, empty_mode=empty_mode)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.zero_grad = Mock(wraps=optimizer.zero_grad)
    optimizer.step = Mock(wraps=optimizer.step)
    before = model.logits.detach().clone()
    model.logits.grad = torch.full_like(model.logits, 3.0)
    old_gradient = model.logits.grad.clone()
    with pytest.raises(ValueError, match="空回答"):
        train_grpo.grpo_train_loop(
            model, optimizer, grpo_tokenizer, lambda response, truth: {"reward": 1.0},
            ["alpha tail"], ["yes"], num_steps=1, n_prompts_per_step=1,
            group_size=2, microbatch_size=1,
        )
    assert any(not response for response in model.generated[0]["responses"])
    assert model.forwards == []  # 拒绝发生在 old 评分之前，也不能清掉已有梯度。
    optimizer.zero_grad.assert_not_called()
    optimizer.step.assert_not_called()
    torch.testing.assert_close(model.logits, before)
    torch.testing.assert_close(model.logits.grad, old_gradient)
    assert model.training


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="需要真实可用的 MPS 设备验证迁移")
def test_loop_moves_encoded_data_and_weights_to_mps(grpo_tokenizer, monkeypatch):
    exercise_two_updates(grpo_tokenizer, monkeypatch, "grpo_clip", device="mps")


def test_real_hf_generate_and_training_without_download(grpo_tokenizer):
    config = GPT2Config(
        vocab_size=len(grpo_tokenizer), n_positions=16, n_embd=8, n_layer=1, n_head=1,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        bos_token_id=None, eos_token_id=None, pad_token_id=grpo_tokenizer.pad_token_id,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        model = GPT2LMHeadModel(config).to(device="cpu", dtype=torch.float32)
        # 不依赖随机生成的具体词，只屏蔽特殊 token，保证解码后有回答可训练。
        model.generation_config.suppress_tokens = grpo_tokenizer.all_special_ids
        before = [parameter.detach().clone() for parameter in model.parameters()]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        modes = []
        hook = model.register_forward_pre_hook(
            lambda module, args: modes.append((module.training, torch.is_grad_enabled()))
        )
        try:
            history = train_grpo.grpo_train_loop(
                model, optimizer, grpo_tokenizer, lambda response, truth: {"reward": 1.0},
                ["alpha tail"], ["yes"], num_steps=1, n_prompts_per_step=1,
                group_size=2, microbatch_size=1, loss_type="no_baseline", max_new_tokens=2,
            )
        finally:
            hook.remove()
    assert len(history) == 1 and model.training
    assert all(math.isfinite(value) for value in history[0].values())
    assert history[0]["loss"] > 0 and history[0]["grad_norm_before_clip"] > 0
    assert (False, False) in modes and modes[-2:] == [(True, True), (True, True)]
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    assert any(not torch.equal(parameter, old) for parameter, old in zip(model.parameters(), before))
