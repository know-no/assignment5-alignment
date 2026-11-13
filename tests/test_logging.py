import json
import math
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from cs336_alignment.utils.log_utils import log_generations
from cs336_alignment.utils.tokenizer_utils import tokenize_prompt_and_output
from tests import adapters
from tests.common import FIXTURES_PATH


@pytest.fixture
def logging_tokenizer():
    backend = Tokenizer(
        WordLevel({"[PAD]": 0, "[UNK]": 1, "hello": 2, "world": 3, "other": 4, "[BOS]": 5}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", bos_token="[BOS]"
    )


class UniformPolicy(torch.nn.Module):
    """Uniform probabilities give an independently known entropy of log(vocab_size)."""

    def __init__(self, vocab_size):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
        self.dropout = torch.nn.Dropout()
        self.vocab_size = vocab_size
        self.forward_batches = []

    def forward(self, input_ids):
        assert not self.training
        assert not self.dropout.training
        assert not torch.is_grad_enabled()
        self.forward_batches.append(input_ids.shape[0])
        logits = torch.zeros(
            (*input_ids.shape, self.vocab_size), device=input_ids.device, dtype=self.anchor.dtype
        ) + self.anchor
        return SimpleNamespace(logits=logits)


def reward(response, ground_truth):
    answer_reward = float(response == ground_truth)
    return {"format_reward": 1.0, "answer_reward": answer_reward, "reward": answer_reward}


def test_response_does_not_repeat_tokenizer_bos(logging_tokenizer):
    logging_tokenizer.backend_tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A", special_tokens=[("[BOS]", logging_tokenizer.bos_token_id)]
    )
    batch = tokenize_prompt_and_output(["hello"], ["world"], logging_tokenizer)
    assert batch["input_ids"].tolist() == [[5, 2]]
    assert batch["labels"].tolist() == [[2, 3]]
    assert batch["response_mask"].tolist() == [[False, True]]


def test_actual_llama_response_and_empty_response():
    tokenizer = AutoTokenizer.from_pretrained(FIXTURES_PATH / "Meta-Llama-3-8B", local_files_only=True)
    batch = tokenize_prompt_and_output(["hello", "hello"], ["world", ""], tokenizer)
    response_ids = tokenizer.encode("world", add_special_tokens=False)
    assert batch["response_mask"].sum(-1).tolist() == [len(response_ids), 0]
    assert batch["labels"][0][batch["response_mask"][0]].tolist() == response_ids
    assert tokenizer.bos_token_id not in batch["labels"][0][batch["response_mask"][0]].tolist()


def test_tokenization_empty_batches_and_responses(logging_tokenizer):
    empty = tokenize_prompt_and_output([], [], logging_tokenizer)
    assert all(tensor.shape == (0, 0) for tensor in empty.values())
    no_response = tokenize_prompt_and_output(["hello"], [""], logging_tokenizer)
    assert all(tensor.shape == (1, 0) for tensor in no_response.values())


def test_tokenization_requires_aligned_lists_and_context(logging_tokenizer):
    with pytest.raises(ValueError, match="长度"):
        tokenize_prompt_and_output(["hello"], [], logging_tokenizer)
    with pytest.raises(ValueError, match="上下文"):
        tokenize_prompt_and_output([""], ["world"], logging_tokenizer)


def test_logging_skips_empty_response_entropy(logging_tokenizer):
    policy = UniformPolicy(len(logging_tokenizer))
    policy.dropout.eval()

    def generate(prompts):
        assert not policy.training
        assert not policy.dropout.training
        assert not torch.is_grad_enabled()
        return ["", "world"]

    result = log_generations(
        ["hello", "hello"], ["world", "world"], generate, reward,
        tokenizer=logging_tokenizer, policy_model=policy,
    )
    assert result["records"][0]["mean_token_entropy"] is None
    assert result["records"][1]["mean_token_entropy"] == pytest.approx(math.log(len(logging_tokenizer)))
    assert result["summary"]["mean_token_entropy"] == pytest.approx(math.log(len(logging_tokenizer)))
    assert [record["response_len_tokens"] for record in result["records"]] == [0, 1]
    assert policy.forward_batches == [1]
    assert policy.training
    assert not policy.dropout.training
    assert policy.anchor.grad is None


def test_logging_all_empty_responses_avoids_model_forward(logging_tokenizer):
    policy = UniformPolicy(len(logging_tokenizer))
    result = log_generations(
        ["hello"], ["world"], lambda _: [""], reward,
        tokenizer=logging_tokenizer, policy_model=policy,
    )
    assert result["records"][0]["mean_token_entropy"] is None
    assert result["summary"]["mean_token_entropy"] is None
    assert result["summary"]["mean_response_len"] == 0
    assert policy.forward_batches == []
    assert policy.training


def test_generation_exception_restores_module_modes(logging_tokenizer):
    policy = UniformPolicy(len(logging_tokenizer))
    policy.dropout.eval()

    def fail(prompts):
        assert not policy.training
        assert not torch.is_grad_enabled()
        raise RuntimeError("generation failed")

    with pytest.raises(RuntimeError, match="generation failed"):
        log_generations(["hello"], ["world"], fail, reward, policy_model=policy, tokenizer=logging_tokenizer)
    assert policy.training
    assert not policy.dropout.training


def test_scoring_exception_restores_module_modes(logging_tokenizer):
    class FailingPolicy(UniformPolicy):
        def forward(self, input_ids):
            raise RuntimeError("scoring failed")

    policy = FailingPolicy(len(logging_tokenizer))
    policy.dropout.eval()
    with pytest.raises(RuntimeError, match="scoring failed"):
        log_generations(
            ["hello"], ["world"], lambda _: ["world"], reward,
            policy_model=policy, tokenizer=logging_tokenizer,
        )
    assert policy.training
    assert not policy.dropout.training


def test_extra_cannot_replace_summary_fields():
    def unexpected_generation(prompts):
        pytest.fail("Invalid metadata must be rejected before generation")

    for prompts, answers in [([], []), (["hello"], ["world"])]:
        with pytest.raises(ValueError):
            log_generations(prompts, answers, unexpected_generation, reward, extra={"n": 99})


def test_empty_batch_has_complete_summary(logging_tokenizer):
    ordinary = log_generations(
        ["hello"], ["world"], lambda _: ["world"], reward,
        tokenizer=logging_tokenizer, step=3, extra={"epoch": 1},
    )
    empty = log_generations([], [], lambda _: pytest.fail("empty batch generated"), reward, step=3, extra={"epoch": 1})
    assert empty["records"] == []
    assert empty["summary"].keys() == ordinary["summary"].keys()
    assert empty["summary"]["n"] == 0
    assert empty["summary"]["step"] == 3
    assert empty["summary"]["epoch"] == 1
    assert all(value is None for name, value in empty["summary"].items() if name.startswith("mean_"))


def test_logging_appends_records_and_aggregates_lengths(logging_tokenizer, tmp_path):
    path = tmp_path / "nested" / "generations.jsonl"
    for step in (1, 2):
        result = log_generations(
            ["hello", "hello"], ["world", "world"], lambda _: ["world", "hello world"], reward,
            tokenizer=logging_tokenizer, log_path=path, step=step, extra={"epoch": 1},
        )
        assert result["summary"]["mean_reward"] == 0.5
        assert result["summary"]["mean_response_len"] == 1.5
        assert result["summary"]["mean_correct_len"] == 1
        assert result["summary"]["mean_wrong_len"] == 2
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["step"] for record in records] == [1, 1, 2, 2]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_sft_logging_with_tiny_causal_lm(logging_tokenizer, tmp_path, dtype):
    """Exercise real generation between two SFT backwards on the same tiny policy."""
    torch.manual_seed(42)
    config = GPT2Config(
        vocab_size=len(logging_tokenizer),
        n_positions=16,
        n_embd=16,
        n_layer=1,
        n_head=2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        bos_token_id=logging_tokenizer.bos_token_id,
        pad_token_id=logging_tokenizer.pad_token_id,
        eos_token_id=None,
    )
    config._attn_implementation = "eager"
    policy = GPT2LMHeadModel(config).to(dtype=dtype)
    policy.train()
    policy.transformer.drop.eval()
    original_modes = [module.training for module in policy.modules()]
    reference_policy = deepcopy(policy)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.25)
    prompts = ["hello other", "other hello"]
    accumulation_steps = 2
    normalize_constant = 2.0

    def backward_microbatch(responses):
        batch = adapters.run_tokenize_prompt_and_output(prompts, responses, logging_tokenizer)
        output = adapters.run_get_response_log_probs(
            policy, batch["input_ids"], batch["labels"], return_token_entropy=True
        )
        output["log_probs"].retain_grad()
        loss, metadata = adapters.run_sft_microbatch_train_step(
            output["log_probs"], batch["response_mask"], accumulation_steps, normalize_constant
        )
        denominator = len(prompts) * normalize_constant * accumulation_steps
        expected_log_prob_grad = -batch["response_mask"].float() / denominator
        torch.testing.assert_close(output["log_probs"].grad, expected_log_prob_grad, rtol=0, atol=0)

        reference_logits = reference_policy(batch["input_ids"]).logits.float()
        token_loss = F.cross_entropy(
            reference_logits.reshape(-1, config.vocab_size), batch["labels"].reshape(-1), reduction="none"
        ).reshape_as(batch["labels"])
        reference_loss = (token_loss * batch["response_mask"]).sum() / denominator
        torch.testing.assert_close(loss.detach(), reference_loss.detach(), rtol=1e-5, atol=1e-6)
        reference_loss.backward()
        assert torch.isfinite(loss) and loss.item() > 0
        assert metadata["unscaled_loss"].item() == pytest.approx(loss.item() * accumulation_steps)

    backward_microbatch(["world other", "world hello"])
    parameters_before_logging = {name: parameter.detach().clone() for name, parameter in policy.named_parameters()}
    gradients_before_logging = {name: parameter.grad.clone() for name, parameter in policy.named_parameters()}
    assert all(torch.isfinite(gradient).all() for gradient in gradients_before_logging.values())
    assert sum(gradient.abs().sum().item() for gradient in gradients_before_logging.values()) > 0

    forward_modes = []
    generation_forward_count = 0

    def check_logging_forward(module, args):
        assert not torch.is_grad_enabled()
        assert all(not child.training for child in module.modules())
        forward_modes.append(module.training)

    def generate(prompts_to_generate):
        nonlocal generation_forward_count
        assert not torch.is_grad_enabled()
        assert all(not module.training for module in policy.modules())
        encoded = logging_tokenizer(
            prompts_to_generate, padding=True, return_tensors="pt", return_token_type_ids=False
        )
        generated = policy.generate(
            **encoded,
            max_new_tokens=2,
            do_sample=False,
            eos_token_id=None,
            bad_words_ids=[[token_id] for token_id in logging_tokenizer.all_special_ids],
        )
        generation_forward_count = len(forward_modes)
        return logging_tokenizer.batch_decode(generated[:, encoded["input_ids"].shape[1]:], skip_special_tokens=True)

    path = tmp_path / "tiny_policy_generations.jsonl"
    hook = policy.register_forward_pre_hook(check_logging_forward)
    try:
        result = log_generations(
            prompts, ["world", "world"], generate, reward,
            policy_model=policy, tokenizer=logging_tokenizer, log_path=path, step=1,
        )
    finally:
        hook.remove()

    assert generation_forward_count > 0
    assert len(forward_modes) > generation_forward_count
    assert [module.training for module in policy.modules()] == original_modes
    for name, parameter in policy.named_parameters():
        torch.testing.assert_close(parameter, parameters_before_logging[name], rtol=0, atol=0)
        torch.testing.assert_close(parameter.grad, gradients_before_logging[name], rtol=0, atol=0)
    assert result["summary"]["n"] == 2
    assert result["summary"]["step"] == 1
    assert result["summary"]["mean_response_len"] == 2
    assert result["summary"]["mean_reward"] == 0
    assert result["summary"]["mean_correct_len"] is None
    assert result["summary"]["mean_wrong_len"] == 2
    for record in result["records"]:
        assert record["response_len_tokens"] == 2
        assert record["ground_truth"] == "world"
        assert math.isfinite(record["mean_token_entropy"])
        assert 0 < record["mean_token_entropy"] <= math.log(config.vocab_size) + 1e-5
    assert [json.loads(line) for line in path.read_text().splitlines()] == result["records"]

    backward_microbatch(["other world", "hello world"])
    gradient_rtol, gradient_atol = (0.02, 0.002) if dtype == torch.bfloat16 else (1e-4, 1e-6)
    for parameter, reference_parameter in zip(policy.parameters(), reference_policy.parameters()):
        assert torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad, reference_parameter.grad, rtol=gradient_rtol, atol=gradient_atol)
    optimizer.step()
    assert all(torch.isfinite(parameter).all() for parameter in policy.parameters())
    assert any(
        not torch.equal(parameter, parameters_before_logging[name])
        for name, parameter in policy.named_parameters()
    )
