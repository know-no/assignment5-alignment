import json
import math
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from cs336_alignment import train_sft
from cs336_alignment.utils.sft_utils import sft_train_loop


@pytest.fixture
def saved_sft_inputs(tmp_path):
    backend = Tokenizer(
        WordLevel({"[PAD]": 0, "[UNK]": 1, "hello": 2, "world": 3, "other": 4}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]"
    )
    config = GPT2Config(
        vocab_size=len(tokenizer), n_positions=16, n_embd=16, n_layer=1, n_head=2,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        bos_token_id=None, eos_token_id=None, pad_token_id=tokenizer.pad_token_id,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        model = GPT2LMHeadModel(config).to(device="cpu", dtype=torch.float32)
    model_path = tmp_path / "tiny_model"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    rows = [
        {"prompt": "hello other", "response": "world"},
        {"prompt": "other hello", "response": "world other"},
        {"prompt": "hello", "response": "other world"},
        {"prompt": "other", "response": "hello world"},
        {"prompt": "world", "response": "hello"},
    ]
    data_path = tmp_path / "sft.jsonl"
    data_path.write_text("\n" + "\n".join(json.dumps(row) for row in rows) + "\n\n", encoding="utf-8")
    return model_path, data_path


def json_records(output):
    return [json.loads(line) for line in output.splitlines() if line.startswith("{")]


def assert_two_updates(records, dtype="float32"):
    assert records[0]["event"] == "config"
    assert records[0]["device"] == "cpu"
    assert records[0]["dtype"] == dtype
    assert records[0]["num_epochs"] == 2
    assert records[0]["batch_size"] == 4
    assert records[0]["microbatch_size"] == 2
    assert len(records) == 3
    for step, record in enumerate(records[1:], start=1):
        assert record["event"] == "step"
        assert record["step"] == step
        assert math.isfinite(record["loss"]) and record["loss"] > 0
        assert math.isfinite(record["grad_norm_before_clip"]) and record["grad_norm_before_clip"] > 0


def test_cli_loads_local_model_and_outputs_two_updates(saved_sft_inputs):
    model_path, data_path = saved_sft_inputs
    result = subprocess.run(
        [
            sys.executable, "-m", "cs336_alignment.train_sft",
            "--model-name-or-path", str(model_path), "--data-path", str(data_path),
            "--num-epochs", "2", "--batch-size", "4", "--microbatch-size", "2",
            "--learning-rate", "0.001", "--device", "cpu", "--dtype", "float32",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, check=True, timeout=60,
    )
    records = json_records(result.stdout)
    assert_two_updates(records)
    assert records[0]["learning_rate"] == 0.001


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_main_moves_model_before_optimizer_and_updates_parameters(saved_sft_inputs, monkeypatch, capsys, dtype):
    model_path, data_path = saved_sft_inputs
    loaded = []
    real_load = train_sft.AutoModelForCausalLM.from_pretrained
    real_adamw = torch.optim.AdamW
    expected_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]

    def record_load(*args, **kwargs):
        assert kwargs["dtype"] == expected_dtype
        model = real_load(*args, **kwargs)
        assert all(parameter.dtype == expected_dtype for parameter in model.parameters())
        before = [parameter.detach().clone() for parameter in model.parameters()]
        model.to = Mock(wraps=model.to)
        loaded.append((model, before))
        return model

    def make_optimizer(parameters, **kwargs):
        model, _ = loaded[0]
        assert model.to.call_count == 1
        parameters = list(parameters)
        assert all(parameter.device.type == "cpu" for parameter in parameters)
        return real_adamw(parameters, **kwargs)

    monkeypatch.setattr(train_sft.AutoModelForCausalLM, "from_pretrained", record_load)
    monkeypatch.setattr(train_sft.torch.optim, "AdamW", make_optimizer)
    assert train_sft.main(
        str(model_path), str(data_path), learning_rate=0.001, device="cpu", dtype=dtype
    ) is None
    assert_two_updates(json_records(capsys.readouterr().out), dtype=dtype)
    model, before = loaded[0]
    assert all(parameter.dtype == expected_dtype for parameter in model.parameters())
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    assert any(not torch.equal(parameter, initial) for parameter, initial in zip(model.parameters(), before))


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"batch_size": 0}, "batch_size"),
        ({"microbatch_size": 0}, "microbatch_size"),
        ({"microbatch_size": 3}, "整除"),
        ({"prompt_strs": ["hello"] * 3, "output_strs": ["world"] * 3}, "完整 batch"),
        ({"num_epochs": 0}, "num_epochs"),
        ({"output_strs": ["world"]}, "长度"),
    ],
)
def test_loop_rejects_invalid_inputs_before_model_use(changes, message):
    arguments = {
        "prompt_strs": ["hello"] * 4, "output_strs": ["world"] * 4,
        "num_epochs": 1, "batch_size": 4, "microbatch_size": 2,
    }
    arguments.update(changes)
    with pytest.raises(ValueError, match=message):
        sft_train_loop(model=None, optimizer=None, tokenizer=None, **arguments)


@pytest.mark.parametrize(
    "content",
    [
        '{"prompt": "hello"}',
        '{"prompt": "hello", "response": 7}',
        "invalid json",
    ],
)
def test_main_reports_bad_data_line_before_loading_model(tmp_path, content):
    data_path = tmp_path / "bad.jsonl"
    data_path.write_text("\n" + content + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="第 2 行"):
        train_sft.main("unused-model", str(data_path), device="cpu")
