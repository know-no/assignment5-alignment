import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.utils.sft_utils import load_sft_data, sft_train_loop
from cs336_alignment.utils.tokenizer_utils import pick_device


def main(
    model_name_or_path: str,
    data_path: str,
    *,
    num_epochs: int = 2,
    batch_size: int = 4,
    microbatch_size: int = 2,
    learning_rate: float = 1e-5,
    device: str | None = None,
    dtype: str = "float32",
) -> None:
    prompts, responses = load_sft_data(data_path)
    dtypes = {"float32": torch.float32, "bfloat16": torch.bfloat16}
    if dtype not in dtypes:
        raise ValueError("dtype 必须为 float32 或 bfloat16")
    target_device = pick_device() if device is None else torch.device(device)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype=dtypes[dtype])
    model = model.to(target_device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    config = {
        "event": "config",
        "model_name_or_path": str(model_name_or_path),
        "data_path": str(data_path),
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "microbatch_size": microbatch_size,
        "learning_rate": learning_rate,
        "device": str(target_device),
        "dtype": dtype,
    }
    print(json.dumps(config, ensure_ascii=False))
    history = sft_train_loop(
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        prompt_strs=prompts,
        output_strs=responses,
        num_epochs=num_epochs,
        batch_size=batch_size,
        microbatch_size=microbatch_size,
    )
    for step, metrics in enumerate(history, start=1):
        print(json.dumps({"event": "step", "step": step, **metrics}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行 SFT 训练并输出每步损失和梯度范数。")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--microbatch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    main(**vars(parser.parse_args()))

