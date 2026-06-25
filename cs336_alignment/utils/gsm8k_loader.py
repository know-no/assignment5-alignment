"""GSM8K 数据集加载工具。

直接读 data/gsm8k/{train,test}.jsonl，跟 HF 上 `openai/gsm8k` "main" config
的字段（question / answer）一致。
"""

import os

from datasets import DatasetDict, load_dataset


# repo root：cs336_alignment/utils/gsm8k_loader.py → 上两层
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BASE = os.path.join(_REPO_ROOT, "data", "gsm8k")


def get_dataset() -> DatasetDict:
    return load_dataset(
        "json",
        data_files={
            "train": os.path.join(_BASE, "train.jsonl"),
            "test": os.path.join(_BASE, "test.jsonl"),
        },
    )
