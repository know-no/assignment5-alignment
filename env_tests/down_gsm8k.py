"""加载本地 GSM8K 数据集（不联网）。

本仓库 data/gsm8k/ 下已经有 train.jsonl 和 test.jsonl，格式与 HF 上
`openai/gsm8k` 的 "main" config 一致（字段：question / answer）。
直接用 datasets.load_dataset 读本地 jsonl 即可，省去下载。

如果以后要 socratic 版本或者更新版，再走 HF 在线接口。
"""

import os
import pathlib

from datasets import load_dataset

cs336_alignment_path = os.path.dirname(os.path.dirname(__file__))
base_path = os.path.join(cs336_alignment_path, 'data/gsm8k')
train_path = os.path.join(base_path, 'train.jsonl')
test_path = os.path.join(base_path, 'test.jsonl')

def get_dataset():
    dataset = load_dataset(
        "json",
        data_files={
            "train": train_path,
            "test": test_path,
        },
    )
    return dataset


def main():
    dataset = load_dataset(
        "json",
        data_files={
            "train": train_path,
            "test": test_path,
        },
    )
    print(dataset)
    print("first test sample:")
    print(dataset["test"][0])
    print("train size:", len(dataset["train"]))
    print("test size:", len(dataset["test"]))


if __name__ == "__main__":
    main()


# import os
# # 1. 注入镜像站终端节点，规避网络重置问题
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# from datasets import load_dataset

# print("正在下载 GSM8K 数据集...")
# # 2. 加载 openai/gsm8k 的主配置
# # gsm8k 包含 'train' 和 'test' 两个 split
# dataset = load_dataset("openai/gsm8k", "main")

# # 3. 查看数据结构
# print(dataset["test"][0])
# # 输出类似: {'question': '...', 'answer': '... \n#### 42'}