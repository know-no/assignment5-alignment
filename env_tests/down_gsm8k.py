import os
# 1. 注入镜像站终端节点，规避网络重置问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset

print("正在下载 GSM8K 数据集...")
# 2. 加载 openai/gsm8k 的主配置
# gsm8k 包含 'train' 和 'test' 两个 split
dataset = load_dataset("openai/gsm8k", "main")

# 3. 查看数据结构
print(dataset["test"][0])
# 输出类似: {'question': '...', 'answer': '... \n#### 42'}