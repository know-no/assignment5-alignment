"""GSM8K zero-shot baseline 评估。

按 PDF Problem (math_baseline) 要求：
  (1) 加载数据集
  (2) 用 r1_zero prompt 格式化
  (3) vllm 推理
  (4) 用 r1_zero_reward_fn 算 format/answer reward 并分桶
  (5) 序列化到 disk

Mac (vllm-metal) 上跑：
    HF_HUB_OFFLINE=1 ~/.venv-vllm-metal/bin/python -m cs336_alignment.math_baseline
"""

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.utils.evaluate import (
    evaluate_vllm,
    generate_sample_params,
    load_vllm_model,
)
from cs336_alignment.utils.gsm8k_loader import get_dataset
from cs336_alignment.utils.prompt_template_reader import read_prompt_template, r1_zero


N_SAMPLES = 10  # 本机调试用小样本；上 GPU 后改成 None 跑全量
MODEL = "mlx-community/Qwen2.5-Math-1.5B-bf16"  # GPU 上换成 "Qwen/Qwen2.5-Math-1.5B"
OUTPUT_PATH = "outputs/gsm8k_baseline.jsonl"


def gsm8k_extract_gt(answer_field: str) -> str:
    """GSM8K 的 answer 字段末尾是 '#### <数字>'，取最后那段。"""
    return answer_field.split("####")[-1].strip()


def main():
    dataset = get_dataset()
    test_data = dataset["test"]
    if N_SAMPLES is not None:
        test_data = test_data.select(range(N_SAMPLES))

    prompt_template = read_prompt_template(r1_zero)
    prompts = [prompt_template.replace("{question}", row["question"]) for row in test_data]
    ground_truths = [gsm8k_extract_gt(row["answer"]) for row in test_data]

    llm = load_vllm_model(MODEL)
    sp = generate_sample_params()

    evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=sp,
        output_path=OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
