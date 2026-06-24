"""Smoke test for vllm-metal on Mac.

Run with the vllm-metal venv's Python:
    HF_HUB_OFFLINE=1 ~/.venv-vllm-metal/bin/python scripts/smoke_test_vllm_metal.py

Assumes Qwen2.5-0.5B-Instruct is already cached locally (HF format).
For full Metal acceleration use an mlx-community model instead, e.g.
`mlx-community/Qwen2.5-0.5B-Instruct-bf16`.
"""

from vllm import LLM, SamplingParams


def main():
    llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct")
    prompts = ["The capital of France is"]
    sp = SamplingParams(temperature=0.0, max_tokens=20)
    outputs = llm.generate(prompts, sp)
    for o in outputs:
        print("PROMPT:", o.prompt)
        print("OUTPUT:", o.outputs[0].text)


if __name__ == "__main__":
    main()
