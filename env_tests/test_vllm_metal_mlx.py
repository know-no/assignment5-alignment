from vllm import LLM, SamplingParams

def main():
    llm = LLM(model="mlx-community/Qwen2.5-0.5B-Instruct-bf16")
    prompts = ["The capital of France is"]
    sp = SamplingParams(temperature=0.0, max_tokens=20)
    outputs = llm.generate(prompts, sp)
    for o in outputs:
        print("PROMPT:", o.prompt)
        print("OUTPUT:", o.outputs[0].text)
    print("OK")


if __name__ == "__main__":
    main()

# HF_HUB_OFFLINE=1 ~/.venv-vllm-metal/bin/python ./env_tests/test_vllm_metal_mlx.py 2>&1 | tee ./env_tests/vllm-metal-mlx-test.log | tail -25