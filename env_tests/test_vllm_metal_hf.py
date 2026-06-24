from vllm import LLM, SamplingParams


def main():
    # Use the locally-cached HF model (no network).
    llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct")
    prompts = ["The capital of France is"]
    sp = SamplingParams(temperature=0.0, max_tokens=20)
    outputs = llm.generate(prompts, sp)
    for o in outputs:
        print("PROMPT:", o.prompt)
        print("OUTPUT:", o.outputs[0].text)
    print("OK")


if __name__ == "__main__":
    main()


# ~/.venv-vllm-metal 是 vllm-metal 官方脚本 install.sh 自动创建的
# HF_HUB_OFFLINE=1 ~/.venv-vllm-metal/bin/python ./env_tests/test_vllm_metal_hf.py 2>&1 | tee ./env_tests/vllm-metal-hf-test.log | tail -40