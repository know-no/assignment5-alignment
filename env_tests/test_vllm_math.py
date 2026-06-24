from vllm import LLM, SamplingParams
# Sample prompts.

def func():
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Create a sampling params object, stopping generation on newline.
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"]
    )

    # Create an LLM.
    llm = LLM(model="mlx-community/Qwen2.5-0.5B-Instruct-bf16")

    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


# HF_HUB_OFFLINE=1 ~/.venv-vllm-metal/bin/python3 ./env_tests/test_vllm_math.py

# 这个func()代码直接运行会发生：
    # raise RuntimeError('''
# RuntimeError: 
    # raise RuntimeError(
# RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {'EngineCore': 1}

# macOS 上 multiprocessing 用 spawn 而不是 fork，需要把代码包在 if __name__ == '__main__':

if __name__ == '__main__':
    func()