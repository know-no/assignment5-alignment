import pathlib
import os

cs336_alignment_path = os.path.dirname(os.path.dirname(__file__))
base_path = os.path.join(cs336_alignment_path, "prompts")
alpaca_sft = "alpaca_sft"
question_only = "question_only"
r1_zero = "r1_zero"
zero_shot_system_prompt = "zero_shot_system_prompt"
prompts = set()
prompts.add(alpaca_sft)
prompts.add(question_only)
prompts.add(r1_zero)
prompts.add(zero_shot_system_prompt)


def read_prompt_template(template_name: str) -> str:
    if template_name not in prompts:
        raise Exception(f"不存在的 prompt 模板 {template_name}")
    template_file = template_name + '.prompt'
    file_path = os.path.join(base_path, template_file)

    content : str = None
    with open(file_path) as f:
        content = f.read()
    return content

if __name__ == '__main__':

    r1_zero_content = read_prompt_template(r1_zero)
    print(f"r1_zero template is: \n{r1_zero_content}")
