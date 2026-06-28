
import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase




def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.backends.cuda.is_available():
        return torch.device("cuda") # todo: 要不要加 gpu 号？
    else:
        return torch.device("cpu")


# tokenizer.encode 默认会加 special tokens（如 BOS）。如果 prompt 和 output 都各自 encode 再拼，可能在中间多/少 BOS。
# 常见做法是 tokenizer.encode(p, add_special_tokens=False) 自己控制，或者直接 tokenizer(p + o) 整体 encode（但那样要重新算 prompt 的 token 数）。
# 课程参考实现一般用 add_special_tokens=False，建议确认下测试期望。
def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    all_ids = []
    prompt_and_output_lens = []
    max_len = 1 # todo：未考虑空字符串的情况
    output_start_idxs = []
    output_sizes = []
    for p, o in zip(prompt_strs,output_strs):
        p_ids = tokenizer.encode(text = p)
        o_ids = tokenizer.encode(text = o)
        output_start_idxs.append(len(p_ids))
        output_sizes.append(len(o_ids))
        full_ids = p_ids + o_ids
        max_len = max(max_len, len(full_ids))
        all_ids.append(full_ids)
        prompt_and_output_lens.append(len(full_ids))
    
    input_ids = []
    labels = []
    for ids in all_ids:
        padded = ids + [pad_id] * (max_len - len(ids))
        input_ids.append(padded[:-1])
        labels.append(padded[1:])

    response_mask = []
    for i in range(len(output_start_idxs)):
        output_start_idx = output_start_idxs[i]
        output_size = output_sizes[i]
        mask = [0 for _ in range(max_len - 1)]
        for j in range(output_start_idx - 1, output_start_idx - 1 + output_size):
            mask[j] = 1
        response_mask.append(mask)

    return {
        "input_ids": torch.tensor(input_ids, dtype = torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "response_mask": torch.tensor(response_mask, dtype = torch.bool ), # long is ok too
    }











    

    

    





