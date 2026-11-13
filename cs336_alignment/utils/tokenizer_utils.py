
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


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """编码 prompt 和回答，返回已对齐的下一 token 输入、标签及回答掩码。

    prompt 遵循传入 tokenizer 的特殊 token 配置。例如 Llama 可自动加 BOS，
    当前 Qwen tokenizer 会直接编码文本。回答是 prompt 的续写，编码时关闭
    自动特殊 token，避免在两段之间再次插入 BOS。

    prompt 至少需要一个编码后的上下文 token。空回答对应全 False 的掩码，
    空批次返回形状 (0, 0) 的张量。
    """
    if len(prompt_strs) != len(output_strs):
        raise ValueError("prompt_strs 与 output_strs 长度必须一致")

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    all_ids = []
    output_start_idxs = []
    output_sizes = []
    for prompt, output in zip(prompt_strs, output_strs):
        p_ids = tokenizer.encode(prompt, add_special_tokens=True)
        o_ids = tokenizer.encode(output, add_special_tokens=False)
        if not p_ids:
            raise ValueError("prompt 编码后至少需要一个上下文 token")

        # labels 比完整序列左移一位，回答首 token 的标签位置为 len(p_ids)-1。
        output_start_idxs.append(len(p_ids) - 1)
        output_sizes.append(len(o_ids))
        all_ids.append(p_ids + o_ids)

    max_len = max((len(ids) for ids in all_ids), default=1)
    sequence_length = max_len - 1
    batch_size = len(all_ids)
    input_ids = []
    labels = []
    response_mask = []
    for ids in all_ids:
        if len(ids) < max_len and pad_id is None:
            raise ValueError("补齐序列需要 tokenizer.pad_token_id 或 eos_token_id")
        padded = ids + [pad_id] * (max_len - len(ids))
        input_ids.append(padded[:-1])
        labels.append(padded[1:])

    for start, size in zip(output_start_idxs, output_sizes):
        mask = [False] * sequence_length
        mask[start:start + size] = [True] * size
        response_mask.append(mask)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long).reshape(batch_size, sequence_length),
        "labels": torch.tensor(labels, dtype=torch.long).reshape(batch_size, sequence_length),
        "response_mask": torch.tensor(response_mask, dtype=torch.bool).reshape(batch_size, sequence_length),
    }











    

    

    





