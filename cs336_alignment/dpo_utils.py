import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

from cs336_alignment.utils.prompt_template_reader import read_prompt_template, alpaca_sft
from cs336_alignment.utils.tokenizer_utils import tokenize_prompt_and_output
from cs336_alignment.utils.compute_entropy import get_response_log_probs

template = read_prompt_template(alpaca_sft)

def compute_per_instance_dpo_loss(
            lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
    ):
    # 在非cpu设备上必须要
    lm_device = next(lm.parameters()).device
    ref_device = next(lm_ref.parameters()).device

    prefix = template.split("{response}", 1)[0].format(instruction=prompt)
    _tokenize_res = tokenize_prompt_and_output(
        prompt_strs = [prefix, prefix],
        output_strs = [response_chosen + tokenizer.eos_token, response_rejected + tokenizer.eos_token],
        tokenizer=tokenizer
    )
    response_mask = _tokenize_res["response_mask"]
    labels = _tokenize_res["labels"]
    input_ids = _tokenize_res["input_ids"]

    _res : dict[str, torch.Tensor]  = get_response_log_probs(
        model=lm,
        input_ids=input_ids.to(lm_device),
        labels=labels.to(lm_device),
        return_token_entropy=True
    )

    log_probs= _res["log_probs"] #(B, L) B==2
    token_entropy = _res["token_entropy"]
    log_probs_sum = (log_probs * response_mask.to(lm_device)).sum(dim = -1, keepdim=False) #(B,) (2,)

    # 参考模型不要梯度回传的
    with torch.no_grad():
        ref_res : dict[str, torch.Tensor]  = get_response_log_probs(
            model=lm_ref,
            input_ids=input_ids.to(ref_device),
            labels=labels.to(ref_device),
            return_token_entropy=True
        )

    ref_log_probs= ref_res["log_probs"] #(B, L) B==2
    ref_token_entropy = ref_res["token_entropy"]
    ref_log_probs_sum = (ref_log_probs * response_mask.to(ref_device)).sum(dim = -1, keepdim=False) #(B,) (2,)
    ref_log_probs_sum = ref_log_probs_sum.to(lm_device)

    z = beta * ((log_probs_sum[0] - log_probs_sum[1]) - (ref_log_probs_sum[0] - ref_log_probs_sum[1]))

    return -F.logsigmoid(z)
