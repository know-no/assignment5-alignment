import torch
import torch.nn.functional as F
from transformers import PreTrainedModel


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    获取下一 token 预测的熵（手动实现，不使用 F.softmax / F.log_softmax / torch.logsumexp）。

    Args:
        logits: torch.Tensor 形状为 (batch_size, sequence_length, vocab_size) 的包含未归一化 logits 的张量。

    Returns:
        torch.Tensor 形状为 (batch_size, sequence_length)。每个下一 token 预测的熵。
    """
    # 手动 logsumexp：log_z = v_max + log(sum(exp(logits - v_max)))
    v_max = torch.max(logits, dim=-1, keepdim=True).values
    shifted = logits - v_max
    exp_shifted = torch.exp(shifted)
    sum_exp = torch.sum(exp_shifted, dim=-1, keepdim=True)
    log_z = v_max + torch.log(sum_exp)  # (..., 1) z 是 sumexp 的值

    # log p = logits - log_z；p = exp(log p)； logits是nn.Linear 的输出，有正有负，不是概率, 它把【0，1】的概率映射到了-infinity，+infinity的实数轴
    # 可以理解成"未归一化的对数概率"——它正比于 $\log p$，但差一个常数 $\log Z$： {logits}_i = log p_i + log Z
    log_probs = logits - log_z # 把"未归一化分数"减掉归一化常数，才变成真正的对数概率; 回归到log 运算内的除法，则是缩放
    probs = exp_shifted / sum_exp  # 等价于 softmax，复用已算的 exp_shifted

    # H(p) = -sum p * log p
    return -torch.sum(probs * log_probs, dim=-1)


def compute_entropy1(logits: torch.Tensor) -> torch.Tensor:
    """
    获取下一 token 预测的熵（调用库版本，使用 torch.nn.functional）。
    """
    log_probs = F.log_softmax(logits, dim=-1)
    probs = F.softmax(logits, dim=-1)
    # H(p) = -sum p * log p
    return -torch.sum(probs * log_probs, dim=-1)

import torch


def compute_entropy2(logits: torch.Tensor) -> torch.Tensor:
    """
    获取下一 token 预测的熵（即，词汇表维度上的熵）。

    Args:
        logits: torch.Tensor 形状为 (batch_size, sequence_length, vocab_size) 的包含未归一化 logits 的张量。

    Returns:
        torch.Tensor 形状为 (batch_size, sequence_length)。每个下一 token 预测的熵。
    """
    # H(p) = -sum p * log p
    #      = -sum softmax(logits) * (logits - logsumexp(logits))
    #      = logsumexp(logits) - sum softmax(logits) * logits
    log_z = torch.logsumexp(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    return log_z - torch.sum(probs * logits, dim=-1)


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Args:
        model: PreTrainedModel 用于评分的 HuggingFace 模型（放置在正确的设备上，并在不需要计算梯度时处于推理模式）。
        input_ids: torch.Tensor 形状为 (batch_size, sequence_length)，由你的分词方法产生的连接后的提示+响应 token。
        labels: torch.Tensor 形状为 (batch_size, sequence_length)，由你的分词方法产生的标签。
        return_token_entropy: bool 如果为 True，则通过调用 compute_entropy 也返回每 token 熵。
        我在 tokenizer_utils 里实现的分词方法，是右移的， hf 里实现的是左移的; sequence_length 是 input_id 的 size，input_id 是原始 input 的 size-1;
        见 tokenizer_utils.py 里，input id 去掉了最后一个值; 但是这里的sequence_length实际上指的是，去掉最后一个值之后的长度
    Returns:
        dict[str, torch.Tensor]。
        "log_probs" 形状为 (batch_size, sequence_length)，条件对数概率 log pθ(xt | x<t)。
        "token_entropy" 可选，形状为 (batch_size, sequence_length)，每个位置的每 token 熵（如果 return_token_entropy=True 则存在）。
    """
    logits = model(input_ids).logits .float() # (B, L, V) V是 vocabulary 的 size # 使用 float 上溢，计算过程是正确的，但是结果不对,加了顺便让 bf16 模型输出能转 numpy（snapshot 测试用）； 直接改模型加载参数类型也可以过测试；
    log_probs_all = F.log_softmax(logits, -1) # (B, L, V)
    # log_probs_all.gather 是 torch 里面【按索引取值】操作，在这里是：从每个位置的整个词表分布里，挑选出真实 label 那个 token 对应的 log 概率
    log_probs = log_probs_all.gather(
        -1, labels.unsqueeze(-1) 
    ).squeeze(-1) # (B, L)

    result = {"log_probs": log_probs}
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)
    return result


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """
    对一个维度求和并用一个常数进行归一化，只考虑掩码为 1 的元素。

    Args:
        tensor: torch.Tensor 要求和和归一化的张量。
        mask: torch.Tensor 与 tensor 形状相同；掩码为 1 的位置包含在求和中。
        normalize_constant: float 用于归一化的常数。
        dim: int | None 要在归一化前求和的维度。如果为 None，则对所有维度求和。

    Returns:
        torch.Tensor 归一化的和，其中掩码元素（mask==0）不贡献于求和。
    """
    masked = (tensor * mask)
    return masked.sum(dim=dim) / normalize_constant