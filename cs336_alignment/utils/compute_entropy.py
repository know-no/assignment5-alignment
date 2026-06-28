import torch
import torch.nn.functional as F


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