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

    # logits 是模型输出的实数分数。对同一位置，所有词表项共享归一化常数 log_z。
    # log p_i = logits_i - log_z，对应 p_i = exp(logits_i) / sum_j exp(logits_j)。
    # log_probs = logits - log_z
    # probs = exp_shifted / sum_exp  # 等价于 softmax，复用已算的 exp_shifted

    # 当 logits 带有很大的共同偏移时，计算 log_z 会把较小的归一化项舍入掉，得到错误的 log-prob,比如 logits = [1e8, 1e8, 1e8]; 当 v_max=1e8 时，加上的 log(3)≈1.0986 可能被 FP32 舍入掉，随后相减就错误地得到 0。
    # 所以，可以直接
    log_probs = shifted - torch.log(sum_exp)
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
        model: PreTrainedModel 用于评分的 HuggingFace 模型，需放置在正确的设备上。
            调用方分别控制 train/eval 模式与是否记录梯度；本函数不会主动切换模式或关闭梯度记录。
        input_ids: torch.Tensor 形状为 (batch_size, sequence_length)，由你的分词方法产生的连接后的提示+响应 token。
        labels: torch.Tensor 形状为 (batch_size, sequence_length)，由完整“提示+回答”token 序列取 [1:] 得到。
            其中参与回答损失的 token，在 SFT 中来自数据集标准回答；
            在 GRPO 中来自模型采样的 rollout 回答，可能包含错误答案。
            例如题目“1+1”的标准答案为“2”，若 rollout 生成“3”，就对“3”的 token 计算 log-prob。
            GRPO 的 ground_truths 用于奖励函数判分，不直接替换 rollout 对应的 labels；
            后续损失通过 response_mask 选中回答位置，再结合奖励或优势决定如何调整概率。
        return_token_entropy: bool 如果为 True，则通过调用 compute_entropy 也返回每 token 熵。
        分词阶段已将 input_ids 设为拼接序列的 [:-1]，labels 设为 [1:]。
        因而 input_ids 位置 t 的模型输出与 labels 位置 t 的下一个 token 已对齐。
    Returns:
        dict[str, torch.Tensor]。
        "log_probs" 形状为 (batch_size, sequence_length)，条件对数概率 log pθ(xt | x<t)。
        "token_entropy" 可选，形状为 (batch_size, sequence_length)，每个位置的每 token 熵（如果 return_token_entropy=True 则存在）。
    """
    # 将前向输出转为 float32，供后续 log_softmax、熵计算和 NumPy 快照比较使用。
    logits = model(input_ids).logits.float()  # (B, L, V)，V 是词表大小。
    log_probs_all = F.log_softmax(logits, -1) # (B, L, V)
    # labels 指定本次要评分的 token，从每个位置的词表分布中取出它的 log-prob。
    # 后续 SFT 训练步对这些值取负，即得到 one-hot 目标的逐 token 交叉熵。
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
