
import torch

from cs336_alignment.utils import compute_entropy


# UPDATE_SNAPSHOTS=1 会将当前输出保存为新的 .npz 参考值。
# 仓库最初为 Mac/BF16 下的本地基线重建加入此开关（提交 37b5cfb）。
# test_get_response_log_probs 包含模型前向，输出会受模型 dtype 和计算后端影响。
# 独立确认结果正确后，可仅对该模型测试开启更新，建立对应环境的本地参考。
# 本函数的三个单元测试直接传入 log-prob 张量，常规验证使用已有快照比较 loss 和梯度。
# 精度对照、更新命令和验证步骤见 notes/sft-loss-normalization.md 的快照小节。

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """计算一个 microbatch 的 SFT 损失并反向传播。

    上游已通过 log_softmax 和 labels 索引得到标准答案 token 的 log-prob。
    对 one-hot 目标，多分类交叉熵 -sum(q_i * log(p_i)) 简化为 -log(p_gold)。
    取负号后，提高标准答案的概率会降低损失。

    记 B 为当前 microbatch 的样本数，C 为 normalize_constant，
    K 为 gradient_accumulation_steps，本函数计算
        loss = -sum(response_mask * policy_log_probs) / (B * C * K)

    各 microbatch 大小相同时，除以 K 使 K 次反传累积出平均损失的梯度。
    外层训练循环在累积完成后调用 optimizer.step() 并清理梯度。

    Args:
        policy_log_probs: (batch_size, sequence_length)，标准答案 token 的对数概率，
            需要保留连接到策略模型参数的计算图。
        response_mask: (batch_size, sequence_length)，对响应 token 为 1，对提示/填充为 0。
        gradient_accumulation_steps: 每个优化器步骤的微批次数量。
        normalize_constant: 对 token 求和后的固定归一化常数 C，可以设为 1.0。

    Returns:
        loss: 已除以 K 并执行 backward() 的标量损失。
        metadata: unscaled_loss 是除以 K 前的样本平均损失，
            n_response_tokens 是当前 microbatch 的有效响应 token 数。
            两个日志张量均已 detach。
    """
    batch_size = policy_log_probs.shape[0]

    # dim=None 将 B、L 两个维度全部求和，得到除以 C 后的负对数似然总和。
    neg_log_lik = -compute_entropy.masked_normalize(
        tensor=policy_log_probs,
        mask=response_mask,
        normalize_constant=normalize_constant,
        dim=None,
    )
    unscaled_loss = neg_log_lik / batch_size
    loss = unscaled_loss / gradient_accumulation_steps

    # 对 log-prob 的梯度为 -response_mask / (B*C*K)。
    # B=2、C=1、K=2 时，有效响应位置的梯度为 -0.25，屏蔽位置为 0。
    # backward() 将梯度沿计算图累加到参数的 .grad。
    loss.backward()

    metadata = {
        "unscaled_loss": unscaled_loss.detach(),
        "n_response_tokens": response_mask.sum().detach(),
    }
    return loss, metadata



