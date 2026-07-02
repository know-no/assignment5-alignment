
import torch
from cs336_alignment.utils import compute_entropy


# 这个测试运行的时候， UPDATE_SNAPSHOTS=1， 因为也会用到get_response_log_probs方法
def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    在微批次上执行前向和反向传递。

    Args:
        policy_log_probs: (batch_size, sequence_length)，正在训练的 SFT 策略的每 token 对数概率。
        response_mask: (batch_size, sequence_length)，对响应 token 为 1，对提示/填充为 0。
        gradient_accumulation_steps: 每个优化器步骤的微批次数量。
        normalize_constant: 用于除法的常数。留为 1.0 也可以。

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]。
        loss: 标量张量。微批次损失，已根据梯度累积进行调整。我们返回它以便记录。
        metadata: 包含底层损失调用的元数据以及你可能想要记录的任何其他统计信息的字典。
    """
    neg_log_lik: torch.Tensor = compute_entropy.masked_normalize(tensor=policy_log_probs, mask=response_mask, normalize_constant=normalize_constant, dim = None)

    # 处理的是微批，这个批次里有多少条数据
    loss = neg_log_lik / gradient_accumulation_steps
    loss.backward()

    metadata = {
        "unscaled_loss": neg_log_lik.detach(),
        "n_response_tokens": response_mask.sum().detach(),
    }
    return loss, metadata



