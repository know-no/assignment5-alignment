# 知识点笔记索引

本目录沉淀对话中浮现的、非琐碎、值得日后回顾的技术知识点。每个文件聚焦一个或一小组紧密相关的概念,文末给出**外延拓展方向**便于按图索骥。

写入规则见仓库根 `.claude/CLAUDE.md` 的「知识蒸馏」一节。

## 索引

| 文件 | 主题 | 关联代码 |
|---|---|---|
| [sft-loss-normalization.md](sft-loss-normalization.md) | SFT loss 全链路:**作用域**(为何只对 response 求 loss,与预训练的区别)+ 掩码归一化(固定常数、`dim=None` 语义)+ microbatch 训练步(梯度累积等价性、四大坑) | [tokenizer_utils.py:21](../cs336_alignment/utils/tokenizer_utils.py#L21) `tokenize_prompt_and_output`;[compute_entropy.py:94](../cs336_alignment/utils/compute_entropy.py#L94) `masked_normalize`;[sft_utils.py:5](../cs336_alignment/utils/sft_utils.py#L5) `sft_microbatch_train_step` |
| [inline-generation-logging.md](inline-generation-logging.md) | 训练循环内采样日志:generate_fn 解耦、熵按 `mask.sum(-1)` 归一化(与训练 loss 归一化的对照)、正误分桶用 `answer_reward` | [log_utils.py](../cs336_alignment/utils/log_utils.py) `log_generations` |
