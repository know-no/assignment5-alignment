使用原模式直接评测的几种典型错法:

case 1：模型直接写推理，完全没生成任何 </think> 标签，跑到 1024 token 截断
case 2：自创了 <end of think> 标签，不是 </think>
case 3：模型生成了 </think> 但没接 <answer>...</answer>，而是 A: To solve...

