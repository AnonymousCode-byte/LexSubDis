'''
(batch_size,num_subwords,vocabsize)
weights同样支持自定义，也支持none
'''

import torch
from torch import Tensor
from typing import Literal, Optional, Union

def outputlogits_stategy(
    logits: Tensor,
    weights: Optional[Union[Tensor, Literal["mean", "first", "linear", "exponential"]]] = "first",
    decay_rate: float = 0.5
) -> Tensor:
    """批量加权求和函数（支持指数衰减）
    
    Args:
        logits: 形状为 (batch_size, num_subwords, vocab_size) 的Tensor
        weights: 支持四种模式：
            - "mean": 均值池化 (默认)
            - "first": 取首个子词
            - "linear": 线性位置衰减权重
            - "exponential": 指数位置衰减权重
            或自定义权重Tensor（形状为 (batch_size, num_subwords)）
        decay_rate: 指数衰减系数（仅当weights="exponential"时生效），默认0.5，
                    值越大衰减越快（建议范围0.1-1.0）
    
    Returns:
        形状为 (batch_size, vocab_size) 的Tensor
    
    Examples:
        >>> # 指数衰减模式
        >>> weighted = weighted_sum(logits, "exponential", decay_rate=0.3)
        >>> 
        >>> # 自定义权重+自动模式混合使用
        >>> custom_weights = torch.tensor([[0.5, 0.3, 0.2], [1.0]])
        >>> weighted = weighted_sum(logits, custom_weights)
    """
    # 确保 logits 是合适的数据类型
    # logits = logits.to(torch.float64)  # 可以根据需要调整
    if isinstance(weights, str):
        batch_size, n_subwords, vocab_size = logits.size()
        device = logits.device
        # 平均池化，logits维度
        if weights == "mean":
            return logits.mean(dim=1)
            
        elif weights == "first":
            return logits[:, 0, :]
        
        elif weights=="max":    # 找出最大值的同时，还会记录这些最大值所在的索引位置
            return logits.max(dim=1)[0]
        elif weights=="min":
            return logits.min(dim=1)[0]
        
        elif weights == "linear":
            # 线性衰减权重 (1.0 → 0.5)
            pos_weights = torch.linspace(1.0, 0.5, n_subwords, device=device)
            
        elif weights == "exponential":
            # 指数衰减权重 exp(-decay_rate * position_index)
            position_indices = torch.arange(n_subwords, device=device)
            pos_weights = torch.exp(-decay_rate * position_indices)
            
        else:
            raise ValueError(f"Unsupported weight mode: {weights}. "
                             "Available options: ['mean', 'first', 'linear', 'exponential']")
        
        # 归一化处理并扩展维度，可能会有精度损失
        pos_weights = pos_weights / pos_weights.sum()
        return (logits * pos_weights[None, :, None]).sum(dim=1)
        
    else:
        # 自定义权重处理
        if weights.dim() != 2 or weights.size(0) != logits.size(0) or weights.size(1) != logits.size(1):
            raise ValueError(f"Weights shape must be (batch_size={logits.size(0)}, "
                             f"num_subwords={logits.size(1)}), but got {weights.shape}")
        return (logits * weights.unsqueeze(-1)).sum(dim=1)