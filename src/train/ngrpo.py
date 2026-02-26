"""
NGRPO (Virtual Perfect Sample GRPO) 实现

当 GRPO 训练中出现以下情况时需要切换到 NGRPO：
  - reward_std ≈ 0（组内回答同质化）
  - frac_reward_zero_std 很高（大部分组的方差接近零）

原理：
  GRPO: advantage = (r_i - mean(r)) / std(r)
  问题: 如果 group 内所有回答都拿 0.3 分，std ≈ 0，advantage ≈ 0，模型不学习
  NGRPO: 在组内加一个虚拟满分样本（如 1.0），强行拉大 std
         只对真实样本计算 advantage，虚拟样本不参与梯度更新

参考实现:
  https://github.com/nangongrui-ngr/NGRPO/blob/ngr/verl/trainer/ppo/core_algos.py#L156

NGRPO 论文:
  https://arxiv.org/abs/2509.18851

与现有代码的关系:
  src/train/common.py 中的 normalize_group_rewards() 已有基础 NGRPO 逻辑
  本文件提供更完整的 PyTorch tensor 版本，可直接嵌入 verl 的 core_algos
"""

from __future__ import annotations

from typing import Optional, Tuple

try:
    import torch
except ImportError:
    torch = None  # type: ignore


def compute_grpo_advantages(
    rewards: "torch.Tensor",
    eps: float = 1e-8,
) -> Tuple["torch.Tensor", float]:
    """
    标准 GRPO advantage 计算

    Args:
        rewards: shape (group_size,) 组内各回答的 reward
        eps: 数值稳定

    Returns:
        advantages: shape (group_size,) 归一化后的 advantage
        std: 组内 reward 标准差
    """
    assert torch is not None, "需要 PyTorch"

    mean = rewards.mean()
    std = rewards.std()

    if std <= eps:
        return torch.zeros_like(rewards), float(std)

    advantages = (rewards - mean) / (std + eps)
    return advantages, float(std)


def compute_ngrpo_advantages(
    rewards: "torch.Tensor",
    virtual_max_reward: float = 1.0,
    eps: float = 1e-8,
) -> Tuple["torch.Tensor", float]:
    """
    NGRPO: 添加虚拟满分样本的 advantage 计算

    核心思路：
    1. 在 rewards 后面追加一个 virtual_max_reward
    2. 用 augmented 序列计算 mean 和 std
    3. 只对原始样本计算 advantage（虚拟样本不参与梯度）

    Args:
        rewards: shape (group_size,) 组内各回答的 reward
        virtual_max_reward: 虚拟满分值（通常为 1.0）
        eps: 数值稳定

    Returns:
        advantages: shape (group_size,) 只有真实样本的 advantage
        std: augmented 序列的标准差
    """
    assert torch is not None, "需要 PyTorch"

    # 添加虚拟满分样本
    virtual = torch.tensor([virtual_max_reward], device=rewards.device, dtype=rewards.dtype)
    augmented = torch.cat([rewards, virtual])

    mean = augmented.mean()
    std = augmented.std()

    if std <= eps:
        return torch.zeros_like(rewards), float(std)

    # 只对原始样本计算 advantage
    advantages = (rewards - mean) / (std + eps)
    return advantages, float(std)


def compute_ngrpo_advantages_asymmetric_clip(
    rewards: "torch.Tensor",
    virtual_max_reward: float = 1.0,
    clip_pos: float = 0.24,
    clip_neg: float = 0.16,
    eps: float = 1e-8,
) -> Tuple["torch.Tensor", float]:
    """
    NGRPO + 非对称 clip

    NGRPO 论文推荐的完整版：
    - 正 advantage 用更大的 clip 范围 (ε_pos = 0.24)
    - 负 advantage 用更小的 clip 范围 (ε_neg = 0.16)
    - 这样对好的回答给更多学习信号，对差的回答限制更新幅度

    返回的 advantages 已经过非对称 clip，可以直接用于 PPO loss 计算。
    """
    assert torch is not None, "需要 PyTorch"

    advantages, std = compute_ngrpo_advantages(rewards, virtual_max_reward, eps)

    # 非对称 clip
    pos_mask = advantages > 0
    neg_mask = advantages <= 0

    clipped = torch.zeros_like(advantages)
    clipped[pos_mask] = torch.clamp(advantages[pos_mask], max=clip_pos / eps)
    clipped[neg_mask] = torch.clamp(advantages[neg_mask], min=-clip_neg / eps)

    return clipped, std


def compute_adaptive_advantages(
    rewards: "torch.Tensor",
    virtual_max_reward: float = 1.0,
    ngrpo_threshold: float = 0.1,
    eps: float = 1e-8,
) -> Tuple["torch.Tensor", float, bool]:
    """
    自适应：先尝试 GRPO，如果 std 太低自动切换 NGRPO

    Args:
        rewards: 组内 reward
        virtual_max_reward: NGRPO 虚拟满分
        ngrpo_threshold: std 低于此阈值时切换 NGRPO
        eps: 数值稳定

    Returns:
        advantages: 归一化 advantage
        std: 标准差
        used_ngrpo: 是否使用了 NGRPO
    """
    advantages, std = compute_grpo_advantages(rewards, eps)

    if std <= ngrpo_threshold:
        advantages, std = compute_ngrpo_advantages(rewards, virtual_max_reward, eps)
        return advantages, std, True

    return advantages, std, False


# =============================================================
#  批量版本（处理整个 batch 的多个 group）
# =============================================================


def compute_batch_advantages(
    rewards: "torch.Tensor",
    group_size: int,
    use_ngrpo: bool = False,
    virtual_max_reward: float = 1.0,
    ngrpo_threshold: float = 0.1,
    eps: float = 1e-8,
) -> Tuple["torch.Tensor", "torch.Tensor", int]:
    """
    批量计算 advantage

    Args:
        rewards: shape (batch_size * group_size,) 或 (batch_size, group_size)
        group_size: 每组采样数
        use_ngrpo: 是否强制使用 NGRPO
        virtual_max_reward: NGRPO 虚拟满分
        ngrpo_threshold: 自适应模式下的阈值
        eps: 数值稳定

    Returns:
        advantages: 同 rewards shape
        stds: shape (num_groups,) 每组的 std
        ngrpo_count: 使用了 NGRPO 的组数
    """
    assert torch is not None, "需要 PyTorch"

    original_shape = rewards.shape
    if rewards.dim() == 1:
        rewards = rewards.view(-1, group_size)

    num_groups = rewards.shape[0]
    all_advantages = torch.zeros_like(rewards)
    all_stds = torch.zeros(num_groups, device=rewards.device)
    ngrpo_count = 0

    for i in range(num_groups):
        group_rewards = rewards[i]

        if use_ngrpo:
            adv, std = compute_ngrpo_advantages(group_rewards, virtual_max_reward, eps)
            ngrpo_count += 1
        else:
            adv, std, used = compute_adaptive_advantages(
                group_rewards, virtual_max_reward, ngrpo_threshold, eps
            )
            if used:
                ngrpo_count += 1

        all_advantages[i] = adv
        all_stds[i] = std

    return all_advantages.view(original_shape), all_stds, ngrpo_count
