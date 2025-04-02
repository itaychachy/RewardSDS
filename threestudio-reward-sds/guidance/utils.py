from .reward_strategy import (
    NoRewardStrategy,
    MinMaxRewardStrategy,
    BestRewardStrategy,
    WeightedRewardStrategy
)
from .reward_model import ImageReward, AestheticReward


def config_to_reward_strategy(cfg):
    strategy = cfg.reward_strategy
    if strategy == 'none':
        return NoRewardStrategy()
    elif strategy == 'min-max':
        return MinMaxRewardStrategy(cfg.n_noises)
    elif strategy == 'best':
        return BestRewardStrategy(cfg.n_noises)
    elif strategy == "weighted":
        return WeightedRewardStrategy(cfg.n_noises)
    raise ValueError(f"Unknown reward strategy: {strategy}")


def config_to_reward_model(cfg):
    model = cfg.reward_model
    if model == 'image-reward':
        return ImageReward()
    elif model == 'aesthetic':
        return AestheticReward()
    elif model == 'none': # No reward
        return None
    raise ValueError(f"Unknown reward model: {model}")
