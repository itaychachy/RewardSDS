import gc
import torch
import random
import os
import numpy as np
from reward_strategy import (
    NoRewardStrategy,
    MinMaxRewardStrategy,
    BestRewardStrategy,
    WeightedRewardStrategy
)
from reward_model import ImageReward, AestheticReward


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def tensor_from_indices(tensor, indices):
    if not isinstance(indices, tuple):
        return tensor[indices]

    output = torch.cat([tensor[idx] for idx in indices])
    return output


def create_reward_strategy(strategy, n_noises):
    if strategy == 'none':
        return NoRewardStrategy()
    elif strategy == 'min-max':
        return MinMaxRewardStrategy(n_noises)
    elif strategy == 'best':
        return BestRewardStrategy(n_noises)
    elif strategy == "weighted":
        return WeightedRewardStrategy(n_noises)
    raise ValueError(f"Unknown reward strategy: {strategy}")


def create_reward_model(model):
    if model == 'image-reward':
        return ImageReward()
    elif model == 'aesthetic':
        return AestheticReward()
    elif model == 'none':
        return None
    raise ValueError(f"Unknown reward model: {model}")
