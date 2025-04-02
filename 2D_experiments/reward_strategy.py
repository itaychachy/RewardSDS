from dataclasses import dataclass
from abc import ABC, abstractmethod
import torch

@dataclass
class MinMaxConfig:
    lambda_max: float = 0.9
    lambda_min: float = 0.1
    # Note that the sum of n_max and n_min must be less than or equal to total number of noises
    n_max: int = 2
    n_min: int = 2

    assert lambda_max >= 0 and lambda_min >= 0, "Lambda values must be non-negative"
    assert lambda_max + lambda_min == 1., "Sum of lambda values must be 1"

@dataclass
class BestConfig:
    n_best: int = 1

class RewardStrategy(ABC):
    def __init__(self, n_noises: int):
        self.n_noises = n_noises

    @abstractmethod
    def extract_loss(self, losses, rewards):
        pass

    @abstractmethod
    def extract_indices_to_consider(self, rewards):
        """
        Instead of considering all noises, this method returns the indices of noises to consider, based on the reward values.
        """
        pass

class NoRewardStrategy(RewardStrategy):
    def __init__(self):
        super().__init__(n_noises=1)

    def extract_loss(self, losses, rewards):
        return losses[0]

    def extract_indices_to_consider(self, rewards):
        return 0


class MinMaxRewardStrategy(RewardStrategy):
    def __init__(self, n_noises: int, min_max_config: MinMaxConfig=MinMaxConfig()):
        assert n_noises >= min_max_config.n_max + min_max_config.n_min, "Number of noises to consider must be less than or equal to total number of noises"
        super().__init__(n_noises)
        self.config = min_max_config


    def extract_loss(self, losses, rewards):
        n_max, n_min = self.config.n_max, self.config.n_min
        assert len(losses) == n_max + n_min, "Total number of losses should match the number of noises to consider"
        l1, l2 = self.config.lambda_max, self.config.lambda_min
        loss = (l1 * losses[:n_max].mean()) - (l2 * losses[-n_min:].mean())
        return loss

    def extract_indices_to_consider(self, rewards):
        n_max, n_min = self.config.n_max, self.config.n_min
        _, k_i = torch.topk(rewards, k=n_max, largest=True)
        _, l_i = torch.topk(rewards, k=n_min, largest=False)
        return k_i, l_i

class BestRewardStrategy(RewardStrategy):
    def __init__(self, n_noises: int, best_config: BestConfig=BestConfig()):
        assert n_noises >= best_config.n_best, "Number of best noises must be less than or equal to total number of noises"

        super().__init__(n_noises)
        self.config = best_config

    def extract_loss(self, losses, rewards):
        return losses.mean()

    def extract_indices_to_consider(self, rewards):
        return torch.topk(rewards, k=self.config.n_best, largest=True)[1]


class WeightedRewardStrategy(RewardStrategy):
    def __init__(self, n_noises: int):
        super().__init__(n_noises)

    def extract_loss(self, losses, rewards):
        exp_scores = torch.exp(rewards)
        weights = exp_scores / exp_scores.sum()
        return torch.sum(losses * weights)

    def extract_indices_to_consider(self, rewards):
        return torch.arange(len(rewards))
