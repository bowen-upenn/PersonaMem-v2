"""Custom VERL extensions for Werewolf RFT project."""

from .default_compute_score import default_compute_score
from .load_reward_manager import load_reward_manager
from .naive_reward_manager import CustomNaiveRewardManager
from .ray_trainer import RayPPOTrainer
from verl.workers.reward_manager import get_reward_manager_cls, register, NaiveRewardManager, BatchRewardManager, DAPORewardManager, PrimeRewardManager

# Note(haibin.lin): no need to include all reward managers here in case of complicated dependencies
__all__ = [
    "BatchRewardManager",
    "DAPORewardManager",
    "NaiveRewardManager",
    "CustomNaiveRewardManager",
    "PrimeRewardManager",
    "register",
    "RayPPOTrainer",
    "get_reward_manager_cls",
    "default_compute_score",
    "load_reward_manager",
]