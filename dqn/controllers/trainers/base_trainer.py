import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
import wandb


class BaseTrainer(ABC):
    def __init__(self, config) -> None:
        self.config = config
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def get_device() -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    @staticmethod
    def init_wandb(project: str, entity: str, config: Dict[str, Any]) -> None:
        if wandb.run is None:
            wandb.login(key=os.environ.get("WANDB_API_KEY", ""))
            wandb.init(project=project, entity=entity, config=config)

    @staticmethod
    def log_metrics(episode: int, reward: float, loss: float, best_reward: float, extra: Dict[str, Any] = None) -> None:
        """
        Unified method for logging metrics to wandb.
        """
        data = {
            "episode": episode,
            "reward": reward,
            "loss": loss,
            "best_reward": best_reward
        }
        if extra:
            data.update(extra)

        wandb.log(data)

    def save_checkpoint(self, model: torch.nn.Module, episode: int, reward: float, filename: Any) -> None:
        """
        Saves a model checkpoint.
        """
        checkpoint = {
            'episode': episode,
            'model_state_dict': model.state_dict(),
            'reward': reward,
            'config': self.config.dict()
        }
        torch.save(checkpoint, filename)
        self.logger.info(f"Saved checkpoint at episode {episode} to {filename}")

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def evaluate(self) -> None:
        pass
