import os
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.optim as optim
from torch.distributions import Categorical

from dqn.controllers.trainers.base_trainer import BaseTrainer
from dqn.controllers.utils.arm_env import ArmEnv
from dqn.controllers.networks.action_critic_model import ActorCriticNetwork


class ActorCriticTrainer(BaseTrainer):
    """
    Trainer for the Actor–Critic method.
    """
    def __init__(self, config: Any) -> None:
        super().__init__(config)
        BaseTrainer.init_wandb(config.wandb_project, config.wandb_entity, config.dict())

        script_dir = Path(__file__).parent.parent
        self.models_dir = script_dir / 'models'
        os.makedirs(self.models_dir, exist_ok=True)
        self.episode_model_path = self.models_dir / 'episode_actor_critic_model.pth'
        self.done_model_path = self.models_dir / 'done_actor_critic_model_{episode}_{timestamp}.pth'

        # Set up device
        self.device = BaseTrainer.get_device()
        self.logger.info(f"Using device: {self.device}")

        # Initialize environment and determine dimensions
        self.env = ArmEnv.initialize_supervisor(self.config.max_steps)
        self.state_dim = len(self.env.motors) + 3
        self.action_dim = len(self.env.motors) * 2

        # Initialize the Actor–Critic network and optimizer
        self.ac_net = ActorCriticNetwork(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.ac_net.parameters(), lr=config.learning_rate)
        self.best_reward = float('-inf')
        self.gamma = config.gamma
        self.value_loss_coef = config.value_loss_coef

        # Save the config object for later use
        self.config = config

    def train(self) -> None:
        """
        Trains the Actor–Critic model.
        """
        self.logger.info(f"Starting Actor–Critic training at {datetime.now().isoformat()}")

        for episode in range(self.config.episodes):
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            log_probs, values, rewards = [], [], []
            episode_reward: float = 0.0

            for step in range(self.config.max_steps):
                action_probs, state_value = self.ac_net(state.unsqueeze(0))
                dist = Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                log_probs.append(log_prob)
                values.append(state_value.squeeze(0))

                next_state, reward, done, info = self.env.step(action.item())
                next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
                rewards.append(reward)
                episode_reward += reward
                state = next_state

                if done:
                    self.logger.info(f"Episode {episode}/{self.config.episodes} done with reward {episode_reward}")
                    break

            # Compute discounted returns for the episode
            returns = []
            G = 0.0
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

            values = torch.stack(values)
            log_probs = torch.stack(log_probs)
            advantages = returns - values

            actor_loss = - (log_probs * advantages.detach()).sum()
            critic_loss = advantages.pow(2).sum()
            loss = actor_loss + self.value_loss_coef * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if episode_reward > self.best_reward:
                self.best_reward = episode_reward

            if episode % 100 == 0 and episode > 0:
                checkpoint_file = self.models_dir / f'checkpoint_actor_critic_episode_{episode}.pth'
                self.save_checkpoint(self.ac_net, episode, episode_reward, checkpoint_file)

            if episode == self.config.episodes - 1:
                final_file = self.models_dir / 'final_ac_model.pth'
                self.save_checkpoint(self.ac_net, episode, episode_reward, final_file)

            self.logger.info(
                f"Episode {episode}/{self.config.episodes} "
                f"({(episode + 1) / self.config.episodes:.1%} complete) | "
                f"Reward: {episode_reward:.2f} | "
                f"Loss: {loss.item():.4f} | "
                f"Best: {self.best_reward:.2f}"
            )

            self.log_metrics(episode, episode_reward, loss.item(), self.best_reward)

        self.logger.info(f"Training completed at {datetime.now().isoformat()}")
        self.logger.info(f"Best reward achieved: {self.best_reward}")

    def evaluate(self) -> None:
        """
        Evaluates the Actor–Critic model.
        """
        self.logger.info("Evaluation not implemented yet.")
        pass
