import os
import random
from datetime import datetime
from pathlib import Path
from typing import Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dqn.controllers.utils.arm_env import ArmEnv
from dqn.controllers.networks.dqn_model import QNetwork
from dqn.controllers.utils.replay_buffer import PrioritizedReplay
from dqn.controllers.trainers.base_trainer import BaseTrainer


class DQNTrainer(BaseTrainer):
    """
    Trainer for the DQN method.
    """
    def __init__(self, config: Any) -> None:
        super().__init__(config)
        BaseTrainer.init_wandb(config.wandb_project, config.wandb_entity, config.dict())

        script_dir = Path(__file__).parent.parent
        self.models_dir = script_dir / 'models'
        os.makedirs(self.models_dir, exist_ok=True)
        self.episode_model_path = self.models_dir / 'episode_dqn_model.pth'
        self.done_model_path = self.models_dir / 'done_dqn_model_{episode}_{timestamp}.pth'

        self.device = BaseTrainer.get_device()
        self.logger.info(f"Using device: {self.device}")

        self.env = ArmEnv.initialize_supervisor(self.config.max_steps)
        self.state_dim = len(self.env.motors) + 3
        self.action_dim = len(self.env.motors) * 2

        self.q_network = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_network = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=20)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = PrioritizedReplay(config.memory_size)
        self.best_reward = float('-inf')

    def train(self) -> None:
        """
        Trains the DQN model.
        """
        self.logger.info(f"Starting training at {datetime.now().isoformat()}")
        for episode in range(self.config.episodes):
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            episode_reward = 0
            episode_loss = 0
            num_steps = 0

            for step in range(self.config.max_steps):
                epsilon = max(self.config.epsilon_end,
                              self.config.epsilon_start * (self.config.epsilon_decay ** episode))
                if random.random() < epsilon:
                    action = random.randint(0, self.action_dim - 1)
                else:
                    with torch.no_grad():
                        q_values = self.q_network(state.unsqueeze(0))
                        action = q_values.argmax().item()

                next_state, reward, done, info = self.env.step(action)
                next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
                episode_reward += reward

                with torch.no_grad():
                    current_q = self.q_network(state.unsqueeze(0)).gather(1, torch.tensor([[action]]).to(self.device))
                    next_q = self.target_network(next_state.unsqueeze(0)).max(1)[0]
                    expected_q = reward + self.config.gamma * next_q * (1 - done)
                    td_error = abs(current_q.item() - expected_q.item())

                self.replay_buffer.add(
                    (state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done),
                    td_error + 1e-6
                )

                if len(self.replay_buffer.buffer) >= self.config.batch_size:
                    states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)
                    states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
                    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
                    rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
                    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
                    dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

                    current_q_values = self.q_network(states).gather(1, actions).squeeze()
                    with torch.no_grad():
                        next_q_values = self.target_network(next_states).max(1)[0]
                        target_q_values = rewards + self.config.gamma * next_q_values * (1 - dones)

                    loss = self.loss_fn(current_q_values, target_q_values)
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.gradient_clip)
                    self.optimizer.step()

                    episode_loss += loss.item()
                    num_steps += 1

                state = next_state

                if done:
                    self.logger.info(f"Episode {episode}/{self.config.episodes} done with reward {episode_reward}")
                    break

            avg_loss = episode_loss / num_steps if num_steps > 0 else 0
            self.scheduler.step(episode_reward)

            if episode % self.config.update_target_every == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

            if episode_reward > self.best_reward:
                self.best_reward = episode_reward

            if episode % 100 == 0 and episode > 0:
                checkpoint_file = self.models_dir / f'checkpoint_dqn_episode_{episode}.pth'
                self.save_checkpoint(self.q_network, episode, episode_reward, checkpoint_file)

            if episode == self.config.episodes - 1:
                final_file = self.models_dir / 'final_dqn_model.pth'
                self.save_checkpoint(self.q_network, episode, episode_reward, final_file)

            self.logger.info(
                f"Episode {episode}/{self.config.episodes} "
                f"({(episode + 1) / self.config.episodes:.1%} complete) | "
                f"Reward: {episode_reward:.2f} | "
                f"Loss: {avg_loss:.4f} | "
                f"Best: {self.best_reward:.2f} | "
                f"Epsilon: {epsilon}"
            )

            self.log_metrics(episode, episode_reward, avg_loss, self.best_reward, extra={"epsilon": epsilon})

        self.logger.info(f"Training completed at {datetime.now().isoformat()}")
        self.logger.info(f"Best reward achieved: {self.best_reward}")

    def evaluate(self) -> None:
        self.logger.info("Evaluation not implemented yet.")
        pass
