import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import wandb

from dqn.controllers.utils.arm_env import ArmEnv
from dqn.controllers.trainers.base_trainer import BaseTrainer

# Define a simple policy network (REINFORCE)
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.layers(x)


class PGTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        if wandb.run is None:
            wandb.login(key=os.environ.get("WANDB_API_KEY", ""))
            wandb.init(project="robot-training", entity="ilya-koyushev1-org", config=config.dict())

        script_dir = Path(__file__).parent.parent
        self.models_dir = script_dir / 'models'
        os.makedirs(self.models_dir, exist_ok=True)
        self.episode_model_path = self.models_dir / 'episode_pg_model.pth'
        self.done_model_path = self.models_dir / 'done_pg_model_{episode}_{timestamp}.pth'

        # Set up device (same logic as in DQNTrainer)
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print(
                    "MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device.")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("mps")
        print(f"Using device: {self.device}")

        # Initialize environment and dimensions
        self.env = ArmEnv.initialize_supervisor()
        self.state_dim = len(self.env.motors) + 3
        self.action_dim = len(self.env.motors) * 2

        # Initialize the policy network and optimizer
        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=config['learning_rate'])
        self.best_reward = float('-inf')

    def train(self):
        print(f"Starting Policy Gradient training at {datetime.now().isoformat()}")
        for episode in range(self.config['episodes']):
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            log_probs = []
            rewards = []
            episode_reward = 0

            for step in range(self.config['max_steps']):
                # Get action probabilities and sample an action
                probs = self.policy_net(state.unsqueeze(0))
                m = Categorical(probs)
                action = m.sample()
                log_prob = m.log_prob(action)
                log_probs.append(log_prob)

                next_state, reward, done, info = self.env.step(action.item())
                next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
                rewards.append(reward)
                episode_reward += reward

                state = next_state

                if done:
                    print(f'\nEpisode {episode}/{self.config["episodes"]} Done! Reward: {episode_reward}\n')
                    break

            # Compute discounted returns (REINFORCE)
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + self.config['gamma'] * G
                returns.insert(0, G)
            returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

            loss = 0
            for log_prob, G in zip(log_probs, returns):
                loss -= log_prob * G

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward

            if episode % 100 == 0 and episode > 0:
                # Save a checkpoint every 100 episodes
                torch.save({
                    'episode': episode,
                    'model_state_dict': self.policy_net.state_dict(),
                    'reward': episode_reward,
                    'config': self.config.dict()
                }, self.models_dir / f'checkpoint_pg_episode_{episode}.pth')

            if episode == self.config.episodes - 1:
                torch.save({
                    'episode': episode,
                    'model_state_dict': self.policy_net.state_dict(),
                    'reward': episode_reward,
                    'config': self.config.dict()
                }, self.models_dir / 'final_pg_model.pth')

            # Print status message
            print(
                f"Episode {episode}/{self.config['episodes']} "
                f"({(episode + 1) / self.config['episodes']:.1%} complete) | "
                f"Reward: {episode_reward:.2f} | "
                f"Loss: {loss.item():.4f} | "
                f"Best: {self.best_reward:.2f}"
            )
            wandb.log({"episode": episode, "reward": episode_reward, "loss": loss.item(), "best_reward": self.best_reward})

        print(f"Training completed at {datetime.now().isoformat()}")
        print(f"Best reward achieved: {self.best_reward}")

    def evaluate(self):
        # TODO: Implement evaluation logic here
        pass
