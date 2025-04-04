import os
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
from torch.distributions import Categorical
import wandb

from dqn.controllers.trainers.base_trainer import BaseTrainer
from dqn.controllers.utils.arm_env import ArmEnv
from dqn.controllers.networks.action_critic_model import ActorCriticNetwork


class ActorCriticTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        if wandb.run is None:
            wandb.login(key=os.environ.get("WANDB_API_KEY", ""))
            wandb.init(project="robot-training", entity="ilya-koyushev1-org", config=config.dict())

        script_dir = Path(__file__).parent.parent
        self.models_dir = script_dir / 'models'
        os.makedirs(self.models_dir, exist_ok=True)
        self.episode_model_path = self.models_dir / 'episode_actor_critic_model.pth'
        self.done_model_path = self.models_dir / 'done_actor_critic_model_{episode}_{timestamp}.pth'

        # Set up device (using MPS if available)
        if not torch.backends.mps.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("mps")
        print(f"Using device: {self.device}")

        # Initialize environment and determine dimensions
        self.env = ArmEnv.initialize_supervisor()
        self.state_dim = len(self.env.motors) + 3  # same state representation as before
        self.action_dim = len(self.env.motors) * 2  # action space based on available motors

        # Initialize the Actor–Critic network and optimizer
        self.ac_net = ActorCriticNetwork(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.ac_net.parameters(), lr=config.learning_rate)
        self.best_reward = float('-inf')
        self.gamma = config.gamma
        self.value_loss_coef = config.value_loss_coef

        # Save the config object for later use
        self.config = config

    def train(self):
        print(f"Starting Actor–Critic training at {datetime.now().isoformat()}")
        for episode in range(self.config.episodes):
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            log_probs = []
            values = []
            rewards = []
            episode_reward = 0

            for step in range(self.config.max_steps):
                # Get action probabilities and value estimate from the network
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
                    print(f'\nEpisode {episode}/{self.config.episodes} Done! Reward: {episode_reward}\n')
                    break

            # Compute discounted returns for the episode
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

            # Convert lists of tensors into single tensors for loss computation
            values = torch.stack(values)
            log_probs = torch.stack(log_probs)
            # Compute advantage: how much better the return is than the estimated value
            advantages = returns - values

            # Actor loss: weighted by advantage (detach the advantage so gradients don't flow into the critic)
            actor_loss = - (log_probs * advantages.detach()).sum()
            # Critic loss: mean squared error of the value estimates
            critic_loss = advantages.pow(2).sum()
            # Combined loss
            loss = actor_loss + self.value_loss_coef * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if episode_reward > self.best_reward:
                self.best_reward = episode_reward

            if episode % 100 == 0 and episode > 0:
                torch.save({
                    'episode': episode,
                    'model_state_dict': self.ac_net.state_dict(),
                    'reward': episode_reward,
                    'config': self.config.dict()
                }, self.models_dir / f'checkpoint_actor_critic_episode_{episode}.pth')

            # At the end of training, save final model
            if episode == self.config.episodes - 1:
                torch.save({
                    'episode': episode,
                    'model_state_dict': self.ac_net.state_dict(),
                    'reward': episode_reward,
                    'config': self.config.dict()
                }, self.models_dir / 'final_ac_model.pth')

            print(
                f"Episode {episode}/{self.config.episodes} | Reward: {episode_reward:.2f} | Loss: {loss.item():.4f} | Best: {self.best_reward:.2f}")
            wandb.log({
                "episode": episode,
                "reward": episode_reward,
                "loss": loss.item(),
                "best_reward": self.best_reward
            })

        print(f"Training completed at {datetime.now().isoformat()}")
        print(f"Best reward achieved: {self.best_reward}")

    def evaluate(self):
        # TODO: Implement evaluation logic here
        pass
