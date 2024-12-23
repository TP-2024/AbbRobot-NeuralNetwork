import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dqn.controllers.utils.arm_env import ArmEnv
from dqn.controllers.utils.dqn_model import QNetwork

JOINT_LIMITS = [
    (-3.1415, 3.1415),  # Link A motor
    (-1.5708, 2.61799),  # Link B motor
    (-3.1415, 1.309),  # Link C motor
    (-6.98132, 6.98132),  # Link D motor
    (-2.18166, 2.0944),  # Link E motor
    (-6.98132, 6.98132)  # Link F motor
]


def train_dqn(supervisor, arm_chain, motors, target_position, pen):
    script_dir = Path(__file__).parent.parent

    models_dir = script_dir / 'models'
    os.makedirs(models_dir, exist_ok=True)

    model_path = models_dir / 'best_dqn_model.pth'
    final_model_path = models_dir / 'final_dqn_model.pth'

    # Set up device
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not built with MPS enabled.")
        else:
            print(
                "MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device."
            )
        device = torch.device("cpu")
    else:
        device = torch.device("mps")

    print(f"Using device: {device}")

    # Initialize environment and models
    env = ArmEnv(supervisor, arm_chain, motors, target_position, JOINT_LIMITS, pen)

    state_dim = len(motors) + 3
    action_dim = len(motors) * 2

    # Initialize networks
    q_network = QNetwork(state_dim, action_dim).to(device)
    target_network = QNetwork(state_dim, action_dim).to(device)
    target_network.load_state_dict(q_network.state_dict())

    # Initialize optimizer with gradient clipping
    optimizer = optim.Adam(q_network.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20)

    # Initialize training parameters
    loss_fn = nn.MSELoss()
    epsilon = 1.0

    # Training metrics
    best_reward = float('-inf')
    episodes_without_improvement = 0

    # training configuration
    config = {
        'episodes': 500,
        'max_steps': 200,
        'batch_size': 64,
        'learning_rate': 0.001,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'memory_size': 10000,
        'update_target_every': 10,
        'gradient_clip': 1.0
    }
    # early stopping
    early_stopping_patience = 50
    min_improvement = 0.01

    replay_buffer = PrioritizedReplay(config['memory_size'])

    print(f"Starting training at {datetime.now().isoformat()}")
    for episode in range(config['episodes']):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).to(device)
        episode_reward = 0
        episode_loss = 0
        num_steps = 0

        for step in range(config['max_steps']):
            # Epsilon-greedy action selection
            epsilon = max(config['epsilon_end'], config['epsilon_start'] * (config['epsilon_decay'] ** episode))

            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    q_values = q_network(state.unsqueeze(0))
                    action = q_values.argmax().item()

            # Take action and observe
            next_state, reward, done, info = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
            episode_reward += reward

            # Calculate TD error for prioritized replay
            with torch.no_grad():
                current_q = q_network(state.unsqueeze(0)).gather(1, torch.tensor([[action]]).to(device))
                next_q = target_network(next_state.unsqueeze(0)).max(1)[0]
                expected_q = reward + config['gamma'] * next_q * (1 - done)
                td_error = abs(current_q.item() - expected_q.item())

            # Store transition with priority
            replay_buffer.add((state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done), td_error + 1e-6)

            if len(replay_buffer.buffer) >= config['batch_size']:
                # Sample batch with priorities
                states, actions, rewards, next_states, dones = replay_buffer.sample(config['batch_size'])

                # Convert to tensors
                states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
                actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
                rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
                dones = torch.tensor(dones, dtype=torch.float32).to(device)

                # Compute Q values
                current_q_values = q_network(states).gather(1, actions).squeeze()
                with torch.no_grad():
                    next_q_values = target_network(next_states).max(1)[0]
                    target_q_values = rewards + config['gamma'] * next_q_values * (1 - dones)

                # Compute loss and optimize
                loss = loss_fn(current_q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_network.parameters(), config['gradient_clip'])
                optimizer.step()

                episode_loss += loss.item()
                num_steps += 1

            state = next_state

            if done:
                break

        # Post-episode updates
        avg_loss = episode_loss / num_steps if num_steps > 0 else 0
        scheduler.step(episode_reward)

        # Update target network
        if episode % config['update_target_every'] == 0:
            target_network.load_state_dict(q_network.state_dict())

        # Save best model and check early stopping
        if episode_reward > best_reward + min_improvement:
            best_reward = episode_reward
            torch.save(
                {
                    'episode': episode,
                    'model_state_dict': q_network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'reward': episode_reward,
                    'config': config
                }, str(model_path)
            )
            episodes_without_improvement = 0
        else:
            episodes_without_improvement += 1

        # Early stopping check
        if episodes_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered after {episode} episodes")
            break

        # Logging
        print(
            f"Episode {episode}/{config['episodes']} "
            f"({(episode + 1) / config['episodes']:.1%} complete) | "
            f"Reward: {episode_reward:.2f} | "
            f"Loss: {avg_loss:.4f} | "
            f"Epsilon: {epsilon:.4f} | "
            f"Best: {best_reward:.2f}"
        )

    print(f"Training completed at {datetime.now().isoformat()}")
    print(f"Best reward achieved: {best_reward}")

    torch.save(q_network.state_dict(), str(final_model_path))


class PrioritizedReplay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []

    def add(self, experience, priority):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
            self.priorities.pop(0)
        self.buffer.append(experience)
        self.priorities.append(priority)

    def sample(self, batch_size):
        probs = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        return zip(*samples)
