import os
import random
import time
from collections import deque
from datetime import datetime, timedelta
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


def train_dqn(supervisor, arm_chain, motors, target_position):
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
    env = ArmEnv(supervisor, arm_chain, motors, target_position, JOINT_LIMITS)

    state_dim = len(motors)
    action_dim = len(motors) * 2

    # Initialize networks
    q_network = QNetwork(state_dim, action_dim).to(device)
    target_network = QNetwork(state_dim, action_dim).to(device)
    target_network.load_state_dict(q_network.state_dict())

    # Initialize optimizer with gradient clipping
    optimizer = optim.Adam(q_network.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20)

    # Initialize training parameters
    replay_buffer = deque(maxlen=10000)
    loss_fn = nn.MSELoss()
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.01
    gamma = 0.99
    batch_size = 64
    update_target_every = 10

    # Training metrics
    best_reward = float('-inf')
    episodes_without_improvement = 0

    start_time = time.time()
    total_epochs = 500

    print(f"Starting training at {datetime.now().isoformat()}")
    for episode in range(500):
        # Calculate progress and time estimates
        elapsed_time = time.time() - start_time
        progress = (episode + 1) / total_epochs
        estimated_total_time = elapsed_time / progress
        remaining_time = estimated_total_time - elapsed_time

        # Convert to hours:minutes:seconds format
        elapsed = str(timedelta(seconds=int(elapsed_time)))
        remaining = str(timedelta(seconds=int(remaining_time)))

        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).to(device)
        total_reward = 0
        episode_loss = 0
        num_steps = 0

        for t in range(200):
            # Action selection
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    state_tensor = state.unsqueeze(0)
                    q_values = q_network(state_tensor)
                    action = q_values.argmax().item()

            # Environment step
            next_state, reward, done, _ = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32).to(device)

            # Store transition
            replay_buffer.append((state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done))
            total_reward += reward

            # Training step
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                # Convert to tensors and move to device
                states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
                actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
                rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
                dones = torch.tensor(dones, dtype=torch.float32).to(device)

                # Compute current Q values
                current_q_values = q_network(states).gather(1, actions).squeeze()

                # Compute next Q values
                with torch.no_grad():
                    next_q_values = target_network(next_states).max(1)[0]
                    target_q_values = rewards + gamma * next_q_values * (1 - dones)

                # Compute loss and optimize
                loss = loss_fn(current_q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=1.0)
                optimizer.step()

                episode_loss += loss.item()
                num_steps += 1

            state = next_state

            if done:
                break

        # Post-episode updates
        avg_loss = episode_loss / num_steps if num_steps > 0 else 0
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # Update target network
        if episode % update_target_every == 0:
            target_network.load_state_dict(q_network.state_dict())

        # Learning rate scheduling
        scheduler.step(total_reward)
        current_lr = optimizer.param_groups[0]['lr']

        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(
                {
                    'episode': episode,
                    'model_state_dict': q_network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'reward': total_reward,
                },
                str(model_path)
            )
            episodes_without_improvement = 0
        else:
            episodes_without_improvement += 1

        print(
            f"Episode {episode}/{total_epochs} "
            f"({progress:.1%} complete) | "
            f"Elapsed: {elapsed} | "
            f"Remaining: {remaining} | "
            f"Reward: {total_reward:.2f}"
        )

        print(
            f"Episode {episode}: Reward = {total_reward:.2f}, Loss = {avg_loss:.4f}, "
            f"Epsilon = {epsilon:.4f}, Learning Rate = {current_lr:.6f}, best_reward = {best_reward:.2f}\n"
        )

    print(f"Training completed at {datetime.now().isoformat()}")
    print(f"Best reward achieved: {best_reward}")

    torch.save(q_network.state_dict(), str(final_model_path))

# import random
# from collections import deque
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
#
# from dqn.controllers.utils.arm_env import ArmEnv
# from dqn.controllers.utils.dqn_model import QNetwork
#
# JOINT_LIMITS = [
#     (-3.1415, 3.1415),  # Link A motor
#     (-1.5708, 2.61799),  # Link B motor
#     (-3.1415, 1.309),  # Link C motor
#     (-6.98132, 6.98132),  # Link D motor
#     (-2.18166, 2.0944),  # Link E motor
#     (-6.98132, 6.98132)  # Link F motor
# ]
#
#
# def train_dqn(supervisor, arm_chain, motors, target_position):
#     env = ArmEnv(supervisor, arm_chain, motors, target_position, JOINT_LIMITS)
#
#     state_dim = len(motors)
#     action_dim = len(motors) * 2
#     q_network = QNetwork(state_dim, action_dim)
#     target_network = QNetwork(state_dim, action_dim)
#     target_network.load_state_dict(q_network.state_dict())
#     optimizer = optim.Adam(q_network.parameters(), lr=0.001)
#
#     replay_buffer = deque(maxlen=10000)
#     loss_fn = nn.MSELoss()
#     epsilon, epsilon_decay, min_epsilon, gamma, batch_size = 1.0, 0.995, 0.01, 0.99, 64
#
#     for episode in range(500):
#         state = env.reset()
#         total_reward = 0
#
#         for t in range(200):
#             if random.random() < epsilon:
#                 action = random.randint(0, action_dim - 1)
#             else:
#                 state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
#                 action = q_network(state_tensor).argmax().item()
#
#             next_state, reward, done, _ = env.step(action)
#             replay_buffer.append((state, action, reward, next_state, done))
#             total_reward += reward
#             state = next_state
#
#             if len(replay_buffer) >= batch_size:
#                 minibatch = random.sample(replay_buffer, batch_size)
#                 states, actions, rewards, next_states, dones = zip(*minibatch)
#                 states_tensor = torch.tensor(states, dtype=torch.float32)
#                 actions_tensor = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
#                 rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
#                 next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
#                 dones_tensor = torch.tensor(dones, dtype=torch.float32)
#                 q_values = q_network(states_tensor).gather(1, actions_tensor).squeeze()
#                 with torch.no_grad():
#                     max_next_q_values = target_network(next_states_tensor).max(1)[0]
#                 target_q_values = rewards_tensor + gamma * max_next_q_values * (1 - dones_tensor)
#                 loss = loss_fn(q_values, target_q_values)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#
#             if done:
#                 break
#
#         epsilon = max(min_epsilon, epsilon * epsilon_decay)
#         if episode % 10 == 0:
#             target_network.load_state_dict(q_network.state_dict())
#         print(f"Episode {episode}, Total Reward: {total_reward}")
#
#     torch.save(q_network.state_dict(), "models/dqn_model.pth")
#     print("Model training complete. Model saved to models/dqn_model.pth.")
