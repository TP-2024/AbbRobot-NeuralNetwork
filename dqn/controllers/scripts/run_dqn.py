from pathlib import Path

import torch

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


def run_dqn(supervisor, arm_chain, motors, target_position, pen):
    script_dir = Path(__file__).parent.parent

    models_dir = script_dir / 'models'

    model_path = models_dir / 'best_dqn_model.pth'

    env = ArmEnv(supervisor, arm_chain, motors, target_position, JOINT_LIMITS, pen)

    state_dim = len(motors) + 3
    action_dim = len(motors) * 2
    q_network = QNetwork(state_dim, action_dim)
    # checkpoint = torch.load(str(model_path))
    checkpoint = torch.load(str(model_path), weights_only=True)

    q_network.load_state_dict(checkpoint['model_state_dict'])

    q_network.eval()

    for episode in range(10):
        state = env.reset()
        total_reward = 0

        for t in range(10000):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = q_network(state_tensor).argmax().item()
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break

        print(f"Episode {episode}, Total Reward: {total_reward}")
