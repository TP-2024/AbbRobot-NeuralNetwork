from pathlib import Path

import torch

from dqn.controllers.utils.arm_env import ArmEnv
from dqn.controllers.networks.dqn_model import QNetwork


def run_dqn():
    script_dir = Path(__file__).parent.parent
    models_dir = script_dir / 'models'
    # model_path = models_dir / 'episode_dqn_model.pth'
    # model_path = models_dir / 'done_dqn_model_25-175432.pth'

    # Get all .pth files in the models directory
    model_files = sorted(models_dir.glob('*300*.pth'))

    env = ArmEnv.initialize_supervisor(200)

    state_dim = len(env.motors) + 3
    action_dim = len(env.motors) * 2

    all_models_performance = {}
    for model_path in model_files:
        print(f"\nEvaluating model: {model_path.name}")

        q_network = QNetwork(state_dim, action_dim)
        checkpoint = torch.load(str(model_path), weights_only=False)
        config = checkpoint.get('config', {})
        q_network.load_state_dict(checkpoint['model_state_dict'])
        q_network.eval()

        state = env.reset()
        total_reward = 0

        for t in range(config['max_steps']):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = q_network(state_tensor).argmax().item()
            state, reward, done, _ = env.step(action)
            total_reward += reward

            if done:
                break

        # Print summary statistics for this model
        print(f"\nModel: {model_path.name}")
        print(f"Reward: {total_reward:.2f}")
        print(f"Done: {str(done)}")
        print("-" * 50)

        all_models_performance[model_path.name] = {
            'reward': total_reward,
            'training_reward': checkpoint.get('reward', 'N/A'),
            'done': done
        }

    # Print comparative analysis
    print("\nComparative Analysis of All Models:")
    print("-" * 80)
    print(f"{'Model Name':<35} {'Reward':<15} {'Training reward':<20} {'Touched':<15}")
    print("-" * 80)

    for model_name, stats in sorted(all_models_performance.items(), key=lambda x: x[1]['reward'], reverse=True):
        print(
            f"{model_name:<35} {stats['reward']:<15.2f} {stats['training_reward']:<20.2f} {stats['done']:}"
        )