import argparse
from dqn.controllers.trainers.train_dqn import DQNTrainer
from dqn.controllers.trainers.train_pg import PGTrainer
from pydantic import BaseModel


class DQNConfig(BaseModel):
    episodes: int = 500
    max_steps: int = 200
    batch_size: int = 64
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    memory_size: int = 10000
    update_target_every: int = 10
    gradient_clip: float = 1.0
    min_improvement: float = 0.01


class PGConfig(BaseModel):
    episodes: int = 500
    max_steps: int = 200
    learning_rate: float = 0.001
    gamma: float = 0.99
    min_improvement: float = 0.01


def main():
    args = parse_args()

    config_dqn = DQNConfig()
    config_pg = PGConfig()

    if args.mode == "train":
        if args.method == "dqn":
            trainer = DQNTrainer(config_dqn)
        elif args.method == "pg":
            trainer = PGTrainer(config_pg)
        else:
            raise ValueError("Unknown training method!")
        trainer.train()
    elif args.mode == "run":
        print("Run mode not implemented yet.")
    else:
        print("Invalid mode!")


def parse_args():
    parser = argparse.ArgumentParser(description='Robot Arm Controller')
    parser.add_argument('--mode', choices=['train', 'run'], help='Mode of operation: train or run')
    parser.add_argument('--method', type=str, default='dqn', choices=['dqn', 'pg'], help='Training method to use')
    return parser.parse_args()

def main():
    args = parse_args()

    config_dqn = DQNConfig()
    config_pg = PGConfig()

    if args.mode == "train":
        if args.method == "dqn":
            trainer = DQNTrainer(config_dqn)
        elif args.method == "pg":
            trainer = PGTrainer(config_pg)
        else:
            raise ValueError("Unknown training method!")
        trainer.train()
    elif args.mode == "run":
        # TODO: Implement evaluation or run logic as needed
        print("Run mode not implemented yet.")
    else:
        print("Invalid mode!")

if __name__ == "__main__":
    main()
