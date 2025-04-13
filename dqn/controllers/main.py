import argparse
import logging
from typing import Any

import yaml

from dqn.controllers.configs.config import DQNConfig, PGConfig, ActorCriticConfig
from dqn.controllers.runners.run_dqn import run_dqn
from dqn.controllers.trainers.train_action_critic import ActorCriticTrainer
from dqn.controllers.trainers.train_dqn import DQNTrainer
from dqn.controllers.trainers.train_pg import PGTrainer


def parse_args() -> Any:
    parser = argparse.ArgumentParser(description='Robot Arm Controller')
    parser.add_argument('--mode', default='train', choices=['train', 'run'], help='Mode of operation: train or run')
    parser.add_argument('--method', type=str, default='dqn', choices=['dqn', 'pg', 'ac'], help='Training method to use')
    parser.add_argument('--config', type=str, default='configs/dqn_config.yaml', help="Configuration YAML file"
    )
    return parser.parse_args()

def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args = parse_args()

    with open(args.config, 'r') as f:
        config_data = yaml.safe_load(f)

    if args.mode == "train":
        if args.method == "dqn":
            config = DQNConfig(**config_data)
            trainer = DQNTrainer(config)
        elif args.method == "pg":
            config = PGConfig(**config_data)
            trainer = PGTrainer(config)
        elif args.method == "ac":
            config = ActorCriticConfig(**config_data)
            trainer = ActorCriticTrainer(config)
        else:
            raise ValueError("Unknown training method!")
        trainer.train()
    elif args.mode == "run":
        if args.method == "dqn":
            run_dqn()
        else:
            logging.info("Run mode not implemented yet.")
    else:
        logging.error("Invalid mode!")

if __name__ == "__main__":
    main()
