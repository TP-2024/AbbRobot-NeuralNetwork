import argparse
from datetime import datetime

import torch

from scripts.run_dqn import run_dqn
from scripts.train_dqn import train_dqn


def parse_args():
    parser = argparse.ArgumentParser(description='Robot Arm Controller')
    parser.add_argument(
        'mode', type=str, choices=['train', 'run'],
        help='Mode of operation: train or run'
    )
    return parser.parse_args()


def main():
    task = parse_args()

    print(f'MPS available: {torch.backends.mps.is_available()}')

    if task.mode == "train":
        train_dqn()
    elif task.mode == "run":
        run_dqn()
    else:
        print("Invalid task!")


if __name__ == "__main__":
    main()
