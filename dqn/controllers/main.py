import tempfile
from datetime import datetime

import torch
from ikpy.chain import Chain

from controller import Supervisor
from scripts.run_dqn import run_dqn
from scripts.train_dqn import train_dqn


def initialize_supervisor():
    supervisor = Supervisor()
    timeStep = int(4 * supervisor.getBasicTimeStep())

    with tempfile.NamedTemporaryFile(suffix='.urdf', delete=False) as file:
        filename = file.name
        file.write(supervisor.getUrdf().encode('utf-8'))
    arm_chain = Chain.from_urdf_file(filename, active_links_mask=[False, True, True, True, True, True, True, False])

    motors = []
    for link in arm_chain.links:
        if 'motor' in link.name:
            motor = supervisor.getDevice(link.name)
            # Different velocities for different joints
            if 'A' in link.name or 'B' in link.name:  # Base and shoulder joints
                motor.setVelocity(5.0)  # Slower for larger movements
            elif 'C' in link.name or 'D' in link.name:  # Elbow and first wrist
                motor.setVelocity(7.0)  # Medium speed
            else:  # E and F joints (wrist joints)
                motor.setVelocity(9.0)  # Faster for smaller movements

            position_sensor = motor.getPositionSensor()
            position_sensor.enable(timeStep)
            motors.append(motor)

    pen = supervisor.getFromDef('PEN')
    target_position = supervisor.getFromDef('START').getPosition()

    return supervisor, arm_chain, motors, target_position, pen


def main():
    supervisor, arm_chain, motors, target_position, pen = initialize_supervisor()
    task = input("Enter task (train/run): ").strip()

    print(f'Starting at {datetime.now().isoformat()}')
    print(f'PyTorch version: {torch.__version__}')
    print(f'MPS available: {torch.backends.mps.is_available()}')

    if task == "train":
        train_dqn(supervisor, arm_chain, motors, target_position, pen)
    elif task == "run":
        run_dqn(supervisor, arm_chain, motors, target_position, pen)
    else:
        print("Invalid task!")


if __name__ == "__main__":
    main()
