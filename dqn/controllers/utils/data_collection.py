import os

import numpy as np


def collect_data(arm_chain, motors, arm, samples=1000):
    IKPY_MAX_ITERATIONS = 4
    inputs = []
    outputs = []

    os.makedirs("data", exist_ok=True)

    def is_within_workspace(x, y, z):
        max_reach = sum(link.length for link in arm_chain.links if link.name != "base")
        return np.sqrt(x ** 2 + y ** 2 + z ** 2) <= max_reach

    for _ in range(samples):
        # Generate random target positions
        target_point = [np.random.uniform(-6.0, -5.0),
            np.random.uniform(-2.2, -1.0),
            np.random.uniform(1.5, 3.2)]

        arm_position = arm.getPosition()
        x = -(target_point[1] - arm_position[1])
        y = target_point[0] - arm_position[0]
        z = target_point[2] - arm_position[2]

        if not is_within_workspace(x, y, z):
            print(f"({x}, {y}, {z}) not in workspace")
            continue

        initial_position = [0] + [m.getPositionSensor().getValue() for m in motors] + [0]

        try:
            ik_results = arm_chain.inverse_kinematics(
                [x, y, z],
                max_iter=IKPY_MAX_ITERATIONS,
                initial_position=initial_position
            )
            inputs.append([x, y, z])
            outputs.append(ik_results[1:-1])
        except ValueError:
            print(f"Invalid IK for target {x, y, z}. Skipping.")

    print(f"Number of collected samples: {len(inputs)}")
    np.savez("data/ik_training_data.npz", inputs=np.array(inputs), outputs=np.array(outputs))
    print("Data collection complete. Data saved to 'data/ik_training_data.npz'.")
