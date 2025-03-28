import numpy as np
import torch


def move_to_point_with_nn(model, global_point, supervisor, motors, arm_chain, threshold=0.005):
    arm = supervisor.getSelf()
    timeStep = int(4 * supervisor.getBasicTimeStep())

    while supervisor.step(timeStep) != -1:
        armPosition = arm.getPosition()
        x = -(global_point[1] - armPosition[1])
        y = global_point[0] - armPosition[0]
        z = global_point[2] - armPosition[2]

        target_position = torch.tensor([[x, y, z]], dtype=torch.float32)
        ik_results = model(target_position).detach().numpy()[0]

        for i, motor in enumerate(motors):
            motor.setPosition(ik_results[i])

        position = arm_chain.forward_kinematics([0] + list(ik_results) + [0])
        squared_distance = (
                (position[0, 3] - x) ** 2 +
                (position[1, 3] - y) ** 2 +
                (position[2, 3] - z) ** 2
        )
        if np.sqrt(squared_distance) < threshold:
            break
