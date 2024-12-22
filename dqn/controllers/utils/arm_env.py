import numpy as np


class ArmEnv:
    def __init__(self, supervisor, arm_chain, motors, target_position, joint_limits):
        self.supervisor = supervisor
        self.arm_chain = arm_chain
        self.motors = motors
        self.target_position = np.array(target_position, dtype=np.float32)
        self.joint_limits = joint_limits
        self.time_step = int(4 * supervisor.getBasicTimeStep())

    def reset(self):
        self.supervisor.simulationReset()
        self.supervisor.step(self.time_step)
        return self._get_state()

    def step(self, action):
        joint_index = action // 2
        direction = 1 if action % 2 == 1 else -1
        current_position = self.motors[joint_index].getPositionSensor().getValue()
        step_size = 0.05
        new_position = current_position + direction * step_size
        lower_limit, upper_limit = self.joint_limits[joint_index]
        new_position = np.clip(new_position, lower_limit, upper_limit)
        self.motors[joint_index].setPosition(new_position)
        self.supervisor.step(self.time_step)
        state = self._get_state()
        reward = -np.linalg.norm(self.target_position - self._get_end_effector_position())
        done = reward > -0.1
        return state, reward, done, {}

    def _get_state(self):
        return np.array([motor.getPositionSensor().getValue() for motor in self.motors], dtype=np.float32)

    def _get_end_effector_position(self):
        joint_positions = [motor.getPositionSensor().getValue() for motor in self.motors]
        fk_result = self.arm_chain.forward_kinematics([0] + list(joint_positions) + [0])
        return fk_result[:3, 3]
