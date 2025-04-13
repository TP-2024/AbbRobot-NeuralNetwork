import os
import tempfile
from typing import Any

import numpy as np
from ikpy.chain import Chain
from numpy import ndarray, floating

from controller import Supervisor
import logging


class ArmEnv(Supervisor):
    """
    Environment for the robotic arm simulation.
    """
    JOINT_LIMITS = [
        (-3.1415, 3.1415),      # Link A motor
        (-1.5708, 2.61799),     # Link B motor
        (-3.1415, 1.309),       # Link C motor
        (-6.98132, 6.98132),    # Link D motor
        (-2.18166, 2.0944),     # Link E motor
        (-6.98132, 6.98132)     # Link F motor
    ]

    def __init__(self, max_steps: int) -> None:
        super().__init__()
        self.time_step = int(4 * self.getBasicTimeStep())
        self.arm_chain = self._initialize_chain()
        self.motors = self._initialize_motors()
        self.target_position = self.getFromDef('START').getPosition()
        self.pen = self.getFromDef('PEN')

        self.start_dist = 0.0
        self.acc_dist = 0.0
        self.prev_dist = 0.0
        self.prev_axis_dists = None
        self.lambda_scale = 1.0
        self.max_steps = max_steps
        self.current_step = 0

        self.touch_sensors = []
        for sensor_name in [
            "A touch sensor", "B touch sensor", "C touch sensor",
            "D touch sensor", "E touch sensor", "F touch sensor"
        ]:
            sensor = self.getDevice(sensor_name)
            sensor.enable(self.time_step)
            self.touch_sensors.append(sensor)

        super().step(self.time_step)

    def _initialize_chain(self) -> Chain:
        """
        Initializes the robot's kinematic chain from its URDF file.
        """
        with tempfile.NamedTemporaryFile(suffix='.urdf', delete=False) as file:
            filename = file.name
            file.write(self.getUrdf().encode('utf-8'))
        active_links_mask = [False, True, True, True, True, True, True, False]
        chain = Chain.from_urdf_file(filename, active_links_mask=active_links_mask)
        os.remove(filename)
        return chain

    def _initialize_motors(self) -> list:
        """
        Initializes motors from the arm chain.
        """
        motors = []
        for link in self.arm_chain.links:
            if 'motor' in link.name:
                motor = self.getDevice(link.name)
                if 'A' in link.name or 'B' in link.name:
                    motor.setVelocity(5.0)
                elif 'C' in link.name or 'D' in link.name:
                    motor.setVelocity(7.0)
                else:
                    motor.setVelocity(9.0)

                position_sensor = motor.getPositionSensor()
                position_sensor.enable(self.time_step)
                motors.append(motor)
        return motors

    @classmethod
    def initialize_supervisor(cls, max_steps: int) -> 'ArmEnv':
        """
        Factory method to initialize and return an ArmEnv instance.
        """
        return cls(max_steps)

    def reset(self) -> np.ndarray:
        """
        Resets the simulation and returns the initial state.
        """
        self.simulationResetPhysics()
        self.simulationReset()
        super().step(self.time_step)
        starting_positions = [0.0, -0.5, 0.5, 0.0, 0.0, 0.0]

        for i, motor in enumerate(self.motors):
            pos = np.clip(starting_positions[i], self.JOINT_LIMITS[i][0], self.JOINT_LIMITS[i][1])
            motor.setPosition(pos)

        super().step(self.time_step * 10)  # Wait for the robot to settle.
        self.acc_dist = 0.0
        current_pen = self.get_pen_position()
        self.start_dist = np.linalg.norm(np.array(self.target_position) - current_pen)
        self.prev_dist = self.start_dist
        self.prev_axis_dists = np.abs(np.array(self.target_position) - current_pen)
        self.current_step = 0

        return self._get_state()

    def step(self, action: int = None) -> tuple[
        ndarray, floating[Any], bool, dict[str, int | floating[Any] | bool | float]
    ]:
        """
        Applies an action and returns the new state, reward, done flag, and additional info.
        """
        prev_pen_pos = self.get_pen_position()

        joint_index = action // 2
        direction = 1 if action % 2 == 1 else -1
        current_joint_position = self.motors[joint_index].getPositionSensor().getValue()

        distance_to_target = np.linalg.norm(np.array(self.target_position) - prev_pen_pos)
        base_step = 0.05
        if distance_to_target > 1.0:
            step_size = base_step * 1.5
        elif distance_to_target < 0.2:
            step_size = base_step * 0.5
        else:
            step_size = base_step

        new_joint_position = current_joint_position + direction * step_size
        lower_limit, upper_limit = self.JOINT_LIMITS[joint_index]
        new_position = np.clip(new_joint_position, lower_limit, upper_limit)
        self.motors[joint_index].setPosition(new_position)
        super().step(self.time_step)

        self.current_step += 1
        current_pos = self.get_pen_position()
        current_dist = np.linalg.norm(np.array(self.target_position) - current_pos)
        step_dist = np.linalg.norm(current_pos - prev_pen_pos)
        self.acc_dist += step_dist

        current_axis_dists = np.abs(np.array(self.target_position) - current_pos)
        reward_components = []
        for i in range(3):
            delta = current_axis_dists[i] - self.prev_axis_dists[i]
            if delta < 0:
                r = 1.1 * np.exp(-self.lambda_scale * (current_axis_dists[i] ** 2))
            else:
                r = -np.exp(-self.lambda_scale * (current_axis_dists[i] ** 2))
            reward_components.append(r)
        reward = np.mean(reward_components)

        success = False
        if current_dist < 0.01:
            reward += 100.0
            success = True

        if self._is_collision():
            reward -= 5.0  # Must not be bigger than ~20 so not to overwhelm main reward

        if self.current_step >= self.max_steps and not success:
            reward -= 15.0  # Terminal penalty for not reaching the goal

        self.prev_axis_dists = current_axis_dists
        self.prev_dist = current_dist

        info = {
            'distance': current_dist,
            'accumulated_distance': self.acc_dist,
            'step': self.current_step,
            'goal_reached': success
        }

        return self._get_state(), reward, success, info

    def _get_state(self) -> np.ndarray:
        """
        Returns the current state as a normalized numpy array.
        """
        joint_positions = np.array([motor.getPositionSensor().getValue() for motor in self.motors])
        normalized_positions = []
        for i, pos in enumerate(joint_positions):
            low, high = self.JOINT_LIMITS[i]
            norm_pos = 2.0 * (pos - low) / (high - low) - 1.0
            normalized_positions.append(norm_pos)
        pen_pos = self.get_pen_position()
        target_vector = np.array(self.target_position) - pen_pos
        norm_target = target_vector / (np.linalg.norm(target_vector) + 1e-8)
        state = np.concatenate([normalized_positions, norm_target])
        return state.astype(np.float32)

    def get_pen_position(self) -> np.ndarray:
        """
        Returns the pen tip position.
        """
        pen_pos = self.pen.getPosition()
        pen_tip_pos = np.array([pen_pos[0], pen_pos[1], pen_pos[2] - 0.09], dtype=np.float32)
        return pen_tip_pos

    def _is_collision(self) -> bool:
        """
        Checks for collisions using joint limits and touch sensors.
        """
        for i, motor in enumerate(self.motors):
            pos = motor.getPositionSensor().getValue()
            lower, upper = self.JOINT_LIMITS[i]
            if abs(pos - lower) < 0.01 or abs(pos - upper) < 0.01:
                return True

        for sensor in self.touch_sensors:
            if sensor.getValue() == 1.0:
                logging.info(f"Collision detected for sensor {sensor.getName()}")
                return True

        return False
