import tempfile

import numpy as np
from ikpy.chain import Chain

from controller import Supervisor


class ArmEnv:
    JOINT_LIMITS = [
        (-3.1415, 3.1415),  # Link A motor
        (-1.5708, 2.61799),  # Link B motor
        (-3.1415, 1.309),  # Link C motor
        (-6.98132, 6.98132),  # Link D motor
        (-2.18166, 2.0944),  # Link E motor
        (-6.98132, 6.98132)  # Link F motor
    ]

    def __init__(self, supervisor, arm_chain, motors, target_position, pen):
        self.supervisor = supervisor
        self.arm_chain = arm_chain
        self.motors = motors
        self.target_position = np.array(target_position, dtype=np.float32)
        self.pen = pen
        self.time_step = int(4 * supervisor.getBasicTimeStep())

        self.start_dist = None
        self.acc_dist = 0.0
        self.prev_dist = None

    @classmethod
    def initialize_supervisor(cls) -> 'ArmEnv':
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

        return ArmEnv(supervisor, arm_chain, motors, target_position, pen)

    def reset(self):
        self.supervisor.simulationResetPhysics()
        self.supervisor.simulationReset()
        self.supervisor.step(self.time_step)

        # Initialize joints to a good starting position
        starting_positions = [
            0.0,  # Base joint
            -0.5,  # Shoulder
            0.5,  # Elbow
            0.0,  # Wrist 1
            0.0,  # Wrist 2
            0.0  # Wrist 3
        ]

        for i, motor in enumerate(self.motors):
            pos = np.clip(
                starting_positions[i],
                self.JOINT_LIMITS[i][0],
                self.JOINT_LIMITS[i][1]
            )
            motor.setPosition(pos)

        self.supervisor.step(self.time_step * 10)  # Wait for robot to settle

        self.acc_dist = 0.0
        self.start_dist = np.linalg.norm(self.target_position - self._get_pen_position())
        self.prev_dist = self.start_dist

        # # Reset motor positions within joint limits
        # for joint_index, motor in enumerate(self.motors):
        #     initial_position = (self.JOINT_LIMITS[joint_index][0] + self.JOINT_LIMITS[joint_index][1]) / 2
        #     motor.setPosition(initial_position)
        #
        # self.supervisor.step(self.time_step)

        return self._get_state()

    def step(self, action):
        # Store previous position
        prev_pos = self._get_pen_position()

        joint_index = action // 2
        direction = 1 if action % 2 == 1 else -1
        current_position = self.motors[joint_index].getPositionSensor().getValue()

        # Calculate distance to target
        current_pos = self._get_pen_position()
        distance_to_target = np.linalg.norm(self.target_position - current_pos)

        # Adjust step size based on distance
        base_step = 0.05
        if distance_to_target > 1.0:
            step_size = base_step * 1.5  # Larger steps when far
        elif distance_to_target < 0.2:
            step_size = base_step * 0.5  # Smaller steps when close
        else:
            step_size = base_step

        new_position = current_position + direction * step_size
        lower_limit, upper_limit = self.JOINT_LIMITS[joint_index]
        new_position = np.clip(new_position, lower_limit, upper_limit)
        self.motors[joint_index].setPosition(new_position)
        self.supervisor.step(self.time_step)

        # Get new position and calculate distances
        current_pos = self._get_pen_position()
        current_dist = np.linalg.norm(self.target_position - current_pos)

        # Calculate movement distance
        step_dist = np.linalg.norm(current_pos - prev_pos)
        self.acc_dist += step_dist

        # Calculate reward components
        distance_improvement = (self.prev_dist - current_dist)  # Positive when getting closer
        movement_efficiency = self.start_dist / (self.acc_dist + self.start_dist)  # 1 to 0 (1 is better)
        target_proximity = 1.0 - (current_dist / self.start_dist)  # 0 to 1 (1 is better)

        # Add proximity bonus for very close positions
        proximity_bonus = 0
        if current_dist < 0.15:  # Within 15cm
            proximity_bonus = (0.15 - current_dist) * 10  # More bonus as it gets closer

        # Combine reward components
        reward = (
                0.4 * distance_improvement +  # Reward for moving closer to target
                0.3 * movement_efficiency +  # Reward for efficient movement
                0.3 * target_proximity +  # Reward for being close to target
                proximity_bonus  # Bonus for very close positions
        )

        # Penalties
        if self._is_collision():  # Implement collision detection
            reward -= 1.0

        # Update previous distance
        self.prev_dist = current_dist

        # Check if done
        done = current_dist < 0.1

        info = {
            'distance': current_dist,
            'accumulated_distance': self.acc_dist,
            'efficiency': movement_efficiency
        }

        return self._get_state(), reward, done, info

    def _get_state(self):
        # return np.array([motor.getPositionSensor().getValue() for motor in self.motors], dtype=np.float32)
        # Get joint positions
        joint_positions = np.array([motor.getPositionSensor().getValue() for motor in self.motors])

        # Normalize joint positions to [-1, 1]
        normalized_positions = []
        for i, pos in enumerate(joint_positions):
            low, high = self.JOINT_LIMITS[i]
            norm_pos = 2.0 * (pos - low) / (high - low) - 1.0
            normalized_positions.append(norm_pos)

        # Add distance to target and orientation
        pen_pos = self._get_pen_position()
        target_vector = self.target_position - pen_pos
        normalized_target = target_vector / np.linalg.norm(target_vector)

        state = np.concatenate([normalized_positions, normalized_target])

        return state.astype(np.float32)

    def _get_end_effector_position(self):
        joint_positions = [motor.getPositionSensor().getValue() for motor in self.motors]
        fk_result = self.arm_chain.forward_kinematics([0] + list(joint_positions) + [0])
        return fk_result[:3, 3]

    def _get_pen_position(self):
        """Get the actual position of the pen tip"""
        # Get pen position and add offset for the tip
        pen_pos = self.pen.getPosition()
        # Adjust Z coordinate for pen tip (0.09 = total pen length)
        pen_tip_pos = np.array([pen_pos[0], pen_pos[1], pen_pos[2] - 0.09], dtype=np.float32)
        return pen_tip_pos

    def _is_collision(self):
        # Check if any joint is at its limit
        for i, motor in enumerate(self.motors):
            pos = motor.getPositionSensor().getValue()
            lower, upper = self.JOINT_LIMITS[i]
            if abs(pos - lower) < 0.01 or abs(pos - upper) < 0.01:
                return True

        # You could also add environment collision checks here
        return False

    @staticmethod
    def _calculate_joint_cost(action):
        # Penalize excessive joint movement
        return -0.01 * np.abs(action)

    def _calculate_smoothness_reward(self, current_pos, prev_pos, prev_prev_pos):
        # Reward smooth trajectories
        acceleration = (current_pos - 2 * prev_pos + prev_prev_pos)
        return -0.1 * np.linalg.norm(acceleration)

    def _is_in_workspace(self, position):
        # Check if position is within valid workspace
        workspace_radius = 1.0
        return np.linalg.norm(position) <= workspace_radius

    def _calculate_orientation_reward(self):
        # Reward for correct end-effector orientation
        current_orientation = self._get_end_effector_orientation()
        target_orientation = self._get_target_orientation()
        orientation_diff = np.abs(current_orientation - target_orientation)
        return -orientation_diff
