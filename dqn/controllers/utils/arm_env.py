import tempfile

import numpy as np
from ikpy.chain import Chain

from controller import Supervisor


class ArmEnv(Supervisor):
    # Define joint limits for each motor
    JOINT_LIMITS = [
        (-3.1415, 3.1415),  # Link A motor
        (-1.5708, 2.61799),  # Link B motor
        (-3.1415, 1.309),  # Link C motor
        (-6.98132, 6.98132),  # Link D motor
        (-2.18166, 2.0944),  # Link E motor
        (-6.98132, 6.98132)  # Link F motor
    ]

    def __init__(self):
        super().__init__()

        # Set the simulation timestep
        self.time_step = int(4 * self.getBasicTimeStep())

        # Initialize properties
        self.arm_chain = self._initialize_chain()
        self.motors = self._initialize_motors()
        self.target_position = self.getFromDef('START').getPosition()
        self.pen = self.getFromDef('PEN')

        self.start_dist = None
        self.acc_dist = 0.0
        self.prev_dist = None

        # For the new reward function: store previous per-axis distances.
        # This will be initialized at the first reset.
        self.prev_axis_dists = None
        # Scaling factor lambda for the exponential terms (adjust as needed)
        self.lambda_scale = 1.0

        # For terminal reward/penalty: set maximum steps per episode.
        self.max_steps = 200  # Adjust to match your training configuration.
        self.current_step = 0

        # Initialize touch sensors
        self.touch_sensors = []
        for sensor_name in [
            "A touch sensor", "B touch sensor", "C touch sensor",
            "D touch sensor", "E touch sensor", "F touch sensor"
        ]:
            sensor = self.getDevice(sensor_name)
            sensor.enable(self.time_step)
            self.touch_sensors.append(sensor)

        # Run one simulation step to reset sensor states
        super().step(self.time_step)

    def _initialize_chain(self) -> 'Chain':
        """
            Initializes the robot's chain from its URDF file.
        """
        # Create a temporary URDF file
        with tempfile.NamedTemporaryFile(suffix='.urdf', delete=False) as file:
            filename = file.name
            file.write(self.getUrdf().encode('utf-8'))

        # Parse the URDF file into a kinematic chain
        active_links_mask = [False, True, True, True, True, True, True, False]
        return Chain.from_urdf_file(filename, active_links_mask=active_links_mask)

    def _initialize_motors(self) -> list:
        """
            Initializes the motors from the arm chain.
        """
        motors = []
        for link in self.arm_chain.links:
            if 'motor' in link.name:
                motor = self.getDevice(link.name)

                # Set different velocities for different joints
                if 'A' in link.name or 'B' in link.name:  # Base and shoulder joints
                    motor.setVelocity(5.0)  # Slower for larger movements
                elif 'C' in link.name or 'D' in link.name:  # Elbow and first wrist
                    motor.setVelocity(7.0)  # Medium speed
                else:  # E and F joints (wrist joints)
                    motor.setVelocity(9.0)  # Faster for smaller movements

                # Enable position sensors
                position_sensor = motor.getPositionSensor()
                position_sensor.enable(self.time_step)
                motors.append(motor)
        return motors

    @classmethod
    def initialize_supervisor(cls) -> 'ArmEnv':
        """
        Factory method to initialize and return an ArmEnv instance.
        """
        return cls()

    def reset(self) -> np.ndarray:
        self.simulationResetPhysics()
        self.simulationReset()
        super().step(self.time_step)

        # Initialize joints to a good starting position
        starting_positions = [
            0.0,   # Base joint
            -0.5,  # Shoulder
            0.5,   # Elbow
            0.0,   # Wrist 1
            0.0,   # Wrist 2
            0.0    # Wrist 3
        ]

        for i, motor in enumerate(self.motors):
            pos = np.clip(starting_positions[i], self.JOINT_LIMITS[i][0], self.JOINT_LIMITS[i][1])
            motor.setPosition(pos)

        super().step(self.time_step * 10)  # Wait for robot to settle

        self.acc_dist = 0.0
        current_pen = self._get_pen_position()
        self.start_dist = np.linalg.norm(self.target_position - current_pen)
        self.prev_dist = self.start_dist
        # Initialize per-axis distances (absolute difference along each axis)
        self.prev_axis_dists = np.abs(self.target_position - current_pen)

        # Reset step counter for a new episode.
        self.current_step = 0

        return self._get_state()

    def step(self, action: int = None) -> tuple[np.ndarray, float, bool, dict]:
        # Store previous pen position
        prev_pos = self._get_pen_position()

        # Apply action: determine joint and direction from action index
        joint_index = action // 2
        direction = 1 if action % 2 == 1 else -1
        current_position = self.motors[joint_index].getPositionSensor().getValue()

        # Adjust step size based on current distance to target
        current_pos = self._get_pen_position()
        distance_to_target = np.linalg.norm(self.target_position - current_pos)
        base_step = 0.05
        if distance_to_target > 1.0:
            step_size = base_step * 1.5
        elif distance_to_target < 0.2:
            step_size = base_step * 0.5
        else:
            step_size = base_step

        new_position = current_position + direction * step_size
        lower_limit, upper_limit = self.JOINT_LIMITS[joint_index]
        new_position = np.clip(new_position, lower_limit, upper_limit)
        self.motors[joint_index].setPosition(new_position)
        super().step(self.time_step)

        # Increment step counter
        self.current_step += 1

        # Get updated position and distances
        current_pos = self._get_pen_position()
        current_dist = np.linalg.norm(self.target_position - current_pos)
        step_dist = np.linalg.norm(current_pos - prev_pos)
        self.acc_dist += step_dist

        # ----- New Reward Function Implementation -----
        # Compute per-axis distances between pen tip and target
        current_axis_dists = np.abs(self.target_position - current_pos)
        reward_components = []
        for i in range(3):
            delta = current_axis_dists[i] - self.prev_axis_dists[i]
            if delta < 0:
                r = 1.1 * np.exp(-self.lambda_scale * (current_axis_dists[i] ** 2))
            else:
                r = -np.exp(-self.lambda_scale * (current_axis_dists[i] ** 2))
            reward_components.append(r)
        r1 = np.mean(reward_components)

        # Determine if the goal is reached
        if current_dist < 0.1:
            bonus = 20.0  # Strong bonus for reaching the goal
            success = True
        else:
            bonus = 0.0
            success = False

        reward = r1 + bonus

        # Apply collision penalty if needed
        if self._is_collision():
            reward -= 1.0

        # If max steps reached without success, apply a terminal penalty and mark as failure
        if self.current_step >= self.max_steps and not success:
            reward -= 5.0  # Terminal penalty for not reaching the goal
            done = False
        else:
            # Terminate episode if goal reached, else continue
            done = success

        # Update previous distances for the next step
        self.prev_axis_dists = current_axis_dists
        self.prev_dist = current_dist
        # ------------------------------------------------

        info = {
            'distance': current_dist,
            'accumulated_distance': self.acc_dist,
            'step': self.current_step,
            'goal_reached': success
        }

        return self._get_state(), reward, done, info

    def _get_state(self) -> np.ndarray:
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

    def _get_end_effector_position(self) -> np.ndarray:
        joint_positions = [motor.getPositionSensor().getValue() for motor in self.motors]
        fk_result = self.arm_chain.forward_kinematics([0] + list(joint_positions) + [0])
        return fk_result[:3, 3]

    def _get_pen_position(self):
        """Get the actual position of the pen tip."""
        pen_pos = self.pen.getPosition()
        # Adjust Z coordinate for pen tip (0.09 = total pen length)
        pen_tip_pos = np.array([pen_pos[0], pen_pos[1], pen_pos[2] - 0.09], dtype=np.float32)
        return pen_tip_pos

    def _is_collision(self) -> bool:
        # Check if any joint is at its limit
        for i, motor in enumerate(self.motors):
            pos = motor.getPositionSensor().getValue()
            lower, upper = self.JOINT_LIMITS[i]
            if abs(pos - lower) < 0.01 or abs(pos - upper) < 0.01:
                return True

        for sensor in self.touch_sensors:
            if sensor.getValue() == 1.0:
                print(f"Collision detected for sensor {sensor.getName()}")
                return True

        return False

    @staticmethod
    def _calculate_joint_cost(action):
        # Penalize excessive joint movement
        return -0.01 * np.abs(action)

    def _calculate_smoothness_reward(
            self, current_pos: np.ndarray, prev_pos: np.ndarray, prev_prev_pos: np.ndarray
    ) -> float:
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
