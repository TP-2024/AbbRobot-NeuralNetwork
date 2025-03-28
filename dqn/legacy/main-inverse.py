import math

from controller import Robot


def calculate_joint_angles(target_x, target_y, target_z):
    # Calculate A motor angle (base rotation)
    a_angle = math.atan2(target_y, target_x)

    # Calculate horizontal distance from base to target
    r = math.sqrt(target_x ** 2 + target_y ** 2)

    # Calculate B and C motor angles using inverse kinematics
    # These calculations are simplified and may need adjustment
    b_angle = -0.5  # Starting position for B motor
    c_angle = 0.5  # Starting position for C motor

    return a_angle, b_angle, c_angle


robot = Robot()
TIMESTEP = int(robot.getBasicTimeStep())

# Initialize motors
a_motor = robot.getDevice("A motor")
b_motor = robot.getDevice("B motor")
c_motor = robot.getDevice("C motor")

# Set motor velocities
a_motor.setVelocity(1.0)  # Slower speeds for more controlled movement
b_motor.setVelocity(1.0)
c_motor.setVelocity(1.0)

# Motor ranges
a_motor_range = [-3.1415, 3.1415]
b_motor_range = [-1.5708, 2.61799]
c_motor_range = [-3.1415, 1.309]
# Target position
target_position = (-5.87, -1.14, 1.64)

# Calculate target angles
target_a, target_b, target_c = calculate_joint_angles(*target_position)

# Clamp angles to motor ranges
target_a = max(min(target_a, a_motor_range[1]), a_motor_range[0])
target_b = max(min(target_b, b_motor_range[1]), b_motor_range[0])
target_c = max(min(target_c, c_motor_range[1]), c_motor_range[0])

# Current positions
current_a = 0.0
current_b = 0.0
current_c = 0.0

# Movement parameters
move_step = 0.001
threshold = 0.01  # Threshold for considering position reached


def move_towards_target(current, target, step):
    if abs(current - target) < threshold:
        return current
    elif current < target:
        return min(current + step, target)
    else:
        return max(current - step, target)


while robot.step(TIMESTEP) != -1:
    # Update positions
    current_a = move_towards_target(current_a, target_a, move_step)
    current_b = move_towards_target(current_b, target_b, move_step)
    current_c = move_towards_target(current_c, target_c, move_step)

    # Set motor positions
    a_motor.setPosition(current_a)
    b_motor.setPosition(current_b)
    c_motor.setPosition(current_c)

    # Check if target position is reached
    if (abs(current_a - target_a) < threshold and
            abs(current_b - target_b) < threshold and
            abs(current_c - target_c) < threshold):
        print("Target position reached")
        # break
