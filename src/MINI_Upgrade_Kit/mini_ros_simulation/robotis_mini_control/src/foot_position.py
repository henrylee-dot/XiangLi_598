#!/usr/bin/env python3
import rospy
import numpy as np
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from robotis_mini import RobotisMini
from dynamic_reconfigure.server import Server
from robotis_mini_control.cfg import RobotisMiniConfig

# Waveform parameters: millimetres and seconds; ROS private parameters override them.
start_time = 0 # sec
end_time = 30 # sec
control_period_ = 0.001 # sec
freq_n = 0.25 # hz
amplitude = 60 # mm; symmetric Gazebo range for 5 mm incremental inputs, not direct steps

signal_type_ = 'sine' # sine or triangle

def callback(config, level):
    """
    Dynamic reconfigure callback used for execute_static_foot_position().
    The robot will move to the specified position in sim when user changes
    x_foot_pos, y_foot_pos, or z_foot_pos param in rqt_dynamic_reconfigure GUI.
    """
    x_foot_pos = config['x_foot_pos']
    y_foot_pos = config['y_foot_pos']
    z_foot_pos = config['z_foot_pos']

    try:
        execute_static_foot_position(robot, x_foot_pos=x_foot_pos,
                                        y_foot_pos=y_foot_pos,
                                        z_foot_pos=z_foot_pos,
                                        time = 1.0)
    except ValueError as exc:
        rospy.logerr("Rejected foot position: %s", exc)

    print("x_foot_pos: ", x_foot_pos,
        "\ny_foot_pos: ", y_foot_pos,
        "\nz_foot_pos: ", z_foot_pos,
        "\n--------------------")

    return config

def execute_static_foot_position(robot, x_foot_pos, y_foot_pos, z_foot_pos, time):
    joint_values_right_hand = [0, 0, 0]
    joint_values_left_hand = [0, 0, 0]
    joint_values_right_foot = robot.ik_right_foot(robot.x_RF0+x_foot_pos, robot.y_RF0+y_foot_pos, z_foot_pos, robot.roll_RF0, robot.pitch_RF0)
    joint_values_left_foot = robot.ik_left_foot(robot.x_LF0+x_foot_pos, robot.y_LF0+y_foot_pos, z_foot_pos, robot.roll_LF0, robot.pitch_LF0)

    joint_pos_values = joint_values_right_hand + joint_values_left_hand \
                        + joint_values_right_foot + joint_values_left_foot
    if not np.all(np.isfinite(joint_pos_values)):
        raise ValueError("Foot target is outside the IK workspace")
    traj_msg = JointTrajectory()
    traj_msg.header.stamp = rospy.Time.now()
    traj_msg.joint_names = ['r_shoulder_joint','r_biceps_joint', 'r_elbow_joint',
                            'l_shoulder_joint','l_biceps_joint', 'l_elbow_joint',
                            'r_hip_joint', 'r_thigh_joint', 'r_knee_joint', 'r_ankle_joint', 'r_foot_joint',
                            'l_hip_joint', 'l_thigh_joint', 'l_knee_joint', 'l_ankle_joint', 'l_foot_joint']

    point = JointTrajectoryPoint()
    point.positions = joint_pos_values
    point.time_from_start = rospy.Duration(time)
    traj_msg.points.append(point)

    robot.execute_pub.publish(traj_msg)
    rospy.sleep(time)

def sine_wave_input():
    """
    Generate sine wave signal with a natural frequency (freq_n)
            of 0.25hz sampled every 0.001 sec (control_period_).
    """
    global start_time
    global end_time
    global control_period_
    global freq_n
    global amplitude

    t = _sample_times()
    return amplitude * np.sin(2.0 * np.pi * freq_n * t)

def triangle_wave_input():
    """
    Generate triangle wave signal with a natural frequency (freq_n)
            of 0.25hz sampled every 0.001 sec (control_period_).
    """
    global start_time
    global end_time
    global control_period_
    global freq_n
    global amplitude

    t = _sample_times()
    return amplitude * (2.0 / np.pi) * np.arcsin(np.sin(2.0 * np.pi * freq_n * t))

def _sample_times():
    if control_period_ <= 0 or end_time <= start_time or freq_n <= 0:
        raise ValueError("Require dt > 0, end > start, frequency > 0")
    return np.arange(start_time, end_time, control_period_)

def execute_variable_foot_position(robot, z_foot_pos):
    global control_period_
    global signal_type_

    if signal_type_ == 'sine':
        wave_signal = sine_wave_input()
    else:
        wave_signal = triangle_wave_input()

    time_from_start = 0.0
    x_offset = 0
    y_offset = 0

    traj_msg = JointTrajectory()
    traj_msg.header.stamp = rospy.Time.now()
    traj_msg.joint_names = ['r_shoulder_joint','r_biceps_joint', 'r_elbow_joint',
                            'l_shoulder_joint','l_biceps_joint', 'l_elbow_joint',
                            'r_hip_joint', 'r_thigh_joint', 'r_knee_joint', 'r_ankle_joint', 'r_foot_joint',
                            'l_hip_joint', 'l_thigh_joint', 'l_knee_joint', 'l_ankle_joint', 'l_foot_joint']

    for i, y_offset in enumerate(wave_signal):
        right = robot.ik_right_foot(robot.x_RF0 + x_offset,
                    robot.y_RF0 + y_offset, z_foot_pos, robot.roll_RF0, robot.pitch_RF0)
        left = robot.ik_left_foot(robot.x_LF0 + x_offset,
                    robot.y_LF0 + y_offset, z_foot_pos, robot.roll_LF0, robot.pitch_LF0)
        positions = [0.0] * 6 + right + left
        if not np.all(np.isfinite(positions)):
            raise ValueError("Non-finite IK at sample %d, y=%g mm" % (i, y_offset))
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = rospy.Duration.from_sec((i + 1) * control_period_)
        traj_msg.points.append(point)

    # Constructing 30,000 IK waypoints takes time. A zero stamp starts on receipt
    # in both ros_control and the provided MuJoCo interface.
    traj_msg.header.stamp = rospy.Time(0)
    robot.execute_pub.publish(traj_msg)


if __name__ == '__main__':
    rospy.init_node('foot_position')
    robot = RobotisMini()
    deadline = rospy.Time.now() + rospy.Duration(10.0)
    while robot.execute_pub.get_num_connections() == 0 and not rospy.is_shutdown():
        if rospy.Time.now() > deadline:
            raise RuntimeError("No trajectory controller connected")
        rospy.sleep(0.05)
    robot.init_pose(z_foot_pos=-166)
    mode = rospy.get_param('~mode', 'static')
    if mode == 'static':
        srv = Server(RobotisMiniConfig, callback)
    elif mode in ('sine', 'triangle'):
        signal_type_ = mode
        amplitude = float(rospy.get_param('~amplitude', amplitude))
        freq_n = float(rospy.get_param('~frequency', freq_n))
        end_time = float(rospy.get_param('~duration', end_time))
        control_period_ = float(rospy.get_param('~sample_period', control_period_))
        execute_variable_foot_position(robot, z_foot_pos=-166)
    else:
        raise ValueError("mode must be static, sine or triangle")
    rospy.spin()
