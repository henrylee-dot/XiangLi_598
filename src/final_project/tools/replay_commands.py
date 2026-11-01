#!/usr/bin/env python3
"""把录下来的关节命令（frames.npz 里的 cmd）按原来的时间节奏重新发布，用来在不开摄像头的情况下
检查真机（或仿真）动作。

用法（系统 python3，先 source ~/ws_mini/devel/setup.bash，并已启动 real_robot_bridge 或仿真）:
    python3 replay_commands.py <frames.npz> [--arms-only] [--start 秒] [--end 秒] [--speed 倍速]
--arms-only  腿保持默认站姿，只重放手臂
"""
import argparse
import os
import sys
import time

import numpy as np
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from pose_retarget import DEFAULT_POSE, JOINT_NAMES  # noqa: E402

ARM_WORDS = ('shoulder', 'biceps', 'elbow')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('frames')
    ap.add_argument('--arms-only', action='store_true')
    ap.add_argument('--start', type=float, default=0.0)
    ap.add_argument('--end', type=float, default=1e9)
    ap.add_argument('--speed', type=float, default=1.0)
    args = ap.parse_args(rospy.myargv()[1:])

    data = np.load(args.frames, allow_pickle=True)
    names = list(data['joint_names'])
    t = data['t_read'] - data['t_read'][0]
    cmd = data['cmd']
    keep = (t >= args.start) & (t <= args.end) & ~np.isnan(cmd[:, 0])
    t, cmd = t[keep] - t[keep][0], cmd[keep]
    if args.arms_only:
        for j, n in enumerate(names):
            if not any(w in n for w in ARM_WORDS):
                cmd[:, j] = DEFAULT_POSE[n]

    rospy.init_node('replay_commands')
    pub = rospy.Publisher('/robotis_mini/whole_body_controller/command', JointTrajectory, queue_size=1)
    time.sleep(1.0)  # let the subscriber connect
    print(f"重放 {len(t)} 条命令，时长 {t[-1] / args.speed:.1f} s" + ("（只有手臂）" if args.arms_only else ""))
    start = time.time()
    for k in range(len(t)):
        delay = t[k] / args.speed - (time.time() - start)
        if delay > 0:
            time.sleep(delay)
        if rospy.is_shutdown():
            break
        msg = JointTrajectory()
        msg.joint_names = JOINT_NAMES
        p = JointTrajectoryPoint()
        p.positions = [float(cmd[k, names.index(n)]) for n in JOINT_NAMES]
        p.time_from_start = rospy.Duration(0.1)
        msg.points.append(p)
        pub.publish(msg)
    print("重放结束")


if __name__ == '__main__':
    main()
