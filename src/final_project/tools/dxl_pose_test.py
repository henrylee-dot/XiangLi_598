#!/usr/bin/env python3
"""让 16 个舵机一起慢慢转到一个姿势，保持几秒，再慢慢回到起始位置并关力矩。

用法（系统 python3，先 source /opt/ros/noetic/setup.bash）:
    python3 dxl_pose_test.py zero|default [保持秒数，默认 5]
  zero    全部关节 0°（仿真里：手臂水平伸向两侧，腿伸直，脚底平）
  default 仿真的默认站姿（微蹲，手臂下垂、手肘弯曲）
角度按 config/real_robot_calibration.yaml 换算成舵机位置。Ctrl+C 随时中断并关力矩。
"""
import os
import sys
import time

import yaml
from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from pose_retarget import DEFAULT_POSE  # noqa: E402

ADDR_TORQUE_ENABLE = 24
ADDR_GOAL_POSITION = 30
ADDR_MOVING_SPEED = 32
ADDR_PRESENT_POSITION = 37
SPEED = 60  # 约 6.7 rpm ≈ 40°/s，全身一起动时慢一点
CALIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config',
                     'real_robot_calibration.yaml')


def retry(fn, *args, tries=3):
    for _ in range(tries):
        out = fn(*args)
        if out[-2] == COMM_SUCCESS:
            return out
        time.sleep(0.01)
    raise RuntimeError(f"通信失败: {fn.__name__}{args[1:]}")


def to_ticks(cal, name, angle):
    j = cal['joints'][name]
    return max(0, min(1023, int(round(j['center'] + j['sign'] * angle * cal['units_per_rad']))))


def wait_until_reached(port, packet, goals, timeout=6.0):
    start = time.time()
    while time.time() - start < timeout:
        far = [i for i, g in goals.items()
               if abs(retry(packet.read2ByteTxRx, port, i, ADDR_PRESENT_POSITION)[0] - g) > 8]
        if not far:
            return []
        time.sleep(0.1)
    return far


def main():
    pose_name = sys.argv[1] if len(sys.argv) > 1 else 'zero'
    hold = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    with open(CALIB) as f:
        cal = yaml.safe_load(f)
    pose = {n: 0.0 for n in cal['joints']} if pose_name == 'zero' else DEFAULT_POSE

    port = PortHandler(cal['port'])
    packet = PacketHandler(cal['protocol'])
    if not port.openPort() or not port.setBaudRate(cal['baudrate']):
        sys.exit(f"打不开串口 {cal['port']}")
    ids = {n: j['id'] for n, j in cal['joints'].items()}
    try:
        start = {i: retry(packet.read2ByteTxRx, port, i, ADDR_PRESENT_POSITION)[0] for i in ids.values()}
        goals = {ids[n]: to_ticks(cal, n, pose[n]) for n in ids}
        for i in ids.values():  # 目标先设为当前位置，开力矩时不会跳
            retry(packet.write2ByteTxRx, port, i, ADDR_MOVING_SPEED, SPEED)
            retry(packet.write2ByteTxRx, port, i, ADDR_GOAL_POSITION, start[i])
            retry(packet.write1ByteTxRx, port, i, ADDR_TORQUE_ENABLE, 1)
        print(f"转到 {pose_name} 姿势 ...", flush=True)
        for i, g in goals.items():
            retry(packet.write2ByteTxRx, port, i, ADDR_GOAL_POSITION, g)
        far = wait_until_reached(port, packet, goals)
        print("到位" + (f"（没完全到位的 ID: {far}）" if far else ""), flush=True)
        print(" ID  关节               目标  实际  差(°)")
        for n, i in sorted(ids.items(), key=lambda x: x[1]):
            p = retry(packet.read2ByteTxRx, port, i, ADDR_PRESENT_POSITION)[0]
            print(f"{i:3d}  {n:18s} {goals[i]:4d}  {p:4d}  {(p - goals[i]) * 0.293:+5.1f}")
        print(f"保持 {hold:.0f} 秒 ...", flush=True)
        time.sleep(hold)
        print("回到起始位置 ...", flush=True)
        for i in ids.values():
            retry(packet.write2ByteTxRx, port, i, ADDR_GOAL_POSITION, start[i])
        wait_until_reached(port, packet, start)
    finally:
        for i in range(1, 17):
            try:
                retry(packet.write1ByteTxRx, port, i, ADDR_TORQUE_ENABLE, 0)
            except RuntimeError:
                print(f"ID{i} 关力矩失败，请拔电池！")
        port.closePort()
        print("已关闭全部力矩")


if __name__ == '__main__':
    main()
