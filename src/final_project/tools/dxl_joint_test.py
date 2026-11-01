#!/usr/bin/env python3
"""关节方向校准：一次只让一个舵机转一个小角度再转回来，其余舵机保持无力矩。

用法（系统 python3，先 source /opt/ros/noetic/setup.bash）:
    python3 dxl_joint_test.py <ID:角度(°)> [ID:角度 ...] [--port /dev/ttyUSB0]
例: python3 dxl_joint_test.py 1:10 2:-10
每个舵机：目标先设为当前位置再开力矩（不会突然跳），限速转到 +角度，停 1 秒，
转回原位，关力矩，间隔 2 秒再测下一个。转动时该舵机的 LED 亮绿灯。
Ctrl+C 随时中断，会关掉所有力矩。
"""
import sys
import time

from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler

ADDR_TORQUE_ENABLE = 24
ADDR_LED = 25            # 0 灭，2 绿
ADDR_GOAL_POSITION = 30  # 2 字节，0-1023
ADDR_MOVING_SPEED = 32   # 2 字节，0 表示最快！所以一定要设一个小的值
ADDR_PRESENT_POSITION = 37
UNIT_DEG = 0.293         # 每个位置单位的角度
SPEED = 90               # 约 10 rpm ≈ 60°/s
NAMES = {1: '右肩', 2: '左肩', 3: '右上臂', 4: '左上臂', 5: '右肘', 6: '左肘', 7: '右髋侧摆',
         8: '左髋侧摆', 9: '右大腿', 10: '左大腿', 11: '右膝', 12: '左膝', 13: '右踝',
         14: '左踝', 15: '右脚侧摆', 16: '左脚侧摆'}


def check(result, error, packet, what):
    if result != COMM_SUCCESS:
        raise RuntimeError(f"{what} 通信失败: {packet.getTxRxResult(result)}")
    if error:
        print(f"  警告 {what}: {packet.getRxPacketError(error)}")


def retry(fn, *args, tries=3):
    """偶发丢包（USB 透传时会遇到）就重发，最多 tries 次。"""
    for _ in range(tries):
        out = fn(*args)
        if out[-2] == COMM_SUCCESS:
            return out
        time.sleep(0.01)
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    port_name = sys.argv[sys.argv.index('--port') + 1] if '--port' in sys.argv else '/dev/ttyUSB0'
    if '--port' in sys.argv:
        args.remove(port_name)
    tests = [(int(i), float(d)) for i, d in (a.split(':') for a in args)]

    port = PortHandler(port_name)
    packet = PacketHandler(2.0)
    if not port.openPort() or not port.setBaudRate(1000000):
        sys.exit(f"打不开串口 {port_name}")
    try:
        for n, (dxl_id, delta) in enumerate(tests):
            p0, r, e = retry(packet.read2ByteTxRx, port, dxl_id, ADDR_PRESENT_POSITION)
            check(r, e, packet, f"ID{dxl_id} 读位置")
            target = int(round(p0 + delta / UNIT_DEG))
            target = max(0, min(1023, target))
            print(f"[{n + 1}/{len(tests)}] ID{dxl_id} {NAMES.get(dxl_id, '')}: "
                  f"{p0} -> {target}（{delta:+.0f}°）", flush=True)
            check(*retry(packet.write2ByteTxRx, port, dxl_id, ADDR_MOVING_SPEED, SPEED), packet, "设速度")
            check(*retry(packet.write2ByteTxRx, port, dxl_id, ADDR_GOAL_POSITION, p0), packet, "目标=当前")
            check(*retry(packet.write1ByteTxRx, port, dxl_id, ADDR_TORQUE_ENABLE, 1), packet, "开力矩")
            retry(packet.write1ByteTxRx, port, dxl_id, ADDR_LED, 2)
            check(*retry(packet.write2ByteTxRx, port, dxl_id, ADDR_GOAL_POSITION, target), packet, "转动")
            time.sleep(1.2)
            p1, r, e = retry(packet.read2ByteTxRx, port, dxl_id, ADDR_PRESENT_POSITION)
            print(f"      实际到达 {p1}（转了 {(p1 - p0) * UNIT_DEG:+.1f}°）", flush=True)
            check(*retry(packet.write2ByteTxRx, port, dxl_id, ADDR_GOAL_POSITION, p0), packet, "转回")
            time.sleep(1.0)
            retry(packet.write1ByteTxRx, port, dxl_id, ADDR_LED, 0)
            check(*retry(packet.write1ByteTxRx, port, dxl_id, ADDR_TORQUE_ENABLE, 0), packet, "关力矩")
            if n + 1 < len(tests):
                time.sleep(2.0)
    finally:
        for dxl_id in range(1, 17):  # 无论如何都把力矩和灯关掉
            retry(packet.write1ByteTxRx, port, dxl_id, ADDR_TORQUE_ENABLE, 0)
            retry(packet.write1ByteTxRx, port, dxl_id, ADDR_LED, 0)
        port.closePort()
        print("已关闭全部力矩")


if __name__ == '__main__':
    main()
