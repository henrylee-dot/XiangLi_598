#!/usr/bin/env python3
"""扫描 Dynamixel 总线：逐个点名 ID，读出型号、位置、电压、温度。只读，不开力矩，舵机不会动。

用法（系统 python3，需要先 source /opt/ros/noetic/setup.bash）:
    python3 dxl_scan.py [串口，默认 /dev/ttyUSB0]
XL-320 出厂波特率 1 Mbps；1 Mbps 扫不到时会自动再试其他波特率。
"""
import sys

from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler

PORT = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyUSB0'
BAUDRATES = [1000000, 57600, 115200, 9600]
MAX_ID = 20  # MINI 是 1-16，头部舵机可能用了别的编号，多扫几个

# XL-320 控制表地址（Protocol 2.0）
ADDR_TORQUE_ENABLE = 24
ADDR_PRESENT_POSITION = 37   # 2 字节，0-1023，512 为中间，每单位 0.29°
ADDR_PRESENT_VOLTAGE = 45    # 1 字节，单位 0.1 V
ADDR_PRESENT_TEMPERATURE = 46  # 1 字节，°C
XL320_MODEL = 350

# 套件代码里的编号（mini_ros_hardware/src/kinematics.cpp）：奇数右侧，偶数左侧
ID_NAMES = {
    1: '右肩', 2: '左肩', 3: '右上臂', 4: '左上臂', 5: '右肘', 6: '左肘',
    7: '右髋(侧摆)', 8: '左髋(侧摆)', 9: '右大腿', 10: '左大腿', 11: '右膝', 12: '左膝',
    13: '右踝', 14: '左踝', 15: '右脚(侧摆)', 16: '左脚(侧摆)',
}


def scan(port, packet, baud):
    port.setBaudRate(baud)
    found = []
    for dxl_id in range(1, MAX_ID + 1):
        model, result, error = packet.ping(port, dxl_id)
        if result == COMM_SUCCESS:
            found.append((dxl_id, model))
    return found


def read(port, packet, dxl_id, addr, size):
    if size == 1:
        value, result, error = packet.read1ByteTxRx(port, dxl_id, addr)
    else:
        value, result, error = packet.read2ByteTxRx(port, dxl_id, addr)
    return value if result == COMM_SUCCESS else None


def main():
    port = PortHandler(PORT)
    packet = PacketHandler(2.0)
    if not port.openPort():
        print(f"打不开串口 {PORT}。检查：U2D2 是否已用 usbipd 挂进 WSL（ls /dev/ttyUSB*），"
              f"以及是否有串口权限（dialout 组）。")
        sys.exit(1)

    found, baud = [], None
    for b in BAUDRATES:
        print(f"用 {b} bps 扫描 ID 1-{MAX_ID} ...")
        found = scan(port, packet, b)
        if found:
            baud = b
            break
    if not found:
        print("一个舵机都没扫到。检查：机器人是否通电（电池或 7.5V 适配器），"
              "U2D2 的 3 针线是否插在电源分配板上。")
        port.closePort()
        sys.exit(1)

    print(f"\n波特率 {baud} bps，找到 {len(found)} 个舵机：")
    print(" ID  部位          型号    力矩  位置(0-1023)  角度(°)  电压(V)  温度(°C)")
    for dxl_id, model in found:
        pos = read(port, packet, dxl_id, ADDR_PRESENT_POSITION, 2)
        volt = read(port, packet, dxl_id, ADDR_PRESENT_VOLTAGE, 1)
        temp = read(port, packet, dxl_id, ADDR_PRESENT_TEMPERATURE, 1)
        torque = read(port, packet, dxl_id, ADDR_TORQUE_ENABLE, 1)
        angle = f"{(pos - 512) * 0.293:7.1f}" if pos is not None else "    ?"
        model_text = "XL-320" if model == XL320_MODEL else str(model)
        print(f"{dxl_id:3d}  {ID_NAMES.get(dxl_id, '(未知)'):12s}  {model_text:6s}  "
              f"{'开' if torque else '关':3s}   {pos if pos is not None else '?':>6}      {angle}"
              f"    {volt / 10 if volt is not None else '?':>5}    {temp if temp is not None else '?':>5}")

    missing = [i for i in ID_NAMES if i not in {f[0] for f in found}]
    print("\n缺少的 ID:", missing if missing else "无，16 个关节舵机都在线")
    port.closePort()


if __name__ == '__main__':
    main()
