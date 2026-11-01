# ws_mini — Robotis MINI camera motion imitation

Catkin workspace for ECE 598 (Robotis MINI). The final project makes the MINI imitate a person in front of a camera in real time, in MuJoCo simulation and on the real robot (KIMLAB upgrade kit).

```
camera -> MediaPipe pose -> retargeting -> center-of-mass safety filter -> /robotis_mini/whole_body_controller/command
                                                                              ├─ MuJoCo simulator
                                                                              └─ real robot bridge (U2D2 -> 16 x XL-320)
```

## Hardware: upgrade the robot first / 先改装机器人

The real-robot part requires a ROBOTIS MINI upgraded with **KIMLAB's ROBOTIS MINI Upgrade Kit** (Orange Pi Lite 2, U2D2, BNO055 IMU, USB camera, power distribution board). Assemble the kit by following KIMLAB's official video before running anything on the real robot:

真机部分需要先给 ROBOTIS MINI 装上 **KIMLAB 的 MINI 升级套件**（Orange Pi Lite 2、U2D2、BNO055 IMU、USB 摄像头、电源分配板）。请先按照 KIMLAB 官方视频完成改装，再运行真机程序：

**▶ [Assembly Instruction for KIMLAB's ROBOTIS MINI Upgrade Kit](https://youtu.be/Su98xT55EOg)**

The written manual is in `src/MINI_Upgrade_Kit/KIMLAB_MINI_Upgrade_Kit_manual.pdf`. The simulation part does not need any hardware.

文字版说明书在 `src/MINI_Upgrade_Kit/KIMLAB_MINI_Upgrade_Kit_manual.pdf`。仿真部分不需要任何硬件。

## Layout

| Path | Contents |
|---|---|
| `src/final_project/` | Final project ROS package: imitation node, retargeting, CoM safety filter, real-robot bridge, launch files, tools, report sources (`docs/`) |
| `vision/` | Standalone MediaPipe skeleton script (`human_pose.py`), Windows camera streamer (`win_camera_sender.py`), pose model |
| `src/MINI_Upgrade_Kit/` | Copy of [uiuckimlab/MINI_Upgrade_Kit](https://github.com/uiuckimlab/MINI_Upgrade_Kit) (branch `assn2`) with assignment-2 changes; provides the robot model and the MuJoCo/Gazebo simulators |

## Setup

Ubuntu 20.04 (WSL2), ROS Noetic, Python 3.8.

```bash
cd ~/ws_mini && catkin_make
# MuJoCo simulator environment
cd src/MINI_Upgrade_Kit/mini_ros_simulation/robotis_mini_mujoco
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# Pose estimation (system python): mediapipe 0.10.11 needs --no-deps on Python 3.8
pip3 install --user --no-deps mediapipe==0.10.11
pip3 install --user absl-py attrs "flatbuffers>=2.0" "protobuf>=3.11,<4" opencv-contrib-python==4.10.0.84 sounddevice
```

## Run

```bash
source ~/ws_mini/devel/setup.bash
# Simulation, laptop camera (/dev/video0)
roslaunch final_project mujoco_imitation.launch camera_source:=0
# Real robot (U2D2 on /dev/ttyUSB0); keep the robot suspended on first runs
roslaunch final_project real_imitation.launch camera_source:=0 imitate_legs:=true
```

Useful arguments: `imitate_legs:=false` (arms only), `balance_filter:=false`, `camera_source:=win` (external camera streamed from Windows with `vision/win_camera_sender.py`), `log_file:=<path>.npz`.

See `src/final_project/docs/` for the midterm report (Chinese and English LaTeX sources) with design details, test results, camera/U2D2 setup under WSL2, and troubleshooting.

## Copyright / 版权

- **Robot upgrade kit — © KIMLAB (Kinetic Intelligent Machine LAB).** The kit design, the assembly video and manual, and everything under `src/MINI_Upgrade_Kit/` (copied from [uiuckimlab/MINI_Upgrade_Kit](https://github.com/uiuckimlab/MINI_Upgrade_Kit)) belong to KIMLAB and remain under the licenses declared there (e.g. Apache 2.0 for `mini_ros_hardware`, BSD for `MINI_CAD`). Only the assignment-2 edits inside that folder are by Xiang Li.
- **Code — © Xiang Li.** The code in `src/final_project/` and `vision/` was written by Xiang Li.
- ROBOTIS MINI and Dynamixel are products of ROBOTIS.

- **机器人升级套件 —— 版权属于 KIMLAB（Kinetic Intelligent Machine LAB）。** 套件设计、组装视频和说明书，以及 `src/MINI_Upgrade_Kit/` 下的全部内容（来自 [uiuckimlab/MINI_Upgrade_Kit](https://github.com/uiuckimlab/MINI_Upgrade_Kit)）归 KIMLAB 所有，并沿用其中声明的许可证（如 `mini_ros_hardware` 为 Apache 2.0，`MINI_CAD` 为 BSD）。该目录中只有 assignment 2 的修改部分出自 Xiang Li。
- **代码 —— 版权属于 Xiang Li。** `src/final_project/` 和 `vision/` 中的代码由 Xiang Li 编写。
- ROBOTIS MINI 和 Dynamixel 是 ROBOTIS 公司的产品。
