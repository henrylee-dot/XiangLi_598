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

Ubuntu 20.04 (tested under WSL2 on Windows), ROS Noetic, Python 3.8. The workspace can be cloned to any folder.

```bash
# 1. ROS and system packages
sudo apt install ros-noetic-desktop-full ros-noetic-dynamixel-sdk \
    ros-noetic-ros-control ros-noetic-ros-controllers python3-pip python3-venv
# 2. Build
git clone https://github.com/henrylee-dot/XiangLi_598.git ws_mini && cd ws_mini
source /opt/ros/noetic/setup.bash && catkin_make
# 3. MuJoCo simulator environment (upgrade pip first: the venv's default pip is too old)
cd src/MINI_Upgrade_Kit/mini_ros_simulation/robotis_mini_mujoco
python3 -m venv .venv && .venv/bin/pip install -U "pip<25" && .venv/bin/pip install -r requirements.txt
cd -
# 4. Pose estimation (system python): mediapipe 0.10.11 is the last release for Python 3.8
#    and must be installed with --no-deps (its jaxlib dependency has no Python 3.8 build)
pip3 install --user --no-deps mediapipe==0.10.11
pip3 install --user absl-py attrs "flatbuffers>=2.0" "protobuf>=3.11,<4" opencv-contrib-python==4.10.0.84 sounddevice
# 5. Device permissions (camera, U2D2); log out/in (or `wsl --shutdown`) afterwards
sudo usermod -aG video,dialout $USER
```

**WSL2 only**: USB devices must be passed into WSL with [usbipd-win](https://github.com/dorssel/usbipd-win) on Windows (`usbipd bind --busid <BUSID> --force` once as admin, then `usbipd attach --wsl --busid <BUSID>` after every reboot or re-plug; find the BUSID with `usbipd list`). On native Ubuntu this step is not needed.

**External USB camera** (`camera_source:=win`): some cameras cannot go through usbipd (uncompressed YUYV only). Capture them on Windows instead: install Python with `opencv-python` on Windows and run `vision/win_camera_sender.py` there (see Run).

**Real robot**: connect the U2D2 to the PC (it appears as `/dev/ttyUSB0`). Joint directions can differ between robots, so check yours before the first run with the tools in `src/final_project/tools/` (with the robot suspended): `dxl_scan.py` (read-only scan), `dxl_joint_test.py` (one joint at a time), `dxl_pose_test.py zero` (all joints to 0°). Record the result in `src/final_project/config/real_robot_calibration.yaml`.

## Run

```bash
source devel/setup.bash
# Simulation, laptop camera (/dev/video0)
roslaunch final_project mujoco_imitation.launch camera_source:=0
# Simulation, external camera streamed from Windows (second terminal)
roslaunch final_project mujoco_imitation.launch camera_source:=win
python.exe "$(wslpath -w vision/win_camera_sender.py)" 0 $(hostname -I | cut -d' ' -f1)
# Real robot; keep it suspended on first runs, unplug power to stop in an emergency
roslaunch final_project real_imitation.launch camera_source:=0 imitate_legs:=true
```

Useful arguments: `imitate_legs:=false` (arms only; the real-robot default), `balance_filter:=false`, `log_file:=<path>.npz`. Stand with your whole body in view for about a second before the robot starts following. Backspace in the MuJoCo window resets the simulated robot.

Offline check without camera or robot (simulation regression test of the center-of-mass filter):

```bash
cd src/MINI_Upgrade_Kit/mini_ros_simulation
robotis_mini_mujoco/.venv/bin/python3 ../../final_project/tools/test_balance_filter.py \
    robotis_mini_description/urdf/mini_scene.xml ../../final_project/src \
    ../../final_project/tools ../../final_project/config/mini_mass_model.json
```

See `src/final_project/docs/` for the midterm report (Chinese and English LaTeX sources) with design details, test results and troubleshooting.

## Copyright / 版权

- **Robot upgrade kit — © KIMLAB (Kinetic Intelligent Machine LAB).** The kit design, the assembly video and manual, and everything under `src/MINI_Upgrade_Kit/` (copied from [uiuckimlab/MINI_Upgrade_Kit](https://github.com/uiuckimlab/MINI_Upgrade_Kit)) belong to KIMLAB and remain under the licenses declared there (e.g. Apache 2.0 for `mini_ros_hardware`, BSD for `MINI_CAD`). Only the assignment-2 edits inside that folder are by Xiang Li.
- **Code — © Xiang Li.** The code in `src/final_project/` and `vision/` was written by Xiang Li.
- ROBOTIS MINI and Dynamixel are products of ROBOTIS.

- **机器人升级套件 —— 版权属于 KIMLAB（Kinetic Intelligent Machine LAB）。** 套件设计、组装视频和说明书，以及 `src/MINI_Upgrade_Kit/` 下的全部内容（来自 [uiuckimlab/MINI_Upgrade_Kit](https://github.com/uiuckimlab/MINI_Upgrade_Kit)）归 KIMLAB 所有，并沿用其中声明的许可证（如 `mini_ros_hardware` 为 Apache 2.0，`MINI_CAD` 为 BSD）。该目录中只有 assignment 2 的修改部分出自 Xiang Li。
- **代码 —— 版权属于 Xiang Li。** `src/final_project/` 和 `vision/` 中的代码由 Xiang Li 编写。
- ROBOTIS MINI 和 Dynamixel 是 ROBOTIS 公司的产品。
