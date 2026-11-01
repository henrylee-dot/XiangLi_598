# ws_mini — Robotis MINI camera motion imitation

Catkin workspace for ECE 598 (Robotis MINI). The final project makes the MINI imitate a person in front of a camera in real time, in MuJoCo simulation and on the real robot (KIMLAB upgrade kit).

```
camera -> MediaPipe pose -> retargeting -> center-of-mass safety filter -> /robotis_mini/whole_body_controller/command
                                                                              ├─ MuJoCo simulator
                                                                              └─ real robot bridge (U2D2 -> 16 x XL-320)
```

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
