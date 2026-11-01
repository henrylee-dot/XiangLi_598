#!/usr/bin/env python3
"""Camera pose imitation: MediaPipe body pose -> Robotis MINI joint trajectory.

Reads the laptop camera, detects the human pose with MediaPipe, retargets it to
the MINI's 16 joints (pose_retarget.py) and streams short JointTrajectory goals
to /robotis_mini/whole_body_controller/command. Left maps to left.

Run after the simulator is up (see launch/mujoco_imitation.launch). Press q in
the camera window to quit; Backspace in the MuJoCo viewer resets the robot.

Camera source (~camera_source):
  "0", "1", ...  a /dev/videoN camera passed into WSL. Laptop camera:
                 usbipd attach --wsl --busid 1-3   (Windows PowerShell)
  "win"          external USB 2.0 PC Cam captured on Windows and pushed over
                 TCP 5600 (it cannot go through usbipd). Start the sender:
                 python.exe "$(wslpath -w ~/ws_mini/vision/win_camera_sender.py)" \
                     0 $(hostname -I | cut -d' ' -f1)
                 (pass the WSL IP: via 127.0.0.1 wslrelay sometimes accepts
                 the connection without forwarding it, so no frames arrive)
                 The PC Cam is mounted upside down, so frames are rotated 180.
See ~/ws_mini/vision/human_pose.py for the full camera notes.
"""
import os
import time

import cv2
import mediapipe as mp
import numpy as np
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from balance_filter import BalanceFilter
from pose_retarget import JOINT_NAMES, Retargeter
from windows_stream import WindowsStream

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# Main body connections (same as vision/human_pose.py)
CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23),
    (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28),
]
BODY_POINTS = sorted({i for pair in CONNECTIONS for i in pair})


def open_camera(index):
    cap = cv2.VideoCapture(index)
    # MJPG keeps bandwidth low enough for WSL2 usbip passthrough
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def camera_hint(index):
    device = f"/dev/video{index}"
    if not os.path.exists(device):
        return (f"没有 {device}：摄像头没挂载到 WSL。请在 Windows PowerShell 运行：\n"
                "    usbipd attach --wsl --busid 1-3")
    if not os.access(device, os.R_OK):
        return f"没有 {device} 读权限：先 wsl --shutdown 重开 WSL（video 组生效）"
    return "摄像头打开了但读不到画面，检查是否被别的程序占用"


def draw_skeleton(frame, landmarks):
    height, width = frame.shape[:2]
    for start, end in CONNECTIONS:
        p1, p2 = landmarks[start], landmarks[end]
        if p1.visibility > 0.5 and p2.visibility > 0.5:
            cv2.line(frame, (int(p1.x * width), int(p1.y * height)),
                     (int(p2.x * width), int(p2.y * height)), (255, 255, 255), 2)
    for i in BODY_POINTS:
        lm = landmarks[i]
        if lm.visibility > 0.5:
            cv2.circle(frame, (int(lm.x * width), int(lm.y * height)), 4, (0, 255, 0), -1)


class FrameLogger:
    """Per-frame timing, landmarks and commands, saved to an .npz for analysis.

    Saved every few seconds and on shutdown so a killed run still leaves data.
    """

    def __init__(self, path, save_every=5.0):
        self.path = path
        self.save_every = save_every
        self.last_save = time.time()
        self.rows = {k: [] for k in ('t_read', 't_done', 'detect_ms', 'skipped', 'world',
                                     'vis', 'image_xy', 'cmd', 'cmd_raw', 'state', 'stance')}

    def add(self, t_read, t_done, detect_ms, skipped, world, vis, image_xy, cmd, cmd_raw,
            state, stance):
        nan = np.full
        self.rows['t_read'].append(t_read)
        self.rows['t_done'].append(t_done)
        self.rows['detect_ms'].append(detect_ms)
        self.rows['skipped'].append(skipped)
        self.rows['world'].append(world if world is not None else nan((33, 3), np.nan))
        self.rows['vis'].append(vis if vis is not None else nan(33, np.nan))
        self.rows['image_xy'].append(image_xy if image_xy is not None else nan((33, 2), np.nan))
        self.rows['cmd'].append([cmd[n] for n in JOINT_NAMES] if cmd else nan(16, np.nan))
        self.rows['cmd_raw'].append([cmd_raw[n] for n in JOINT_NAMES] if cmd_raw else nan(16, np.nan))
        self.rows['state'].append(state)
        self.rows['stance'].append(stance)
        if t_done - self.last_save > self.save_every:
            self.save()

    def save(self):
        self.last_save = time.time()
        tmp = self.path + '.tmp.npz'
        np.savez(tmp, joint_names=JOINT_NAMES,
                 **{k: np.array(v) for k, v in self.rows.items()})
        os.replace(tmp, self.path)


def main():
    rospy.init_node('pose_imitation')
    # Default: <workspace>/vision/, found relative to this file (src/final_project/src/)
    workspace = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
    model_path = os.path.expanduser(rospy.get_param(
        '~model_path', os.path.join(workspace, 'vision', 'pose_landmarker_full.task')))
    camera_source = str(rospy.get_param('~camera_source', '0'))
    use_windows_stream = camera_source == 'win'
    rotate_180 = rospy.get_param('~rotate_180', use_windows_stream)
    show_window = rospy.get_param('~show_window', True)
    # Trajectory duration per command; the sim interpolates toward each new goal
    goal_time = rospy.get_param('~goal_time', 0.1)
    log_file = os.path.expanduser(rospy.get_param('~log_file', ''))
    # Static center-of-mass safety filter on the leg commands (balance_filter.py)
    balance = None
    if rospy.get_param('~balance_filter', True):
        balance = BalanceFilter(os.path.expanduser(rospy.get_param(
            '~mass_model', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        '..', 'config', 'mini_mass_model.json'))))
    logger = FrameLogger(log_file) if log_file else None
    retargeter = Retargeter(
        visibility=rospy.get_param('~visibility', 0.5),
        smoothing=rospy.get_param('~smoothing', 0.3),
        max_step=rospy.get_param('~max_step', 0.4),
        imitate_legs=rospy.get_param('~imitate_legs', True),
    )

    pub = rospy.Publisher('/robotis_mini/whole_body_controller/command',
                          JointTrajectory, queue_size=1)

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    if use_windows_stream:
        cap = WindowsStream(5600)
    else:
        cap = open_camera(int(camera_source))
    rospy.loginfo("pose_imitation: camera %s, model %s, legs %s, balance filter %s",
                  camera_source, model_path, retargeter.imitate_legs, balance is not None)

    with PoseLandmarker.create_from_options(options) as detector:
        start_time = time.perf_counter()
        while not rospy.is_shutdown():
            ret, frame = cap.read()
            if not ret and use_windows_stream:
                # Sender away or camera unplugged: keep waiting, robot holds its pose
                if show_window and cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            if not ret:
                rospy.logerr("Failed to read camera.\n%s", camera_hint(int(camera_source)))
                break
            t_read = time.time()
            if rotate_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((time.perf_counter() - start_time) * 1000)
            result = detector.detect_for_video(mp_image, timestamp_ms)
            detect_ms = (time.time() - t_read) * 1000
            points = vis = image_xy = cmd = cmd_raw = None

            status = "no person"
            if not (result.pose_landmarks and result.pose_world_landmarks):
                retargeter.lost()
            else:
                landmarks = result.pose_landmarks[0]
                world = result.pose_world_landmarks[0]
                points = np.array([[lm.x, lm.y, lm.z] for lm in world])
                # Image-space visibility is the reliable one; world landmarks copy it
                vis = np.array([lm.visibility for lm in landmarks])
                image_xy = np.array([[lm.x, lm.y] for lm in landmarks])
                cmd = cmd_raw = retargeter.update(points, vis, image_xy)
                if balance:
                    cmd = balance.update(cmd_raw)

                msg = JointTrajectory()
                msg.joint_names = JOINT_NAMES
                point = JointTrajectoryPoint()
                point.positions = [cmd[n] for n in JOINT_NAMES]
                point.time_from_start = rospy.Duration(goal_time)
                msg.points.append(point)
                pub.publish(msg)

                if retargeter.state == retargeter.WAITING:
                    status = "stand with full body in view %d/%d" % (
                        retargeter.progress, retargeter.start_frames)
                elif retargeter.state == retargeter.BLENDING:
                    status = "starting %d/%d" % (retargeter.progress, retargeter.blend_frames)
                else:
                    legs_seen = retargeter._visible(vis, (25, 26, 27, 28), image_xy)
                    status = "tracking" + ("" if legs_seen else " (legs not in view, holding)")
                if balance:
                    status += "  [support: %s]" % balance.info['stance']
                draw_skeleton(frame, landmarks)

            if show_window:
                cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)
                cv2.imshow("Pose Imitation", frame)
                key = cv2.waitKey(1) & 0xFF
            if logger:
                logger.add(t_read, time.time(), detect_ms, getattr(cap, 'skipped', 0),
                           points, vis, image_xy, cmd, cmd_raw,
                           retargeter.state if cmd else 'no person',
                           balance.info.get('stance', '') if balance else '')
            if show_window and key == ord("q"):
                break

    if logger:
        logger.save()
        rospy.loginfo("pose_imitation: log saved to %s", log_file)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
