# 人体关节识别。必须在 vision/ 目录下运行（模型是相对路径）。
#
# 外接 USB 2.0 PC Cam（BUSID 2-2，画面自动旋转 180 度）：
#   不能 usbipd 挂进 WSL（只支持未压缩 YUYV，usbip 丢包导致绿屏），也不要 bind 它。
#   改为 Windows 采集、TCP 推给 WSL：
#     终端 1: python3 human_pose.py win
#     终端 2: python.exe "$(wslpath -w win_camera_sender.py)" 0 $(hostname -I | cut -d' ' -f1)
#   第二个参数是 WSL 的 IP。不传则连 127.0.0.1，那条路依赖 wslrelay 转发，
#   有时推流端显示"已连接"但这边一直收不到画面（窗口也不出来）。
#   按 q 退出。线材掉线时窗口显示 NO SIGNAL，推流端会自动重连摄像头。
#
# 笔记本自带摄像头（BUSID 1-3，已 bind）：
#   PowerShell 挂载: usbipd attach --wsl --busid 1-3   （重启 / 拔插后要重做）
#   运行: python3 human_pose.py 0
#   还给 Windows: usbipd detach --busid 1-3
#   读不到画面：ls /dev/video* 确认已挂载；Permission denied 就用
#     sg video -c "python3 human_pose.py 0"

import os
import socket
import struct
import sys
import time
import cv2
import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


MODEL_PATH = "pose_landmarker_full.task"

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Main body connections
connections = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
]

# Only draw joints used by the skeleton (no face, fingers or toes)
body_points = sorted({i for pair in connections for i in pair})

class WindowsStream:
    """Receives JPEG frames pushed by win_camera_sender.py over TCP.

    read() never blocks for long: while the sender is away or the camera has
    no signal it returns (False, None), so the window stays responsive and a
    new sender connection is accepted automatically.
    """

    def __init__(self, port):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("0.0.0.0", port))
        self.server.listen(1)
        self.server.settimeout(0.5)
        self.conn = None
        self.buf = b""
        print(f"等待 Windows 端 win_camera_sender.py 连接端口 {port} ...")

    def _drop(self):
        print("Windows 端断开，等待重新连接 ...")
        self.conn.close()
        self.conn = None
        self.buf = b""

    def read(self):
        if self.conn is None:
            try:
                self.conn, _ = self.server.accept()
            except socket.timeout:
                return False, None
            self.conn.settimeout(0.5)
            print("Windows 摄像头已连接")

        # Grab everything already waiting so stale frames are skipped, not queued:
        # the sender pushes 30 fps and pose detection is slower than that
        self.conn.setblocking(False)
        try:
            while True:
                chunk = self.conn.recv(1 << 20)
                if not chunk:
                    self._drop()
                    return False, None
                self.buf += chunk
        except BlockingIOError:
            pass
        except OSError:
            self._drop()
            return False, None
        self.conn.settimeout(0.5)

        # Keep only the newest whole frame (4-byte length + JPEG); wait if none yet
        while True:
            latest = None
            while len(self.buf) >= 4:
                size = struct.unpack(">I", self.buf[:4])[0]
                if len(self.buf) < 4 + size:
                    break
                latest = self.buf[4:4 + size]
                self.buf = self.buf[4 + size:]
            if latest is not None:
                frame = cv2.imdecode(np.frombuffer(latest, np.uint8), cv2.IMREAD_COLOR)
                return frame is not None, frame
            try:
                chunk = self.conn.recv(65536)
            except socket.timeout:
                return False, None
            except OSError:
                chunk = b""
            if not chunk:
                self._drop()
                return False, None
            self.buf += chunk

    def release(self):
        if self.conn is not None:
            self.conn.close()
        self.server.close()


SOURCE = sys.argv[1] if len(sys.argv) > 1 else "0"
USE_WINDOWS_STREAM = SOURCE == "win"
# The external PC Cam is mounted upside down
ROTATE_180 = USE_WINDOWS_STREAM

if USE_WINDOWS_STREAM:
    CAMERA_INDEX = None
    DEVICE = None
    cap = WindowsStream(5600)
else:
    CAMERA_INDEX = int(SOURCE)
    DEVICE = f"/dev/video{CAMERA_INDEX}"

    cap = cv2.VideoCapture(CAMERA_INDEX)
    # MJPG keeps bandwidth low enough for WSL2 usbip passthrough
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

with PoseLandmarker.create_from_options(options) as detector:

    start_time = time.perf_counter()
    last_frame = None

    while True:
        ret, frame = cap.read()

        if not ret and USE_WINDOWS_STREAM:
            # No signal: keep the window alive and wait for the stream to come back
            if last_frame is not None:
                no_signal = last_frame.copy()
                cv2.putText(no_signal, "NO SIGNAL - reconnecting...", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Human Pose Detection", no_signal)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        if not ret:
            print("Failed to read camera.")
            if not os.path.exists(DEVICE):
                print(f"没有 {DEVICE}：摄像头没挂载到 WSL（或编号不对）。请在 Windows PowerShell 运行：")
                print("    usbipd attach --wsl --busid 1-3    # 笔记本自带摄像头")
            elif not os.access(DEVICE, os.R_OK):
                print(f"没有 {DEVICE} 读权限：先 wsl --shutdown 重开，或用：")
                print(f'    sg video -c "python3 human_pose.py {CAMERA_INDEX}"')
            break

        if ROTATE_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        rgb_frame =cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(
            (time.perf_counter() - start_time) * 1000
        )

        result = detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        if result.pose_landmarks:

            landmarks = result.pose_landmarks[0]

            height, width = frame.shape[:2]

            # Draw landmarks
            for i in body_points:
                landmark = landmarks[i]
                if landmark.visibility > 0.5:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    cv2.circle(
                        frame,
                        (x, y),
                        4,
                        (0, 255, 0),
                        -1
                    )

            # Draw skeleton
            for start, end in connections:
                p1 = landmarks[start]
                p2 = landmarks[end]

                if p1.visibility > 0.5 and p2.visibility > 0.5:
                    x1 = int(p1.x * width)
                    y1 = int(p1.y * height)

                    x2 = int(p2.x * width)
                    y2 = int(p2.y * height)

                    cv2.line(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (255, 255, 255),
                        2
                    )

            # Print 3D coordinates
            if result.pose_world_landmarks:
                world = result.pose_world_landmarks[0]

                left_shoulder = world[11]
                left_elbow = world[13]
                left_wrist = world[15]

                print(
                    f"Shoulder: "
                    f"{left_shoulder.x:.3f}, "
                    f"{left_shoulder.y:.3f}, "
                    f"{left_shoulder.z:.3f}"
                )

        last_frame = frame
        cv2.imshow("Human Pose Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()