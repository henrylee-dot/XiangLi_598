# 在 Windows 上采集外接 USB 2.0 PC Cam，通过 TCP 推给 WSL 里的 human_pose.py win。
# 用法见 human_pose.py 顶部。摄像头不能处于 usbipd bind 状态
# （已 bind 的话管理员 PowerShell: usbipd unbind --busid 2-2）。
#
# 拔线后 DirectShow 的 cap.read() 会卡死而不是返回失败，所以读帧放在后台线程，
# 超过 NO_FRAME_TIMEOUT 秒没有新帧就当作掉线，重新打开摄像头。

import socket
import struct
import sys
import threading
import time
import cv2

# Readable output when launched from WSL (stdout is a pipe there, not a console)
sys.stdout.reconfigure(encoding="utf-8")

# 用法: python.exe win_camera_sender.py [摄像头编号] [WSL 的 IP]
# 推荐传 WSL 的 IP（WSL 里 hostname -I 的第一个地址）。127.0.0.1 依赖 wslrelay
# 转发，它有时会接受连接却不转发给 WSL：推流端显示已连接，WSL 端却收不到画面。
CAMERA_INDEX = int(sys.argv[1]) if len(sys.argv) > 1 else 0
HOST = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1"
PORT = 5600
NO_FRAME_TIMEOUT = 3.0


class CameraReader:
    """Reads frames in a daemon thread so a hung cap.read() can be abandoned."""

    def __init__(self, cap):
        self.cap = cap
        self.frame = None
        self.frame_id = 0
        self.last_time = time.time()
        self.alive = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self.alive:
            ret, frame = self.cap.read()
            if not self.alive:
                break
            if ret:
                self.frame = frame
                self.frame_id += 1
                self.last_time = time.time()
            else:
                time.sleep(0.05)

    def stalled(self):
        return time.time() - self.last_time > NO_FRAME_TIMEOUT

    def abandon(self):
        self.alive = False
        # release() may hang on a vanished device too, so do it off the main thread
        threading.Thread(target=self.cap.release, daemon=True).start()


def open_camera():
    """Keeps trying until the camera opens and delivers a frame."""
    while True:
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if cap.isOpened():
            reader = CameraReader(cap)
            # DirectShow sometimes drops the first frames right after opening
            while not reader.stalled():
                if reader.frame_id > 0:
                    return reader
                time.sleep(0.05)
            reader.abandon()
        else:
            cap.release()
        print(f"摄像头 {CAMERA_INDEX} 无信号，1 秒后重试（检查线材；如果是 usbipd bind 状态，先 usbipd unbind --busid 2-2）", flush=True)
        time.sleep(1)


reader = open_camera()
print("摄像头已打开", flush=True)

while True:
    try:
        sock = socket.create_connection((HOST, PORT))
    except OSError:
        print(f"等待 WSL 端 human_pose.py 监听 {PORT} ...", flush=True)
        time.sleep(1)
        continue

    print("已连接，开始推流（Ctrl+C 停止）", flush=True)
    sent_id = 0
    try:
        while True:
            if reader.stalled():
                print("摄像头掉线，尝试重新连接 ...", flush=True)
                reader.abandon()
                reader = open_camera()
                sent_id = 0
                print("摄像头已恢复", flush=True)
            if reader.frame_id == sent_id:
                time.sleep(0.005)
                continue
            sent_id = reader.frame_id
            ok, jpg = cv2.imencode(".jpg", reader.frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            data = jpg.tobytes()
            sock.sendall(struct.pack(">I", len(data)) + data)
    except OSError:
        print("WSL 端断开", flush=True)
        sock.close()
        break
    except KeyboardInterrupt:
        break

reader.abandon()
