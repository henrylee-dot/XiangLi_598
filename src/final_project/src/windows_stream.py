"""TCP receiver for frames pushed by ~/ws_mini/vision/win_camera_sender.py.

Same as WindowsStream in vision/human_pose.py (that file is a script, so it
cannot be imported). Used for the external USB 2.0 PC Cam, which cannot be
passed into WSL with usbipd.
"""
import socket
import struct

import cv2
import numpy as np


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
        self.skipped = 0  # stale frames dropped by the last successful read()
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
            complete = 0
            while len(self.buf) >= 4:
                size = struct.unpack(">I", self.buf[:4])[0]
                if len(self.buf) < 4 + size:
                    break
                latest = self.buf[4:4 + size]
                self.buf = self.buf[4 + size:]
                complete += 1
            if latest is not None:
                self.skipped = complete - 1
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
