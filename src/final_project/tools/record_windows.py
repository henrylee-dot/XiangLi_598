"""在 Windows 上录制 MuJoCo 和摄像头两个窗口，左右拼成一个 mp4。

用 Windows 的 python.exe 运行（WSLg 窗口在 Windows 桌面上显示）：
    python.exe record_windows.py <输出.mp4> <停止标记文件> [fps]
停止标记文件出现后结束并写完视频。录的是窗口所在的屏幕区域，所以录制期间
把两个窗口设为"总在最前"并左右分开摆放，结束后取消置顶。第一帧另存为
<输出.mp4>.first.jpg，方便确认录到的是对的画面。

输出是 OpenCV 的 mpeg4 编码，VS Code / 浏览器放不了，录完在 WSL 里转成 H.264：
    ffmpeg -i screen.mp4 -c:v libx264 -pix_fmt yuv420p -movflags +faststart out.mp4
"""
import ctypes
import ctypes.wintypes as wt
import os
import sys
import time

import cv2
import numpy as np
from PIL import ImageGrab

TITLES = ["MuJoCo : ", "Pose Imitation"]  # 窗口标题前缀，左、右
HEIGHT = 540  # 每个窗口缩放到的高度

user32 = ctypes.windll.user32
user32.SetProcessDPIAware()  # 拿到真实像素坐标，和截屏一致


def find_window(prefix):
    found = []

    def callback(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            if buf.value.startswith(prefix):
                found.append(hwnd)
        return True

    proc = ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)(callback)
    user32.EnumWindows(proc, 0)
    return found[0] if found else None


def window_rect(hwnd):
    rect = wt.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    return rect.left, rect.top, rect.right, rect.bottom


HWND_TOPMOST, HWND_NOTOPMOST = -1, -2
SWP_NOSIZE, SWP_NOMOVE, SWP_SHOWWINDOW = 0x0001, 0x0002, 0x0040


def pin_side_by_side(hwnds):
    """置顶并排开：第一个窗口贴左边，最后一个贴右边，互不遮挡。"""
    screen_w = user32.GetSystemMetrics(0)
    for i, hwnd in enumerate(hwnds):
        left, top, right, bottom = window_rect(hwnd)
        x = 0 if i == 0 else screen_w - (right - left)
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE，防止窗口是最小化的
        user32.SetWindowPos(hwnd, HWND_TOPMOST, x, 0, 0, 0, SWP_NOSIZE | SWP_SHOWWINDOW)


def unpin(hwnds):
    for hwnd in hwnds:
        if user32.IsWindow(hwnd):
            user32.SetWindowPos(hwnd, HWND_NOTOPMOST, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE)


def grab(hwnd):
    """窗口区域截图，缩放到 HEIGHT 高；窗口不在了返回 None。"""
    if hwnd is None or not user32.IsWindow(hwnd):
        return None
    left, top, right, bottom = window_rect(hwnd)
    if right - left < 10 or bottom - top < 10:
        return None
    img = np.array(ImageGrab.grab(bbox=(left, top, right, bottom), all_screens=True))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    width = int(img.shape[1] * HEIGHT / img.shape[0])
    return cv2.resize(img, (width, HEIGHT))


def main():
    out_path, stop_file = sys.argv[1], sys.argv[2]
    fps = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0

    hwnds = [None] * len(TITLES)
    print("等待窗口出现: " + ", ".join(TITLES), flush=True)
    while not all(hwnds) and not os.path.exists(stop_file):
        hwnds = [h or find_window(t) for h, t in zip(hwnds, TITLES)]
        time.sleep(0.5)
    if os.path.exists(stop_file):
        return
    pin_side_by_side(hwnds)
    time.sleep(0.5)  # 等窗口移动、重绘完成
    print("找到窗口，已置顶并排，开始录制", flush=True)
    try:
        record(hwnds, out_path, stop_file, fps)
    finally:
        unpin(hwnds)


def record(hwnds, out_path, stop_file, fps):
    widths = [None] * len(TITLES)
    writer = None
    frames = 0
    next_t = time.perf_counter()
    while not os.path.exists(stop_file):
        parts = []
        for i, hwnd in enumerate(hwnds):
            img = grab(hwnd)
            if img is None:
                img = np.zeros((HEIGHT, widths[i] or 640, 3), np.uint8)
            if widths[i] is None:
                widths[i] = img.shape[1]
            elif img.shape[1] != widths[i]:  # 窗口被拉伸过，保持视频尺寸不变
                img = cv2.resize(img, (widths[i], HEIGHT))
            parts.append(img)
        frame = np.hstack(parts)
        if writer is None:
            cv2.imwrite(out_path + ".first.jpg", frame)
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (frame.shape[1], frame.shape[0]))
        # 截屏跟不上时重复写这一帧补齐错过的时间槽，保证视频播放速度和现实一致
        slots = 1
        next_t += 1.0 / fps
        while time.perf_counter() > next_t + 1.0 / fps:
            next_t += 1.0 / fps
            slots += 1
        for _ in range(slots):
            writer.write(frame)
        frames += slots
        delay = next_t - time.perf_counter()
        if delay > 0:
            time.sleep(delay)

    if writer is not None:
        writer.release()
    print(f"录制结束，共 {frames} 帧，保存到 {out_path}", flush=True)


if __name__ == "__main__":
    main()
