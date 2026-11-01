"""合成的人体关键点（MediaPipe 世界坐标），离线测试用。人面向摄像头。"""
import numpy as np
# Person facing the camera, in MediaPipe world coords (x right, y down, z away)
FWD, LEFT, UP = np.array([0, 0, -1.]), np.array([1., 0, 0]), np.array([0, -1., 0])
def to_mp(v): return v[0] * FWD + v[1] * LEFT + v[2] * UP
D = {'down': [0, 0, -1], 'up': [0, 0, 1], 'fwd': [1, 0, 0], 'back': [-1, 0, 0],
     'out_l': [0, 1, 0], 'out_r': [0, -1, 0], 'fwd_down': [0.7, 0, -0.7], 'back_down': [-0.5, 0, -0.87]}
def make(la=('down', 'down'), ra=('down', 'down'), ll=('down', 'down'), rl=('down', 'down'), lean=0.0):
    p = np.zeros((33, 3))
    def put(a, b, c, root, d1, d2, l1, l2):
        v1 = np.array(D[d1] if isinstance(d1, str) else d1, float); v2 = np.array(D[d2] if isinstance(d2, str) else d2, float)
        p[a] = root; p[b] = root + l1 * to_mp(v1 / np.linalg.norm(v1))
        p[c] = p[b] + l2 * to_mp(v2 / np.linalg.norm(v2))
    sh_l, sh_r = to_mp(np.array([0, .18, .5])), to_mp(np.array([0, -.18, .5]))
    hp_l, hp_r = to_mp(np.array([0, .1, 0])), to_mp(np.array([0, -.1, 0]))
    p[11], p[12], p[23], p[24] = sh_l, sh_r, hp_l, hp_r
    put(11, 13, 15, sh_l, *la, .3, .25); put(12, 14, 16, sh_r, *ra, .3, .25)
    put(23, 25, 27, hp_l, *ll, .45, .42); put(24, 26, 28, hp_r, *rl, .45, .42)
    if lean:  # rotate whole body about the person's left axis: torso frame must cancel it
        c, s = np.cos(lean), np.sin(lean); ax = LEFT
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        Rl = np.eye(3) + s * K + (1 - c) * K @ K; p = p @ Rl.T
    return p, np.ones(33)
