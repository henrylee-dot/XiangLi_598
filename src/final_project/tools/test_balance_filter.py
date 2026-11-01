"""重心安全过滤的仿真回归测试：几种腿部动作，比较有/无过滤时机器人会不会倒。

用法（仿真 venv 的 python，在 mini_ros_simulation 目录下）:
    robotis_mini_mujoco/.venv/bin/python3 <本文件> robotis_mini_description/urdf/mini_scene.xml \
        ~/ws_mini/src/final_project/src ~/ws_mini/src/final_project/tools \
        ~/ws_mini/src/final_project/config/mini_mass_model.json
"""
import sys, time, mujoco, numpy as np
sys.path.insert(0, sys.argv[2]); sys.path.insert(0, sys.argv[3])
from pose_retarget import Retargeter, DEFAULT_POSE, JOINT_NAMES
from balance_filter import BalanceFilter
from synthetic_human import make
m = mujoco.MjModel.from_xml_path(sys.argv[1])
act = [m.joint(m.actuator_trnid[a, 0]).name for a in range(m.nu)]
MODEL = sys.argv[4]
def ang(deg_fwd, deg_side=0.0):
    """unit vector pointing down, rotated forward / sideways (robot frame)"""
    f, s = np.radians(deg_fwd), np.radians(deg_side)
    v = np.array([np.sin(f), np.sin(s), -np.cos(f) * np.cos(s)]); return v / np.linalg.norm(v)
def scenario(kind, t):
    a = np.clip(min(t / 2.0, (7.0 - t) / 2.0), 0, 1) if t < 7 else 0.0  # 2s 抬起，保持 3s，2s 放下
    kw = {}
    if kind == '抬左膝 70°':   kw['ll'] = (ang(70 * a), ang(0))
    if kind == '抬右膝 70°':   kw['rl'] = (ang(70 * a), ang(0))
    if kind == '左腿前迈 30°': kw['ll'] = (ang(30 * a), ang(30 * a))
    if kind == '右腿侧抬 40°': kw['rl'] = (ang(0, -40 * a), ang(0, -40 * a))
    if kind == '深蹲':        kw['ll'] = kw['rl'] = (ang(80 * a), ang(-60 * a))
    if kind == '中蹲 40°/20°': kw['ll'] = kw['rl'] = (ang(40 * a), ang(-20 * a))
    return make(**kw)
def run(cmds, ts):
    d = mujoco.MjData(m)
    for n, v in DEFAULT_POSE.items(): d.qpos[m.jnt_qposadr[m.joint(n).id]] = v
    d.ctrl[:] = [DEFAULT_POSE[n] for n in act]
    for _ in range(1000): mujoco.mj_step(m, d)
    k = 0; start = d.ctrl.copy(); goal = start.copy(); tg = 0; maxtilt = 0; fell = None; t0 = d.time
    while d.time - t0 < ts[-1] + 1.0:
        now = d.time - t0
        while k < len(ts) and ts[k] <= now:
            start, goal, tg = d.ctrl.copy(), np.array([cmds[k][n] for n in act]), now; k += 1
        d.ctrl[:] = start + min(1, (now - tg) / 0.1) * (goal - start)
        mujoco.mj_step(m, d)
        tl = np.degrees(np.arccos(np.clip(d.xmat[1].reshape(3, 3)[2, 2], -1, 1))); maxtilt = max(maxtilt, tl)
        if fell is None and tl > 45: fell = now
    return maxtilt, fell
def make_cmds(frames, use_filter):
    rt = Retargeter(start_frames=0, blend_frames=1); bf = BalanceFilter(MODEL) if use_filter else None
    out, info, cost = [], [], []
    for world, vis, xy in frames:
        c = rt.update(world, vis, xy) if world is not None else dict(rt.cmd)
        if bf:
            t = time.perf_counter(); c = bf.update(c); cost.append(time.perf_counter() - t); info.append(dict(bf.info))
        out.append(c)
    return out, info, cost
fps = 20; xy = np.full((33, 2), 0.5)
print(f"{'场景':12s} {'无过滤':>16s} {'有过滤':>16s}   过滤器信息")
for kind in ['抬左膝 70°', '抬右膝 70°', '左腿前迈 30°', '右腿侧抬 40°', '深蹲', '中蹲 40°/20°']:
    ts = np.arange(0, 8, 1 / fps)
    frames = [(lambda p: (p[0], p[1], xy))(scenario(kind, t)) for t in ts]
    res = []
    for f in (False, True):
        cmds, info, cost = make_cmds(frames, f)
        tilt, fell = run(cmds, ts); res.append(f"{tilt:4.0f}° " + (f"{fell:.1f}s倒" if fell else "站稳  "))
    sh = np.array([i['shift'] for i in info]); sw = min(i['swing_w'] for i in info); lw = min(i['legs_w'] for i in info)
    print(f"{kind:12s} {res[0]:>16s} {res[1]:>16s}   最大重心转移 侧{np.degrees(np.abs(sh[:,0]).max()):.0f}° 前后{np.degrees(np.abs(sh[:,1]).max()):.0f}°，抬腿保留{sw:.2f} 全腿保留{lw:.2f} 下蹲保留{min(i['squat_k'] for i in info):.2f}，耗时{1000*np.mean(cost):.1f}ms")
