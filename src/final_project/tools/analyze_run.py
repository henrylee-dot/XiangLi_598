"""分析一次模仿录制：网络延迟、平滑延迟、关节跟随延迟、站立情况（改前/改后对比）。

用法（仿真 venv 的 python，需要 mujoco）:
    robotis_mini_mujoco/.venv/bin/python3 analyze_run.py <录制目录> <final_project/src> <mini_scene.xml>
先用 extract_bag.py 生成 robot_extracted.npz。
"""
import sys, numpy as np
run, src, xml = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, src)
from pose_retarget import Retargeter, DEFAULT_POSE, JOINT_NAMES
F = np.load(f'{run}/frames.npz', allow_pickle=True)
R = np.load(f'{run}/robot_extracted.npz', allow_pickle=True)
t_read, t_done, det, skipped = F['t_read'], F['t_done'], F['detect_ms'], F['skipped']
state, cmd, world, vis, xy = F['state'], F['cmd'], F['world'], F['vis'], F['image_xy']
T = t_done[-1] - t_read[0]; N = len(t_read)
print("=" * 60); print("一、本次运行概况")
print(f"  记录时长 {T:.1f} s，处理帧数 {N}，处理速度 {N / T:.1f} fps")
cam_frames = int((skipped + 1).sum()); cam_fps = cam_frames / T
print(f"  摄像头送来 {cam_frames} 帧 ({cam_fps:.1f} fps)，丢掉旧帧 {int(skipped.sum())} 帧")
proc = t_done - t_read
print(f"  每帧处理耗时：中位 {1000*np.median(proc):.0f} ms（其中 MediaPipe {np.median(det):.0f} ms），95% {1000*np.percentile(proc,95):.0f} ms")
st = list(state); first = {s: st.index(s) for s in dict.fromkeys(st)}
print("  状态出现时刻:", {s: f"{t_read[i]-t_read[0]:.1f}s" for s, i in first.items()})
tracking = np.array([s == 'tracking' for s in st])
print(f"  正常跟踪帧占比 {100*tracking.mean():.0f}%，无人帧占比 {100*np.mean([s=='no person' for s in st]):.0f}%")

print("=" * 60); print("二、网络传输：画面'年龄'（帧到达 WSL → 开始处理）")
# Same arrivals (camera rate) and same measured service times for both policies
period = 1 / cam_fps; arr = t_read[0] + np.arange(int(T * cam_fps)) * period
service = np.diff(t_read)  # full loop time between frame starts
service = np.clip(service, 0, None)
def simulate(policy):
    t, i, k, ages = arr[0], 0, 0, []
    while True:
        avail = np.searchsorted(arr, t, side='right')  # frames arrived by t
        if policy == 'fifo':
            if i >= len(arr): break
            if i >= avail: t = arr[i]
            ages.append((t, max(0, t - arr[i]))); i += 1
        else:
            if avail <= i:
                if avail >= len(arr): break
                t = arr[avail]; avail += 1
            i = avail; ages.append((t, max(0, t - arr[i - 1])))
        t += service[k % len(service)]; k += 1
        if t > arr[-1]: break
    return np.array(ages)
for name, pol in (('改前（按顺序排队）', 'fifo'), ('改后（只处理最新帧）', 'latest')):
    a = simulate(pol); rel = a[:, 0] - arr[0]
    seg = {f"{int(s)}-{int(s+20)}s": f"{np.mean(a[(rel>=s)&(rel<s+20),1]):.2f}s" for s in range(0, int(rel[-1]), 20) if np.any((rel>=s)&(rel<s+20))}
    print(f"  {name}: 平均 {a[:,1].mean():.2f}s，最后时刻 {a[-1,1]:.2f}s；分段 {seg}")

print("=" * 60); print("三、平滑带来的延迟（用录到的关键点重算，只看正常跟踪段）")
valid = ~np.isnan(vis[:, 0])
def replay(**kw):
    rt = Retargeter(start_frames=0, blend_frames=1, **kw); out = []
    for i in range(N):
        if valid[i]:
            c = rt.update(world[i], vis[i], xy[i])
        else:
            c = dict(rt.cmd)
        out.append([c[n] for n in JOINT_NAMES])
    return np.array(out)
raw = replay(smoothing=0.0, max_step=10.0)
old = replay(smoothing=0.5, max_step=0.15)
new = replay(smoothing=0.3, max_step=0.4)
mask = tracking
def lag(ref, sig, ts):
    # cross-correlation lag on a uniform 100 Hz grid, joints with enough motion
    g = np.arange(ts[0], ts[-1], 0.01); lags = []
    for j in range(ref.shape[1]):
        r = np.interp(g, ts, ref[:, j]); s = np.interp(g, ts, sig[:, j])
        if np.std(r) < 0.05: continue
        r -= r.mean(); s -= s.mean()
        best = max(range(0, 150), key=lambda d: np.dot(r[:len(r)-d], s[d:]))
        lags.append(best * 10)
    return np.median(lags) if lags else float('nan'), len(lags)
seg_t = t_read[mask]
for name, sig in (('改前 (0.5, 0.15 rad/帧)', old), ('改后 (0.3, 0.4 rad/帧)', new)):
    l, n = lag(raw[mask], sig[mask], seg_t)
    err = np.degrees(np.abs(sig[mask] - raw[mask]).mean())
    print(f"  {name}: 延迟中位 {l:.0f} ms（{n} 个有明显动作的关节），平均跟不上 {err:.1f}°")
# check recorded commands match the new filter replay during tracking
print(f"  （校验：实际发出的命令与离线重算差 {np.degrees(np.nanmax(np.abs(cmd[mask]-new[mask]))):.2f}°）" if mask.any() else "")

print("=" * 60); print("四、机器人关节跟随命令的延迟（仿真内部，改前改后一样）")
c_t, c_q, js_t, js_q = R['c_t'], R['c_q'], R['js_t'], R['js_q']
cn, jn = list(R['c_names']), list(R['js_names'])
cmd_frames = np.where(~np.isnan(cmd[:, 0]))[0]
k = min(len(cmd_frames), len(c_t))
A = np.polyfit(c_t[:k], t_done[cmd_frames[:k]], 1)  # bag (sim) time -> wall time
print(f"  仿真时间与真实时间比例 {A[0]:.3f}（1.000 表示仿真跑得和现实一样快）")
cw = np.polyval(A, c_t); jw = np.polyval(A, js_t)
order = [jn.index(n) for n in cn]
g0, g1 = max(cw[0], jw[0]), min(cw[-1], jw[-1])
sel = (jw > g0) & (jw < g1)
cmd_on_js = np.array([np.interp(jw[sel], cw, c_q[:, j]) for j in range(len(cn))]).T
l, n = lag(cmd_on_js, js_q[sel][:, order], jw[sel])
print(f"  关节实际角度落后命令：中位 {l:.0f} ms（{n} 个关节）")

print("=" * 60); print("五、站立情况")
ms_t, ms_q, ms_p = R['ms_t'], R['ms_q'], R['ms_p']
w, x, y, z = ms_q.T; up = 1 - 2 * (x**2 + y**2)
tilt = np.degrees(np.arccos(np.clip(up, -1, 1)))
mw = np.polyval(A, ms_t) - t_read[0]
print(f"  本次实际：躯干最大倾斜 {tilt.max():.1f}°，躯干高度 {ms_p[:,2].min():.3f}~{ms_p[:,2].max():.3f} m，"
      f"倾斜超过 45° 的时间 {np.mean(tilt>45)*100:.0f}%")
if (tilt > 45).any():
    print(f"  首次倾斜超过 45° 在第 {mw[np.argmax(tilt>45)]:.1f} s")

import mujoco
m = mujoco.MjModel.from_xml_path(xml)
act = [m.joint(m.actuator_trnid[a, 0]).name for a in range(m.nu)]
idx = [JOINT_NAMES.index(n) for n in act]
def sim_commands(cmds, ts, label):
    d = mujoco.MjData(m)
    for n, v in DEFAULT_POSE.items(): d.qpos[m.jnt_qposadr[m.joint(n).id]] = v
    cur = np.array([DEFAULT_POSE[n] for n in act]); d.ctrl[:] = cur
    for _ in range(1000): mujoco.mj_step(m, d)          # settle like the real start
    t0 = ts[0]; k = 0; start = cur.copy(); goal = cur.copy(); tg = 0; fell = None; maxtilt = 0
    while d.time - 2.0 < ts[-1] - t0:
        now = d.time - 2.0 + t0
        while k < len(ts) and ts[k] <= now:
            start, goal, tg = d.ctrl.copy(), cmds[k][idx], now; k += 1
        a = min(1, (now - tg) / 0.1); d.ctrl[:] = start + a * (goal - start)
        mujoco.mj_step(m, d)
        tl = np.degrees(np.arccos(np.clip(d.xmat[1].reshape(3, 3)[2, 2], -1, 1))); maxtilt = max(maxtilt, tl)
        if fell is None and tl > 45: fell = now - t0
    print(f"  {label}: 最大倾斜 {maxtilt:.0f}°，" + (f"第 {fell:.1f} s 摔倒" if fell is not None else "全程站稳"))
def replay_gate(gate, **kw):
    rt = Retargeter(**kw) if gate else Retargeter(start_frames=0, blend_frames=1, **kw); out = []
    for i in range(N):
        if valid[i]: c = rt.update(world[i], vis[i], xy[i])
        else: rt.lost(); c = dict(rt.cmd)
        out.append([c[n] for n in JOINT_NAMES])
    return np.array(out)
print("  用录到的动作在离线仿真里重放：")
sim_commands(replay_gate(False, smoothing=0.5, max_step=0.15), t_read, "改前（无启动保护，旧平滑）")
sim_commands(replay_gate(True, smoothing=0.3, max_step=0.4), t_read, "改后（有启动保护，新平滑）")
