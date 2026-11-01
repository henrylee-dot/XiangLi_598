"""关节方向校准的"预期动作表"：假设真机方向与仿真一致（位置 = 512 + 角度 × 195.38），
从当前姿势出发每个关节转 ±10°，用 MuJoCo 算出手/脚往哪个方向动，选远离身体的方向。

用法（仿真 venv 的 python）:
    robotis_mini_mujoco/.venv/bin/python3 joint_test_plan.py <mini_scene.xml> <id:位置 ...>
输出每个 ID 的测试角度（+10 或 -10）和预期动作描述。
"""
import sys

import mujoco
import numpy as np

ID_JOINT = {1: 'r_shoulder', 2: 'l_shoulder', 3: 'r_biceps', 4: 'l_biceps', 5: 'r_elbow',
            6: 'l_elbow', 7: 'r_hip', 8: 'l_hip', 9: 'r_thigh', 10: 'l_thigh', 11: 'r_knee',
            12: 'l_knee', 13: 'r_ankle', 14: 'l_ankle', 15: 'r_foot', 16: 'l_foot'}
NAMES = {1: '右肩', 2: '左肩', 3: '右上臂', 4: '左上臂', 5: '右肘', 6: '左肘', 7: '右髋侧摆',
         8: '左髋侧摆', 9: '右大腿', 10: '左大腿', 11: '右膝', 12: '左膝', 13: '右踝',
         14: '左踝', 15: '右脚侧摆', 16: '左脚侧摆'}


def main():
    m = mujoco.MjModel.from_xml_path(sys.argv[1])
    d = mujoco.MjData(m)
    pos = {int(k): int(v) for k, v in (a.split(':') for a in sys.argv[2:])}
    q0 = {i: (p - 512) / 195.3786 for i, p in pos.items()}

    def point(i):
        """被观察的点（躯干坐标系）：手臂看手尖，髋/大腿/膝看脚，踝看脚尖，脚侧摆看脚外侧。"""
        side = ID_JOINT[i][0]
        if i <= 6:
            b = m.body(f'{side}_elbow_link').id
            gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]
            return max((d.geom_xpos[g] for g in gs), key=lambda p: np.linalg.norm(p - d.xpos[b]))
        b = m.body(f'{side}_foot_link').id
        gs = np.array([d.geom_xpos[g] for g in range(m.ngeom) if m.geom_bodyid[g] == b])
        if i <= 12:
            return gs.mean(0)
        if i <= 14:
            return gs[np.argmax(gs[:, 0])]               # 最靠前
        return gs[np.argmax(np.abs(gs[:, 1]))]            # 最靠外

    def fk(q):
        d.qpos[:] = m.qpos0
        for i, v in q.items():
            d.qpos[m.jnt_qposadr[m.joint(ID_JOINT[i] + '_joint').id]] = v
        mujoco.mj_forward(m, d)
        return d.xpos[1].copy()

    print(" ID  部位      测试角度  预期动作（以机器人自己的前后左右为准）")
    for i in sorted(q0):
        best = None
        for sgn in (+1, -1):
            q = dict(q0)
            fk(q)
            p0 = point(i).copy() - d.xpos[1]
            q[i] = q0[i] + sgn * np.radians(10)
            fk(q)
            p1 = point(i).copy() - d.xpos[1]
            away = np.linalg.norm(p1[:2]) - np.linalg.norm(p0[:2])  # 水平方向离身体更远
            if best is None or away > best[0]:
                best = (away, sgn, p1 - p0)
        _, sgn, dp = best
        words = []
        for k, (pos_w, neg_w) in enumerate((('前', '后'), ('左', '右'), ('上', '下'))):
            if abs(dp[k]) > 0.004:
                words.append(f"向{pos_w if dp[k] > 0 else neg_w} {abs(dp[k]) * 100:.1f} cm")
        what = {True: '手', False: '脚'}[i <= 6]
        if 13 <= i <= 14:
            what = '脚尖'
        if i >= 15:
            what = '脚外侧边缘'
        print(f"{i:3d}  {NAMES[i]:8s} {sgn * 10:+4d}°    {what}" + ("、".join(words) or "几乎不动"))


if __name__ == '__main__':
    main()
