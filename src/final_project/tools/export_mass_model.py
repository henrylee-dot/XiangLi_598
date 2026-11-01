"""从 MuJoCo 模型导出算重心用的参数，写成 JSON 给 balance_filter.py 用。

模仿节点跑在系统 python 里，没有 mujoco，所以把需要的量提前导出：
每个 body 的父节点、相对父节点的位置/姿态、质量、质心、铰链关节轴，
以及两只脚的脚底接触点（脚部碰撞网格最低处的顶点，在脚的 body 坐标系下）。

用法（仿真 venv 的 python）:
    robotis_mini_mujoco/.venv/bin/python3 export_mass_model.py <mini_scene.xml> <输出.json>
"""
import json
import sys

import mujoco
import numpy as np

SOLE_TOLERANCE = 0.001  # 离最低点 1 mm 以内的顶点算脚底


def main():
    m = mujoco.MjModel.from_xml_path(sys.argv[1])
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)  # 全零关节角：腿直、脚底平

    bodies = []
    for b in range(1, m.nbody):
        joint = None
        for j in range(m.njnt):
            if m.jnt_bodyid[j] == b and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                assert np.allclose(m.jnt_pos[j], 0), "joint not at body origin"
                joint = dict(name=m.joint(j).name, axis=m.jnt_axis[j].tolist())
        bodies.append(dict(
            name=m.body(b).name,
            parent=m.body(m.body_parentid[b]).name,
            pos=m.body_pos[b].tolist(),
            quat=m.body_quat[b].tolist(),
            mass=float(m.body_mass[b]),
            ipos=m.body_ipos[b].tolist(),
            joint=joint,
        ))

    soles = {}
    for side in 'lr':
        b = m.body(f'{side}_foot_link').id
        R_b, p_b = d.xmat[b].reshape(3, 3), d.xpos[b]
        pts_world = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != b or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh = m.geom_dataid[g]
            start, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
            verts = m.mesh_vert[start:start + count]
            R_g = d.geom_xmat[g].reshape(3, 3)
            pts_world.append(verts @ R_g.T + d.geom_xpos[g])
        pts_world = np.vstack(pts_world)
        low = pts_world[pts_world[:, 2] < pts_world[:, 2].min() + SOLE_TOLERANCE]
        soles[side] = ((low - p_b) @ R_b).tolist()  # 转到脚的 body 坐标系
        span = low.max(0) - low.min(0)
        print(f"{side} 脚底: {len(low)} 个点, 前后 {span[0]*1000:.1f} mm, 左右 {span[1]*1000:.1f} mm")

    with open(sys.argv[2], 'w') as f:
        json.dump(dict(bodies=bodies, soles=soles,
                       total_mass=float(m.body_subtreemass[1])), f, indent=1)
    print("已导出", sys.argv[2])


if __name__ == '__main__':
    main()
