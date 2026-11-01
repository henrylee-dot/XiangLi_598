#!/usr/bin/env python3
"""Static center-of-mass safety filter for Robotis MINI leg commands.

Sits between the retargeter and the robot. For each target pose it computes the
robot's center of mass (CoM) with the torso upright and both soles flat (the
retargeter keeps soles parallel to the torso), finds which foot carries the
weight, and makes sure the CoM projects inside the support polygon:

1. Weight shift: when one foot is lifted, both legs roll / pitch together
   (soles stay flat) to move the CoM over the stance foot. The shift is rate
   limited, so the robot leans first and lifts afterwards.
2. Swing hold: until the CoM is over the stance foot, the lifted leg is blended
   back toward a foot-on-ground pose.
3. Fallback: if still unsafe, both legs are blended toward DEFAULT_POSE.

Small foot height differences (below lift_enter) are treated as pose
estimation jitter: the feet are leveled and the robot stays in double support,
so noisy leg landmarks do not make it shift its weight back and forth.

Squat depth is capped for legs on the ground: with the torso kept upright, a
deep human-style squat puts the knee far in front of the hip, and the 0.39 Nm
knee servo then cannot straighten the leg again (the robot tips backward while
standing up). Simulation showed thigh flexion up to ~30 deg recovers.

Static only: assumes slow motion. Model data comes from
config/mini_mass_model.json (tools/export_mass_model.py).
"""
import json

import numpy as np
from scipy.spatial import ConvexHull

from pose_retarget import DEFAULT_POSE

# Actuator ctrlrange (robotis_mini.xml)
LIMITS = {n: (-1.5708, 1.5708) for n in DEFAULT_POSE}
LIMITS.update({'r_thigh_joint': (-1.5708, 0.86), 'l_thigh_joint': (-0.86, 1.5708),
               'r_knee_joint': (-0.189, 1.5708), 'l_knee_joint': (-1.5708, 0.189)})
LEG_JOINTS = {s: [f'{s}_{j}_joint' for j in ('hip', 'thigh', 'knee', 'ankle', 'foot')]
              for s in 'lr'}


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def axis_rot(axis, q):
    x, y, z = axis
    c, s = np.cos(q), np.sin(q)
    C = 1 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


class MassModel:
    """Forward kinematics, CoM and sole outlines with the torso fixed upright."""

    def __init__(self, path):
        with open(path) as f:
            data = json.load(f)
        self.bodies = data['bodies']  # MuJoCo order: parents before children
        for b in self.bodies:
            b['R0'] = quat_to_mat(b['quat'])
            b['pos'] = np.array(b['pos'])
            b['ipos'] = np.array(b['ipos'])
        self.total_mass = sum(b['mass'] for b in self.bodies)
        # Sole outline: convex hull of the lowest mesh vertices, in the foot frame
        zero = self.fk({})
        self.soles = {}
        for side in 'lr':
            pts = np.array(data['soles'][side])
            R, p = zero[f'{side}_foot_link']
            world = pts @ R.T + p
            hull = ConvexHull(world[:, :2])
            self.soles[side] = pts[hull.vertices]

    def fk(self, q):
        poses = {'world': (np.eye(3), np.zeros(3))}
        for b in self.bodies:
            if b['name'] == 'torso':
                poses['torso'] = (np.eye(3), np.zeros(3))
                continue
            R_p, p_p = poses[b['parent']]
            R = R_p @ b['R0']
            if b['joint'] is not None:
                R = R @ axis_rot(b['joint']['axis'], q.get(b['joint']['name'], 0.0))
            poses[b['name']] = (R, p_p + R_p @ b['pos'])
        return poses

    def com_and_soles(self, q):
        poses = self.fk(q)
        com = sum(b['mass'] * (poses[b['name']][0] @ b['ipos'] + poses[b['name']][1])
                  for b in self.bodies) / self.total_mass
        soles = {s: pts @ poses[f'{s}_foot_link'][0].T + poses[f'{s}_foot_link'][1]
                 for s, pts in self.soles.items()}
        return com, soles


def polygon_margin(point, pts):
    """Signed distance from point to the convex hull of pts (positive = inside)."""
    hull = ConvexHull(pts)
    # hull.equations: n . x + d <= 0 inside, with unit normals n
    return float(-np.max(hull.equations[:, :2] @ point + hull.equations[:, 2]))


class BalanceFilter:
    def __init__(self, model_path, contact_tol=0.003, lift_enter=0.010, lift_exit=0.005,
                 margin_double=0.010, margin_single=0.008,
                 max_shift=(0.5, 0.3), shift_step=0.03, max_support_flex=0.52):
        self.model = MassModel(model_path)
        self.contact_tol = contact_tol      # foot counts as on the ground below this height
        self.lift_enter = lift_enter        # foot height that starts single support
        self.lift_exit = lift_exit          # foot height that ends single support
        self.margin = {'double': margin_double, 'single': margin_single}
        self.max_shift = np.array(max_shift)  # (roll, pitch) rad
        self.shift_step = shift_step          # rad per update
        self.max_support_flex = max_support_flex  # thigh flexion cap for loaded legs
        self.shift = np.zeros(2)
        self.stance = None  # None = double support, else 'l' / 'r'
        self.info = {}

    # ---------- pose helpers ----------
    @staticmethod
    def _clip(q):
        return {n: float(np.clip(v, *LIMITS[n])) if n in LIMITS else v for n, v in q.items()}

    @classmethod
    def apply_shift(cls, q, shift):
        """Roll both legs by shift[0] and pitch both by shift[1], soles kept flat."""
        roll, pitch = shift
        q = dict(q)
        for s in 'lr':
            hip, thigh, _, ankle, foot = LEG_JOINTS[s]
            q[hip] += roll
            q[foot] += roll
            sign = 1.0 if s == 'r' else -1.0  # thigh/ankle axes are mirrored
            q[thigh] += sign * pitch
            q[ankle] += sign * pitch
        return cls._clip(q)

    @staticmethod
    def mirror_stance(q, stance):
        """q with the swing leg copying the stance leg (mirrored), so both soles
        are at the same height: the 'swing foot back on the ground' pose."""
        swing = 'l' if stance == 'r' else 'r'
        q = dict(q)
        for js, jw in zip(LEG_JOINTS[stance], LEG_JOINTS[swing]):
            roll = js.endswith(('_hip_joint', '_foot_joint'))  # same axis on both sides
            q[jw] = q[js] if roll else -q[js]
        return q

    @staticmethod
    def blend_legs(q, toward, w, sides='lr'):
        """w = 1 keeps q, w = 0 gives `toward` for the given legs."""
        q = dict(q)
        for s in sides:
            for n in LEG_JOINTS[s]:
                q[n] = toward[n] + w * (q[n] - toward[n])
        return q

    def cap_squat(self, q, legs):
        """Scale thigh and knee of the given legs (same factor) so no thigh flexes
        forward more than max_support_flex; ankle re-derived to keep soles flat."""
        flex = {s: (-q[f'{s}_thigh_joint'] if s == 'r' else q[f'{s}_thigh_joint']) for s in legs}
        worst = max(flex.values(), default=0.0)
        if worst <= self.max_support_flex:
            return q, 1.0
        k = self.max_support_flex / worst
        q = dict(q)
        for s in legs:
            hip, thigh, knee, ankle, _ = LEG_JOINTS[s]
            q[thigh] *= k
            q[knee] *= k
            q[ankle] = float(np.clip(q[thigh] + q[knee], *LIMITS[ankle]))
        return q, k

    def level_feet(self, q):
        """Lower the higher foot: blend its leg's pitch joints toward the mirrored
        loaded leg until both soles are at the same height (bisection). The loaded
        leg is left alone, so the start of a real leg lift is simply held down."""
        h = self.analyze(q)['heights']
        low, high = ('l', 'r') if h['l'] < h['r'] else ('r', 'l')
        mirrored = self.mirror_stance(q, low)

        def pitch_blend(u):
            out = dict(q)
            for n in LEG_JOINTS[high][1:4]:  # thigh, knee, ankle
                out[n] = q[n] + u * (mirrored[n] - q[n])
            return out

        lo, hi = 0.0, 1.0
        for _ in range(10):
            mid = 0.5 * (lo + hi)
            hm = self.analyze(pitch_blend(mid))['heights']
            if hm[high] > hm[low]:
                lo = mid
            else:
                hi = mid
        return pitch_blend(hi)

    # ---------- support analysis ----------
    def analyze(self, q):
        com, soles = self.model.com_and_soles(q)
        heights = {s: pts[:, 2].min() for s, pts in soles.items()}
        ground = min(heights.values())
        support = [s for s in 'lr' if heights[s] - ground < self.contact_tol]
        pts = np.vstack([soles[s][:, :2] for s in support])
        mode = 'double' if len(support) == 2 else 'single'
        return dict(com=com, soles=soles, heights=heights, support=support,
                    margin=polygon_margin(com[:2], pts) - self.margin[mode],
                    center=pts.mean(0))

    def _inside_point(self, com_xy, pts, margin):
        """Point nearest the CoM (along the line to the polygon center) that is
        `margin` inside pts. Shifting only that far keeps the roll, and so the
        hip torque, small: the MINI's servos are limited to 0.39 Nm."""
        center = pts.mean(0)
        if polygon_margin(com_xy, pts) >= margin:
            return com_xy
        lo, hi = 0.0, 1.0  # fraction of the way from center to CoM
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            if polygon_margin(center + mid * (com_xy - center), pts) >= margin:
                lo = mid
            else:
                hi = mid
        return center + lo * (com_xy - center)

    def _shift_for(self, q, target_fn):
        """Shift (roll, pitch) that puts the CoM on target_fn(analysis), 3 Newton steps."""
        shift = self.shift.copy()
        h = 0.01
        for _ in range(3):
            a = self.analyze(self.apply_shift(q, shift))
            err = a['com'][:2] - target_fn(a)
            if np.linalg.norm(err) < 1e-4:
                break
            J = np.empty((2, 2))
            for k in range(2):
                d = shift.copy()
                d[k] += h
                ad = self.analyze(self.apply_shift(q, d))
                J[:, k] = ((ad['com'][:2] - target_fn(ad)) - err) / h
            shift = np.clip(shift - np.linalg.lstsq(J, err, rcond=None)[0],
                            -self.max_shift, self.max_shift)
        return shift

    def _largest_safe(self, make_pose, steps=6):
        """Largest w in [0, 1] whose pose make_pose(w) is statically safe (bisection)."""
        if self.analyze(make_pose(1.0))['margin'] >= 0:
            return 1.0
        lo, hi = 0.0, 1.0
        for _ in range(steps):
            mid = 0.5 * (lo + hi)
            if self.analyze(make_pose(mid))['margin'] >= 0:
                lo = mid
            else:
                hi = mid
        return lo

    # ---------- main entry ----------
    def update(self, target):
        """target: joint dict from the retargeter. Returns a statically safe joint dict."""
        target = self._clip(target)
        a = self.analyze(target)
        target, squat_k = self.cap_squat(target, a['support'])
        a = self.analyze(target)
        dz = a['heights']['l'] - a['heights']['r']
        # Single support once a foot is clearly lifted; back to double only once it
        # is down again, so jitter cannot toggle the mode
        if self.stance is None and abs(dz) >= self.lift_enter:
            self.stance = 'r' if dz > 0 else 'l'
        elif self.stance is not None and abs(dz) < self.lift_exit:
            self.stance = None
        if self.stance is None and abs(dz) >= self.contact_tol * 0.1:
            target = self.level_feet(target)  # jitter: put both feet down

        if self.stance is not None:
            stance = self.stance
            goal_margin = self.margin['single'] + 0.002
            want = self._shift_for(target, lambda an: self._inside_point(
                an['com'][:2], an['soles'][stance][:, :2], goal_margin))
        elif self.analyze(self.apply_shift(target, np.zeros(2)))['margin'] >= 0:
            want = np.zeros(2)
        else:
            goal_margin = self.margin['double'] + 0.002
            want = self._shift_for(target, lambda an: self._inside_point(
                an['com'][:2], np.vstack([an['soles'][s][:, :2] for s in an['support']]),
                goal_margin))
        self.shift += np.clip(want - self.shift, -self.shift_step, self.shift_step)

        q = self.apply_shift(target, self.shift)
        swing_w = legs_w = 1.0
        if self.analyze(q)['margin'] < 0 and self.stance is not None:
            # Keep the swing foot down until the CoM has moved over the stance foot
            swing = 'l' if self.stance == 'r' else 'r'
            grounded = self.mirror_stance(q, self.stance)
            swing_w = self._largest_safe(lambda w: self.blend_legs(q, grounded, w, swing))
            q = self.blend_legs(q, grounded, swing_w, swing)
        if self.analyze(q)['margin'] < 0:
            base = self.apply_shift(DEFAULT_POSE, self.shift)
            legs_w = self._largest_safe(lambda w: self.blend_legs(q, base, w))
            q = self.blend_legs(q, base, legs_w)

        final = self.analyze(q)
        self.info = dict(stance=self.stance or 'double', shift=self.shift.copy(),
                         swing_w=swing_w, legs_w=legs_w, squat_k=squat_k,
                         margin=final['margin'],
                         com=final['com'])
        return q
