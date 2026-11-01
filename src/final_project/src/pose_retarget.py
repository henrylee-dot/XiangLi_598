#!/usr/bin/env python3
"""Retarget MediaPipe pose world landmarks to Robotis MINI joint angles.

Pure numpy/scipy (no ROS) so it can be tested offline.

Robot torso frame: x forward, y left, z up. At all-zero joint angles the arms
point straight out sideways (T-pose), the legs point straight down and the soles
are flat. Joint axes below are the MuJoCo world axes at that zero pose
(robotis_mini.xml), so limb directions follow from chained axis rotations.

Mapping is left-to-left: the human's left arm drives the robot's l_* joints.
"""
import numpy as np
from scipy.optimize import least_squares

X = np.array([1.0, 0.0, 0.0])
Y = np.array([0.0, 1.0, 0.0])
Z = np.array([0.0, 0.0, 1.0])

# MediaPipe pose landmark indices
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

# Per limb: joint names, axes at zero pose, segment direction at zero pose,
# MediaPipe landmarks (root, middle, end), joint limits (actuator ctrlrange).
ARMS = {
    'l': dict(joints=['l_shoulder_joint', 'l_biceps_joint', 'l_elbow_joint'],
              axes=[-Y, -X, -X], dir0=Y,
              landmarks=(L_SHOULDER, L_ELBOW, L_WRIST),
              lower=[-1.5708, -1.5708, -1.5708], upper=[1.5708, 1.5708, 1.5708]),
    'r': dict(joints=['r_shoulder_joint', 'r_biceps_joint', 'r_elbow_joint'],
              axes=[Y, -X, -X], dir0=-Y,
              landmarks=(R_SHOULDER, R_ELBOW, R_WRIST),
              lower=[-1.5708, -1.5708, -1.5708], upper=[1.5708, 1.5708, 1.5708]),
}
LEGS = {
    'l': dict(joints=['l_hip_joint', 'l_thigh_joint', 'l_knee_joint',
                      'l_ankle_joint', 'l_foot_joint'],
              axes=[-X, -Y, -Y], dir0=-Z,
              landmarks=(L_HIP, L_KNEE, L_ANKLE),
              lower=[-1.5708, -0.86, -1.5708], upper=[1.5708, 1.5708, 0.189]),
    'r': dict(joints=['r_hip_joint', 'r_thigh_joint', 'r_knee_joint',
                      'r_ankle_joint', 'r_foot_joint'],
              axes=[-X, Y, Y], dir0=-Z,
              landmarks=(R_HIP, R_KNEE, R_ANKLE),
              lower=[-1.5708, -1.5708, -0.189], upper=[1.5708, 0.86, 1.5708]),
}
LIMB_LIMIT = 1.5708  # ankle / foot ctrlrange

# Robot's default crouched standing pose (robotis_mini_mujoco/config/robotis_mini.yaml)
DEFAULT_POSE = {
    'r_hip_joint': 0.0, 'r_thigh_joint': -0.111, 'r_knee_joint': 1.12,
    'r_ankle_joint': 1.03, 'r_foot_joint': 0.0,
    'l_hip_joint': 0.0, 'l_thigh_joint': 0.111, 'l_knee_joint': -1.12,
    'l_ankle_joint': -1.03, 'l_foot_joint': 0.0,
    'l_shoulder_joint': 0.0, 'l_biceps_joint': 0.501, 'l_elbow_joint': 1.39,
    'r_shoulder_joint': 0.0, 'r_biceps_joint': -0.501, 'r_elbow_joint': -1.39,
}
JOINT_NAMES = list(DEFAULT_POSE)


def rot(axis, q):
    """Rotation matrix about a unit axis (Rodrigues)."""
    x, y, z = axis
    c, s = np.cos(q), np.sin(q)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def limb_fk(limb, q):
    """Unit directions of the first and second segment for 3 chain angles."""
    R1 = rot(limb['axes'][0], q[0]) @ rot(limb['axes'][1], q[1])
    upper = R1 @ limb['dir0']
    lower = R1 @ rot(limb['axes'][2], q[2]) @ limb['dir0']
    return upper, lower


def torso_frame(p):
    """Rotation whose rows are the human torso x(fwd), y(left), z(up) axes.

    p: (33, 3) MediaPipe world landmarks. Their frame (x right, y down,
    z away from camera) is right-handed, so x = y cross z faces forward.
    """
    left = (p[L_SHOULDER] + p[L_HIP]) / 2 - (p[R_SHOULDER] + p[R_HIP]) / 2
    up = (p[L_SHOULDER] + p[R_SHOULDER]) / 2 - (p[L_HIP] + p[R_HIP]) / 2
    z = up / np.linalg.norm(up)
    y = left - left.dot(z) * z
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    return np.vstack([x, y, z])


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def solve_limb(limb, upper, lower, q_prev, lower_weight=0.7, reg=0.05):
    """Chain angles whose segment directions best match the target directions.

    Warm-started from q_prev; a few extra seeds avoid getting stuck on the
    wrong branch (e.g. biceps sign flip when the arm passes horizontal).
    """
    lo = np.array(limb['lower'])
    hi = np.array(limb['upper'])
    q_prev = np.clip(q_prev, lo, hi)

    def residual(q):
        u, l = limb_fk(limb, q)
        return np.concatenate([u - upper, lower_weight * (l - lower), reg * (q - q_prev)])

    def fit(seed):
        return least_squares(residual, np.clip(seed, lo + 1e-6, hi - 1e-6),
                             bounds=(lo, hi), xtol=1e-4, ftol=1e-4)

    best = fit(q_prev)
    if best.cost > 0.02:  # warm start landed badly, try the other branches too
        for seed in (np.zeros(3), 0.5 * (lo + hi) + 0.25 * (hi - lo), 0.5 * (lo + hi) - 0.25 * (hi - lo)):
            sol = fit(seed)
            if sol.cost < best.cost - 1e-6:
                best = sol
    return best.x


class Retargeter:
    """Stateful landmark -> joint angle mapping with per-limb hold and smoothing.

    Start-up gate: the robot holds DEFAULT_POSE until the whole body is inside
    the image and standing (thighs roughly vertical) for start_frames frames,
    then blends from DEFAULT_POSE into the imitation over blend_frames frames.
    This keeps a seated or half-visible person from knocking the robot over.
    """

    WAITING, BLENDING, TRACKING = 'waiting', 'blending', 'tracking'
    BODY = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE)

    def __init__(self, visibility=0.5, smoothing=0.3, max_step=0.4, imitate_legs=True,
                 start_frames=25, blend_frames=25, standing_angle=30.0):
        self.visibility = visibility
        self.smoothing = smoothing    # EMA weight on the previous command
        self.max_step = max_step      # rad per update, limits sudden jumps
        self.imitate_legs = imitate_legs
        self.start_frames = start_frames
        self.blend_frames = blend_frames
        self.standing_cos = np.cos(np.radians(standing_angle))
        self.state = self.WAITING
        self.progress = 0  # frames counted in WAITING / BLENDING
        self.raw = dict(DEFAULT_POSE)  # unsmoothed solution, used as warm start
        self.cmd = dict(DEFAULT_POSE)

    def _visible(self, vis, idxs, image_xy=None):
        """Visibility above threshold and, when given, inside the image.

        MediaPipe extrapolates off-screen landmarks (e.g. legs below a desk)
        with fairly high visibility, so the image bounds check matters for legs.
        """
        for i in idxs:
            if vis[i] < self.visibility:
                return False
            if image_xy is not None:
                x, y = image_xy[i]
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                    return False
        return True

    def _standing(self, world, R):
        for hip, knee in ((L_HIP, L_KNEE), (R_HIP, R_KNEE)):
            thigh = _unit(R @ (world[knee] - world[hip]))
            if -thigh[2] < self.standing_cos:
                return False
        return True

    def lost(self):
        """Call on frames with no person; restarts the start-up countdown."""
        if self.state == self.WAITING:
            self.progress = 0

    def update(self, world, vis, image_xy):
        """world: (33, 3) world landmarks, vis: (33,) visibility,
        image_xy: (33, 2) normalized image coordinates. Returns joint dict."""
        R = torso_frame(world)
        if self.state == self.WAITING:
            ready = self._visible(vis, self.BODY, image_xy) and self._standing(world, R)
            self.progress = self.progress + 1 if ready else 0
            if self.progress >= self.start_frames:
                self.state, self.progress = self.BLENDING, 0
            return dict(self.cmd)

        # Arms use the torso as root; landmarks needed: the limb plus torso corners
        torso_ok = self._visible(vis, (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP))
        if torso_ok:
            for limb in ARMS.values():
                self._solve(limb, world, vis, R, None)
            if self.imitate_legs:
                for limb in LEGS.values():
                    self._solve(limb, world, vis, R, image_xy)
        cmd = self._smooth()

        if self.state == self.BLENDING:
            self.progress += 1
            w = min(1.0, self.progress / self.blend_frames)
            if w >= 1.0:
                self.state = self.TRACKING
            return {n: DEFAULT_POSE[n] + w * (cmd[n] - DEFAULT_POSE[n]) for n in JOINT_NAMES}
        return cmd

    def _solve(self, limb, world, vis, R, image_xy):
        a, b, c = limb['landmarks']
        if not self._visible(vis, (a, b, c), image_xy):
            return  # hold the last solution for this limb
        upper = _unit(R @ (world[b] - world[a]))
        lower = _unit(R @ (world[c] - world[b]))
        names = limb['joints']
        q_prev = np.array([self.raw[n] for n in names[:3]])
        q = solve_limb(limb, upper, lower, q_prev)
        for n, v in zip(names[:3], q):
            self.raw[n] = float(v)
        if len(names) == 5:
            # Keep the sole parallel to the torso: pitch and roll cancel along the chain
            hip, thigh, knee = q
            self.raw[names[3]] = float(np.clip(thigh + knee, -LIMB_LIMIT, LIMB_LIMIT))
            self.raw[names[4]] = float(np.clip(hip, -LIMB_LIMIT, LIMB_LIMIT))

    def _smooth(self):
        for n in JOINT_NAMES:
            target = (1 - self.smoothing) * self.raw[n] + self.smoothing * self.cmd[n]
            step = np.clip(target - self.cmd[n], -self.max_step, self.max_step)
            self.cmd[n] += float(step)
        return dict(self.cmd)
