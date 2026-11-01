"""Virtual spring band for lifting humanoids in MuJoCo (from unitree_mujoco)."""

import mujoco
import numpy as np


class ElasticBand:
    def __init__(self, stiffness=150, damping=200, point=None, length=0.0, active=True):
        self.stiffness = stiffness
        self.damping = damping
        self.point = np.asarray(point if point is not None else [0.0, 0.0, 3.0], dtype=float)
        self.length = length
        self.active = active

    def advance(self, position, velocity):
        delta = self.point - position
        distance = np.linalg.norm(delta)
        if distance < 1e-8:
            return np.zeros(3)
        direction = delta / distance
        speed = np.dot(velocity, direction)
        return (self.stiffness * (distance - self.length) - self.damping * speed) * direction

    def mujoco_key_callback(self, key):
        glfw = mujoco.glfw.glfw
        if key == glfw.KEY_7:
            self.length -= 0.1
        elif key == glfw.KEY_8:
            self.length += 0.1
        elif key == glfw.KEY_9:
            self.active = not self.active
