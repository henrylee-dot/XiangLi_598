"""Adapter around mujoco.viewer.launch_passive for MuJoCoParserClass."""

import mujoco
import mujoco.viewer
import numpy as np


class PassiveViewerAdapter:
    """Wraps the official MuJoCo Simulate passive viewer with parser-compatible API."""

    def __init__(
        self,
        model,
        data,
        width=1400,
        height=1000,
        hide_menus=False,
        fontscale=mujoco.mjtFontScale.mjFONTSCALE_200.value,
        maxgeom=10000,
        key_callback=None,
    ):
        self.model = model
        self.data = data
        self._width = width
        self._height = height
        self._markers = []
        self._paused = False
        self._key_pressed = None
        self._is_key_pressed = False
        self._last_mouse_x = 0
        self._last_mouse_y = 0
        self._left_double_click_pressed = False

        def _on_key(keycode):
            self._key_pressed = keycode
            self._is_key_pressed = True
            if key_callback is not None:
                key_callback(keycode)

        self._handle = mujoco.viewer.launch_passive(
            model,
            data,
            key_callback=_on_key,
            show_left_ui=not hide_menus,
            show_right_ui=not hide_menus,
        )

        self._maxgeom = maxgeom
        self._fontscale = fontscale
        self._offscreen_scn = None
        self._offscreen_ctx = None
        self.viewport = mujoco.MjrRect(0, 0, width, height)

    def _ensure_offscreen(self):
        if self._offscreen_ctx is None:
            self._offscreen_scn = mujoco.MjvScene(self.model, maxgeom=self._maxgeom)
            self._offscreen_ctx = mujoco.MjrContext(self.model, self._fontscale)

    @property
    def cam(self):
        return self._handle.cam

    @property
    def vopt(self):
        return self._handle.opt

    @property
    def opt(self):
        return self._handle.opt

    @property
    def pert(self):
        return self._handle.perturb

    @property
    def scn(self):
        self._ensure_offscreen()
        return self._offscreen_scn

    @property
    def ctx(self):
        self._ensure_offscreen()
        return self._offscreen_ctx

    @property
    def is_alive(self):
        return self._handle.is_running()

    def add_marker(self, **marker_params):
        self._markers.append(marker_params)

    def _add_marker_to_scene(self, scn, marker):
        if scn.ngeom >= scn.maxgeom:
            raise RuntimeError('Ran out of geoms. maxgeom: %d' % scn.maxgeom)

        g = scn.geoms[scn.ngeom]
        g.dataid = -1
        g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
        g.objid = -1
        g.category = mujoco.mjtCatBit.mjCAT_DECOR
        g.texid = -1
        g.texuniform = 0
        g.texrepeat[0] = 1
        g.texrepeat[1] = 1
        g.emission = 0
        g.specular = 0.5
        g.shininess = 0.5
        g.reflectance = 0
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size[:] = np.ones(3) * 0.1
        g.mat[:] = np.eye(3)
        g.rgba[:] = np.ones(4)

        for key, value in marker.items():
            if isinstance(value, (int, float, mujoco._enums.mjtGeom)):
                setattr(g, key, value)
            elif isinstance(value, (tuple, list, np.ndarray)):
                attr = getattr(g, key)
                attr[:] = np.asarray(value).reshape(attr.shape)
            elif isinstance(value, str):
                assert key == 'label', 'Only label is a string in mjtGeom.'
                if value is None:
                    g.label[0] = 0
                else:
                    g.label = value
            elif hasattr(g, key):
                raise ValueError(
                    'mjtGeom has attr {} but type {} is invalid'.format(
                        key, type(value)))
            else:
                raise ValueError("mjtGeom doesn't have field %s" % key)
        scn.ngeom += 1

    def _apply_markers_to_user_scn(self):
        user_scn = self._handle.user_scn
        if user_scn is None:
            return
        user_scn.ngeom = 0
        for marker in self._markers:
            self._add_marker_to_scene(user_scn, marker)

    def update_offscreen_scene(self):
        self._ensure_offscreen()
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self._handle.opt,
            self._handle.perturb,
            self._handle.cam,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self._offscreen_scn,
        )

    def render(self):
        if not self.is_alive:
            return
        if self._paused:
            return
        self._apply_markers_to_user_scn()
        self._handle.sync()
        self._markers.clear()

    def close(self):
        if self._handle.is_running():
            self._handle.close()
        if self._offscreen_ctx is not None:
            self._offscreen_ctx.free()
