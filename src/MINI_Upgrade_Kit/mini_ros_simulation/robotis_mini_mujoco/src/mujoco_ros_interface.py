#!/usr/bin/env python3
import math
import threading
import time

import mujoco
import numpy as np
import rospy
import tf.transformations as tft
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose, Twist, WrenchStamped
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, Imu, JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Empty, EmptyResponse
from trajectory_msgs.msg import JointTrajectory

from utils.elastic_band import ElasticBand
from utils.mujoco_parser import MuJoCoParserClass


class MuJoCoRosNode:
    def __init__(self):
        # Robot config (loaded from config/*.yaml via launch file)
        self.model_name = rospy.get_param('~model_name')
        self.base_body_name = rospy.get_param('~base_body_name')
        self.joint_names = rospy.get_param('~joint_names')
        self.joint_states_topic = rospy.get_param('~joint_states_topic')
        self.joint_cmd_topic = rospy.get_param('~joint_cmd_topic')
        self.joint_traj_topic = rospy.get_param(
            '~joint_trajectory_topic', '/robotis_mini/whole_body_controller/command')
        self.floating_joint_name = rospy.get_param('~floating_joint_name')
        offset_rpy = rospy.get_param('~base_link_offset_rpy')
        offset_xyz = rospy.get_param('~base_link_offset_xyz')

        # Simulation parameters
        self.xml_path = rospy.get_param('~model_xml')
        rospy.loginfo("Loading MuJoCo model: %s", self.xml_path)
        self.sim_rate = rospy.get_param('~sim_rate', 500)
        self.realtime = rospy.get_param('~realtime', True)
        self.realtime_factor = rospy.get_param('~realtime_factor', 1.0)
        self.camera_name = rospy.get_param('~camera_name')
        self.camera_ns = rospy.get_param('~camera_ns')
        self.camera_frame_id = rospy.get_param('~camera_frame_id')
        self.camera_rate = rospy.get_param('~camera_rate')
        self.camera_pub_width = rospy.get_param('~camera_pub_width', 0)
        self.camera_pub_height = rospy.get_param('~camera_pub_height', 0)
        self.imu_topic = rospy.get_param('~imu_topic')
        self.imu_frame_id = rospy.get_param('~imu_frame_id')
        self.imu_rate = rospy.get_param('~imu_rate')
        self.imu_quat_sensor = rospy.get_param('~imu_quat_sensor')
        self.imu_gyro_sensor = rospy.get_param('~imu_gyro_sensor')
        self.imu_accel_sensor = rospy.get_param('~imu_accel_sensor')
        self.ft_rate = rospy.get_param('~ft_rate')
        self.lf_ft_topic = rospy.get_param('~lf_ft_topic')
        self.rf_ft_topic = rospy.get_param('~rf_ft_topic')
        self.lf_ft_frame_id = rospy.get_param('~lf_ft_frame_id')
        self.rf_ft_frame_id = rospy.get_param('~rf_ft_frame_id')
        self.lf_ft_force_sensor = rospy.get_param('~lf_ft_force_sensor')
        self.lf_ft_torque_sensor = rospy.get_param('~lf_ft_torque_sensor')
        self.rf_ft_force_sensor = rospy.get_param('~rf_ft_force_sensor')
        self.rf_ft_torque_sensor = rospy.get_param('~rf_ft_torque_sensor')
        self.model_states_topic = rospy.get_param('~model_states_topic', '/gazebo/model_states')
        self.pd_control_enabled = rospy.get_param('~pd_control_enabled', False)
        self.pd_kp_default = rospy.get_param('~pd_kp', 100.0)
        self.pd_kd_default = rospy.get_param('~pd_kd', 2.0)
        self.pd_gains = rospy.get_param('~pd_gains', {})
        print(f"joint names: {self.joint_names}\n")
        # MuJoCo (parser based on your code)
        self.sim = MuJoCoParserClass('MuJoCoSim', rel_xml_path=self.xml_path, verbose=True)
        if self.base_body_name not in self.sim.body_names:
            rospy.logfatal(
                "Base body '%s' not found in model (available: %s)",
                self.base_body_name, ', '.join(self.sim.body_names))
            raise rospy.ROSInitException(f"unknown base body: {self.base_body_name}")

        # Optional MuJoCo body -> URDF base_link offset (identity when they match).
        self._T_base_body = tft.quaternion_matrix(tft.quaternion_from_euler(*offset_rpy))
        self._T_base_body[:3, 3] = offset_xyz
        self._T_body_base = np.linalg.inv(self._T_base_body)
        self.last_cmd = np.zeros(self.sim.n_ctrl)
        # Active JointTrajectory: dict(t0, times, positions) or None.
        self._traj = None
        self._ctrl_idxs = self.sim.get_idxs_step(self.sim.ctrl_qpos_names)
        self.sim.reset(step=False)
        self._apply_initial_configuration()
        self._setup_pd_control()
        # Viewer Backspace calls mj_resetData, which zeros data.time. Keep the
        # last time so /clock stays monotonic after that reset.
        self._last_sim_time = float(self.sim.data.time)
        self._pending_viewer_reset = False

        # Physics substeps per control loop so sim time tracks wall clock.
        sim_dt = self.sim.dt
        loop_dt = 1.0 / self.sim_rate
        self.nstep = max(1, int(round(loop_dt / sim_dt)))
        rospy.loginfo(
            "MuJoCo dt=%.4fs, loop=%.1fHz, nstep=%d -> %.3fs sim per loop",
            sim_dt, self.sim_rate, self.nstep, self.nstep * sim_dt,
        )

        # ROS publishers/subscribers
        self.joint_pub = rospy.Publisher(self.joint_states_topic, JointState, queue_size=1)
        self.joint_cmd_sub = rospy.Subscriber(
            self.joint_cmd_topic, Float64MultiArray, self.joint_cmd_callback)
        self.joint_traj_sub = rospy.Subscriber(
            self.joint_traj_topic, JointTrajectory, self.joint_traj_callback, queue_size=10)
        self.image_pub = rospy.Publisher(f'{self.camera_ns}/image_raw', Image, queue_size=1)
        self.camera_info_pub = rospy.Publisher(f'{self.camera_ns}/camera_info', CameraInfo, queue_size=1)
        self.imu_pub = rospy.Publisher(self.imu_topic, Imu, queue_size=1)
        self.lf_ft_pub = rospy.Publisher(self.lf_ft_topic, WrenchStamped, queue_size=1)
        self.rf_ft_pub = rospy.Publisher(self.rf_ft_topic, WrenchStamped, queue_size=1)
        self.model_states_pub = rospy.Publisher(
            self.model_states_topic, ModelStates, queue_size=1)
        self.use_sim_time = rospy.get_param('/use_sim_time', False)
        self.clock_pub = None
        if self.use_sim_time:
            self.clock_pub = rospy.Publisher('/clock', Clock, queue_size=10)
            rospy.loginfo("Publishing /clock from MuJoCo sim time")

        # # ROS services
        self.reset_srv = rospy.Service('reset_sim', Empty, self.reset_cb)

        # Internal command buffer
        self._camera_renderer = None
        self._camera_render_data = None
        self._camera_rgb = None
        self._camera_info_msg = None
        self._camera_busy = threading.Lock()
        self._last_camera_pub = rospy.Time(0)
        self._imu_enabled = False
        self._last_imu_pub = rospy.Time(0)
        self._ft_enabled = False
        self._last_ft_pub = rospy.Time(0)
        self._sim_lock = threading.Lock()
        self._setup_camera()
        self._setup_imu()
        self._setup_ft_sensors()
        self._setup_elastic_band()
        rospy.loginfo(
            "Gazebo model_states shim enabled: %s on %s (body=%s)",
            self.model_name, self.model_states_topic, self.base_body_name,
        )

        self.sim.init_viewer(
            azimuth=rospy.get_param('~viewer_azimuth'),
            distance=rospy.get_param('~viewer_distance'),
            elevation=rospy.get_param('~viewer_elevation'),
            lookat=rospy.get_param('~viewer_lookat'),
            key_callback=self._viewer_key_callback,
        )
        self._start_sensor_pub_thread()
        self._start_camera_pub_thread()

    def _resolve_initial_joint_positions(self):
        if rospy.has_param('~initial_joint_positions'):
            return rospy.get_param('~initial_joint_positions')
        return {name: 0.0 for name in self.joint_names}

    def _apply_initial_configuration(self):
        joint_pos = self._resolve_initial_joint_positions()
        self.sim.set_qpos_joints(
            self.joint_names, [joint_pos[name] for name in self.joint_names])
        self.sim.data.qvel[:] = 0.0
        mujoco.mj_forward(self.sim.model, self.sim.data)
        self.last_cmd = np.array([joint_pos[name] for name in self.sim.ctrl_qpos_names])

    def _setup_pd_control(self):
        self._pd_kp = np.full(self.sim.n_ctrl, self.pd_kp_default)
        self._pd_kd = np.full(self.sim.n_ctrl, self.pd_kd_default)
        for idx, joint_name in enumerate(self.sim.ctrl_qpos_names):
            gains = self.pd_gains.get(joint_name, {})
            if 'kp' in gains:
                self._pd_kp[idx] = gains['kp']
            if 'kd' in gains:
                self._pd_kd[idx] = gains['kd']

        if self.pd_control_enabled:
            rospy.loginfo(
                "PD control enabled for %d joints (default kp=%.1f, kd=%.3f)",
                self.sim.n_ctrl, self.pd_kp_default, self.pd_kd_default,
            )
        else:
            rospy.loginfo("PD control disabled; joint commands applied directly as ctrl")

    def _compute_joint_ctrl(self):
        if not self.pd_control_enabled:
            return self.last_cmd

        qpos = self.sim.get_qpos_joints(self.sim.ctrl_qpos_names)
        qvel = self.sim.get_qvel_joints(self.sim.ctrl_qpos_names)
        tau = self._pd_kp * (self.last_cmd - qpos) - self._pd_kd * qvel
        return np.clip(tau, self.sim.ctrl_mins, self.sim.ctrl_maxs)

    def joint_cmd_callback(self, msg):
        if len(msg.data) != self.sim.n_ctrl:
            return

        cmd = np.array(msg.data)
        if not np.all(np.isfinite(cmd)):
            # MuJoCo discards the whole ctrl vector if any entry is non-finite,
            # which would servo every joint to 0. Hold the last good command.
            bad = [self.sim.ctrl_qpos_names[i] for i in np.flatnonzero(~np.isfinite(cmd))]
            rospy.logwarn_throttle(
                1.0, "Ignoring non-finite joint command for: %s", ', '.join(bad))
            return

        with self._sim_lock:
            self._traj = None
            self.last_cmd = cmd

    def joint_traj_callback(self, msg):
        """Execute a JointTrajectory like ros_control's joint_trajectory_controller.

        Waypoints are linearly interpolated from the current command, starting at
        header.stamp. A zero or past stamp starts the trajectory on receipt instead
        of skipping the elapsed part (senders often stamp before a long IK loop).
        A new trajectory preempts the active one.
        """
        if not msg.points:
            return

        ctrl_names = list(self.sim.ctrl_qpos_names)
        unknown = [name for name in msg.joint_names if name not in ctrl_names]
        if unknown:
            rospy.logwarn("Ignoring trajectory with unknown joints: %s", ', '.join(unknown))
            return
        cols = [ctrl_names.index(name) for name in msg.joint_names]

        times = np.empty(len(msg.points))
        pts = np.empty((len(msg.points), len(cols)))
        for i, point in enumerate(msg.points):
            if len(point.positions) != len(cols):
                rospy.logwarn(
                    "Ignoring trajectory: point %d has %d positions for %d joints",
                    i, len(point.positions), len(cols))
                return
            times[i] = point.time_from_start.to_sec()
            pts[i] = point.positions
        if not (np.all(np.isfinite(pts)) and np.all(np.diff(times) >= 0) and times[0] >= 0):
            rospy.logwarn("Ignoring trajectory with non-finite positions or unsorted times")
            return

        with self._sim_lock:
            now = self._now().to_sec()
            t0 = max(msg.header.stamp.to_sec(), now)
            # Start from the current command so motion ramps from where the robot is;
            # joints not named in the message hold their current command.
            positions = np.tile(self.last_cmd, (len(times) + 1, 1))
            positions[1:, cols] = pts
            self._traj = {
                't0': t0,
                'times': np.concatenate(([0.0], times)),
                'positions': positions,
            }

    def _update_traj_cmd(self):
        """Set last_cmd from the active trajectory at the current sim/wall time."""
        if self._traj is None:
            return

        traj = self._traj
        t = self._now().to_sec() - traj['t0']
        times = traj['times']
        positions = traj['positions']
        if t >= times[-1]:
            self.last_cmd = positions[-1].copy()
            self._traj = None
            return
        if t <= times[0]:
            return

        k = int(np.searchsorted(times, t, side='right'))
        t_a, t_b = times[k - 1], times[k]
        alpha = (t - t_a) / (t_b - t_a) if t_b > t_a else 1.0
        self.last_cmd = positions[k - 1] + alpha * (positions[k] - positions[k - 1])

    def reset_cb(self, req):
        with self._sim_lock:
            self.sim.reset(step=False)
            self._on_sim_reset()
        return EmptyResponse()

    def _viewer_key_callback(self, key):
        glfw = mujoco.glfw.glfw
        if key == glfw.KEY_BACKSPACE:
            self._pending_viewer_reset = True
        if self.elastic_band is not None:
            self.elastic_band.mujoco_key_callback(key)

    def _on_sim_reset(self):
        """Restore pose, last joint command, and monotonic /clock after mj_resetData."""
        self._apply_initial_configuration()
        self._traj = None
        t = float(self.sim.data.time)
        if t < self._last_sim_time:
            self.sim.data.time = self._last_sim_time
        else:
            self._last_sim_time = t

    def _restore_sim_time(self):
        """Undo data.time = 0 from mj_resetData so /clock never jumps backward.

        Viewer Backspace also zeros pose/ctrl; restore the initial configuration
        and last_cmd so leftover walking commands are not applied after reset.
        """
        t = float(self.sim.data.time)
        if t < self._last_sim_time or self._pending_viewer_reset:
            self._pending_viewer_reset = False
            self._on_sim_reset()
        else:
            self._last_sim_time = t

    def _sim_time(self):
        self._restore_sim_time()
        return rospy.Time.from_sec(self.sim.data.time)

    def _now(self):
        if self.use_sim_time:
            return self._sim_time()
        return rospy.Time.now()

    def _publish_clock(self):
        if self.clock_pub is None:
            return
        msg = Clock()
        msg.clock = self._sim_time()
        self.clock_pub.publish(msg)

    def _setup_camera(self):
        cam_id = mujoco.mj_name2id(
            self.sim.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        if cam_id < 0:
            rospy.logwarn("Camera '%s' not found in model; camera publishing disabled", self.camera_name)
            return

        resolution = self.sim.model.cam_resolution[cam_id]
        model_width = int(resolution[0])
        model_height = int(resolution[1])
        self._img_width = int(self.camera_pub_width) or model_width
        self._img_height = int(self.camera_pub_height) or model_height
        self._camera_render_data = mujoco.MjData(self.sim.model)
        self._camera_renderer = mujoco.Renderer(
            self.sim.model, height=self._img_height, width=self._img_width)
        self._camera_rgb = np.empty(
            (self._img_height, self._img_width, 3), dtype=np.uint8)
        self._camera_info_msg = self._make_camera_info(cam_id)
        if (self._img_width, self._img_height) != (model_width, model_height):
            rospy.loginfo(
                "Camera publish resolution %dx%d (model camera %dx%d)",
                self._img_width, self._img_height, model_width, model_height,
            )
        rospy.loginfo(
            "Camera publisher enabled: %s (%dx%d @ %.1f Hz, frame=%s)",
            self.camera_ns, self._img_width, self._img_height,
            self.camera_rate, self.camera_frame_id,
        )

    def _setup_imu(self):
        missing = [
            name for name in (
                self.imu_quat_sensor, self.imu_gyro_sensor, self.imu_accel_sensor)
            if name not in self.sim.sensor_names
        ]
        if missing:
            rospy.logwarn(
                "IMU sensors not found (%s); IMU publishing disabled", ', '.join(missing))
            return

        self._imu_enabled = True
        rospy.loginfo(
            "IMU publisher enabled: %s @ %.1f Hz, frame=%s",
            self.imu_topic, self.imu_rate, self.imu_frame_id,
        )

    def _setup_ft_sensors(self):
        required = (
            self.lf_ft_force_sensor, self.lf_ft_torque_sensor,
            self.rf_ft_force_sensor, self.rf_ft_torque_sensor,
        )
        missing = [name for name in required if name not in self.sim.sensor_names]
        if missing:
            rospy.logwarn(
                "FT sensors not found (%s); FT publishing disabled", ', '.join(missing))
            return

        self._ft_enabled = True
        rospy.loginfo(
            "FT publishers enabled: %s, %s @ %.1f Hz",
            self.lf_ft_topic, self.rf_ft_topic, self.ft_rate,
        )

    def _setup_elastic_band(self):
        self.elastic_band_enabled = rospy.get_param('~elastic_band_enabled', False)
        self.elastic_band = None
        self.band_attached_body_id = None

        if not self.elastic_band_enabled:
            return

        body_name = rospy.get_param('~elastic_band_body_name')
        if body_name not in self.sim.body_names:
            rospy.logfatal(
                "Elastic band body '%s' not found in model (available: %s)",
                body_name, ', '.join(self.sim.body_names))
            raise rospy.ROSInitException(f"unknown elastic band body: {body_name}")

        self.elastic_band = ElasticBand(
            stiffness=rospy.get_param('~elastic_band_stiffness', 150),
            damping=rospy.get_param('~elastic_band_damping', 200),
            point=rospy.get_param('~elastic_band_point', [0.0, 0.0, 3.0]),
            length=rospy.get_param('~elastic_band_length', 0.0),
            active=rospy.get_param('~elastic_band_active', True),
        )
        self.band_attached_body_id = mujoco.mj_name2id(
            self.sim.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        rospy.loginfo(
            "Elastic band enabled on %s (keys 7/8: length, 9: toggle)",
            body_name,
        )

    def _apply_elastic_band(self):
        if self.elastic_band is None:
            return

        if not self.elastic_band.active:
            self.sim.data.xfrc_applied[self.band_attached_body_id, :3] = 0.0
            return

        force = self.elastic_band.advance(
            self.sim.data.qpos[:3].copy(),
            self.sim.data.qvel[:3].copy(),
        )
        self.sim.data.xfrc_applied[self.band_attached_body_id, :3] = force

    def _plot_elastic_band(self):
        if self.elastic_band is None:
            return

        p_body = np.asarray(self.sim.data.xpos[self.band_attached_body_id], dtype=float)
        p_anchor = np.asarray(self.elastic_band.point, dtype=float)
        if np.linalg.norm(p_body - p_anchor) < 1e-8:
            return

        if self.elastic_band.active:
            rgba = [0.15, 0.85, 0.25, 0.95]
        else:
            rgba = [0.55, 0.55, 0.55, 0.35]
        self.sim.plot_cylinder_fr2to(p_fr=p_anchor, p_to=p_body, r=0.008, rgba=rgba)
        self.sim.plot_sphere(p=p_anchor, r=0.025, rgba=rgba)

    def _make_camera_info(self, cam_id):
        fovy_deg = float(self.sim.model.cam_fovy[cam_id])
        width = self._img_width
        height = self._img_height
        v_fov = math.radians(fovy_deg)
        h_fov = 2.0 * math.atan(math.tan(v_fov / 2.0) * width / height)
        fx = 0.5 * width / math.tan(h_fov / 2.0)
        fy = 0.5 * height / math.tan(v_fov / 2.0)
        cx = width / 2.0
        cy = height / 2.0

        msg = CameraInfo()
        msg.width = width
        msg.height = height
        msg.distortion_model = 'plumb_bob'
        msg.D = [0.0, 0.0, 0.0, 0.0, 0.0]
        msg.K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        msg.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.P = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return msg

    def _snapshot_for_camera(self):
        """Copy sim state under a brief lock so rendering can run in parallel."""
        with self._sim_lock:
            stamp = self._now()
            self._camera_render_data.qpos[:] = self.sim.data.qpos
            if self.sim.model.nmocap > 0:
                self._camera_render_data.mocap_pos[:] = self.sim.data.mocap_pos
                self._camera_render_data.mocap_quat[:] = self.sim.data.mocap_quat
            self._camera_render_data.time = self.sim.data.time
        mujoco.mj_forward(self.sim.model, self._camera_render_data)
        return stamp

    def publish_camera(self):
        if self._camera_renderer is None:
            return
        if not self._camera_busy.acquire(blocking=False):
            return

        try:
            stamp = self._snapshot_for_camera()
            self._camera_renderer.update_scene(
                self._camera_render_data, camera=self.camera_name)
            rgb = self._camera_renderer.render(out=self._camera_rgb)

            img_msg = Image()
            img_msg.header.stamp = stamp
            img_msg.header.frame_id = self.camera_frame_id
            img_msg.height = self._img_height
            img_msg.width = self._img_width
            img_msg.encoding = 'rgb8'
            img_msg.is_bigendian = 0
            img_msg.step = self._img_width * 3
            img_msg.data = rgb.tobytes()
            self.image_pub.publish(img_msg)

            info_msg = self._camera_info_msg
            info_msg.header.stamp = stamp
            info_msg.header.frame_id = self.camera_frame_id
            self.camera_info_pub.publish(info_msg)
        finally:
            self._camera_busy.release()

    def publish_imu(self):
        if not self._imu_enabled:
            return

        with self._sim_lock:
            quat = self.sim.get_sensor_value(self.imu_quat_sensor)
            gyro = self.sim.get_sensor_value(self.imu_gyro_sensor)
            accel = self.sim.get_sensor_value(self.imu_accel_sensor)
            stamp = self._now()

        msg = Imu()
        msg.header.stamp = stamp
        msg.header.frame_id = self.imu_frame_id
        msg.orientation.w = quat[0]
        msg.orientation.x = quat[1]
        msg.orientation.y = quat[2]
        msg.orientation.z = quat[3]
        msg.angular_velocity.x = gyro[0]
        msg.angular_velocity.y = gyro[1]
        msg.angular_velocity.z = gyro[2]
        msg.linear_acceleration.x = accel[0]
        msg.linear_acceleration.y = accel[1]
        msg.linear_acceleration.z = accel[2]
        msg.orientation_covariance[0] = 0.001
        msg.orientation_covariance[4] = 0.001
        msg.orientation_covariance[8] = 0.001
        msg.angular_velocity_covariance[0] = 0.005
        msg.angular_velocity_covariance[4] = 0.005
        msg.angular_velocity_covariance[8] = 0.005
        msg.linear_acceleration_covariance[0] = 0.05
        msg.linear_acceleration_covariance[4] = 0.1
        msg.linear_acceleration_covariance[8] = 0.2
        self.imu_pub.publish(msg)

    def _make_wrench_stamped(self, frame_id, force_sensor, torque_sensor, stamp):
        force = self.sim.get_sensor_value(force_sensor)
        torque = self.sim.get_sensor_value(torque_sensor)

        msg = WrenchStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.wrench.force.x = force[0]
        msg.wrench.force.y = force[1]
        msg.wrench.force.z = force[2]
        msg.wrench.torque.x = torque[0]
        msg.wrench.torque.y = torque[1]
        msg.wrench.torque.z = torque[2]
        return msg

    def publish_ft_sensors(self):
        if not self._ft_enabled:
            return

        with self._sim_lock:
            stamp = self._now()
            lf_msg = self._make_wrench_stamped(
                self.lf_ft_frame_id, self.lf_ft_force_sensor,
                self.lf_ft_torque_sensor, stamp)
            rf_msg = self._make_wrench_stamped(
                self.rf_ft_frame_id, self.rf_ft_force_sensor,
                self.rf_ft_torque_sensor, stamp)

        self.lf_ft_pub.publish(lf_msg)
        self.rf_ft_pub.publish(rf_msg)

    def _start_sensor_pub_thread(self):
        self._sensor_pub_thread = threading.Thread(
            target=self._sensor_pub_loop,
            name='mujoco_sensor_pub',
            daemon=True,
        )
        self._sensor_pub_thread.start()

    def _sensor_pub_loop(self):
        tick_rate = max(self.imu_rate, self.ft_rate)
        tick_dt = 1.0 / tick_rate
        next_tick = time.monotonic()
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            if (now - self._last_imu_pub).to_sec() >= 1.0 / self.imu_rate:
                self.publish_imu()
                self._last_imu_pub = now
            if (now - self._last_ft_pub).to_sec() >= 1.0 / self.ft_rate:
                self.publish_ft_sensors()
                self._last_ft_pub = now

            next_tick += tick_dt
            sleep_dt = next_tick - time.monotonic()
            if sleep_dt > 0:
                time.sleep(sleep_dt)
            elif sleep_dt < -tick_dt:
                next_tick = time.monotonic()

    def _start_camera_pub_thread(self):
        if self._camera_renderer is None:
            return
        self._camera_pub_thread = threading.Thread(
            target=self._camera_pub_loop,
            name='mujoco_camera_pub',
            daemon=True,
        )
        self._camera_pub_thread.start()

    def _camera_pub_loop(self):
        tick_dt = 1.0 / self.camera_rate
        next_tick = time.monotonic()
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            if (now - self._last_camera_pub).to_sec() >= tick_dt:
                self.publish_camera()
                self._last_camera_pub = now

            next_tick += tick_dt
            sleep_dt = next_tick - time.monotonic()
            if sleep_dt > 0:
                time.sleep(sleep_dt)
            elif sleep_dt < -tick_dt:
                next_tick = time.monotonic()

    def _get_base_link_pose(self):
        """World pose of base_link from the MuJoCo floating-base body."""
        p_body = self.sim.get_p_body(self.base_body_name)
        R_body = self.sim.get_R_body(self.base_body_name)
        T_world_body = np.eye(4)
        T_world_body[:3, :3] = R_body
        T_world_body[:3, 3] = p_body
        T_world_base = T_world_body @ self._T_body_base
        position = T_world_base[:3, 3]
        quat_xyzw = tft.quaternion_from_matrix(T_world_base)
        return position, quat_xyzw

    def _get_base_link_twist(self):
        if self.floating_joint_name not in self.sim.joint_names:
            return np.zeros(3), np.zeros(3)

        qvel = np.asarray(self.sim.get_qvel_joint(self.floating_joint_name)).reshape(-1)
        if len(qvel) < 6:
            return np.zeros(3), np.zeros(3)
        return qvel[:3], qvel[3:6]

    def publish_model_states(self):
        """Publish Gazebo-compatible model pose for mini_ros/scripts/odom.py."""
        with self._sim_lock:
            position, quat_xyzw = self._get_base_link_pose()
            linear, angular = self._get_base_link_twist()

        pose = Pose()
        pose.position.x = position[0]
        pose.position.y = position[1]
        pose.position.z = position[2]
        pose.orientation.x = quat_xyzw[0]
        pose.orientation.y = quat_xyzw[1]
        pose.orientation.z = quat_xyzw[2]
        pose.orientation.w = quat_xyzw[3]

        twist = Twist()
        twist.linear.x = linear[0]
        twist.linear.y = linear[1]
        twist.linear.z = linear[2]
        twist.angular.x = angular[0]
        twist.angular.y = angular[1]
        twist.angular.z = angular[2]

        msg = ModelStates()
        msg.name = [self.model_name]
        msg.pose = [pose]
        msg.twist = [twist]
        self.model_states_pub.publish(msg)

    def step(self):
        with self._sim_lock:
            self._restore_sim_time()
            for _ in range(self.nstep):
                self._apply_elastic_band()
                self._update_traj_cmd()
                ctrl = self._compute_joint_ctrl()
                self.sim.step(ctrl=ctrl, ctrl_idxs=self._ctrl_idxs, nstep=1)
            stamp = self._now()
            js_names = list(self.sim.ctrl_qpos_names)
            js_position = self.sim.get_qpos_joints(self.sim.ctrl_qpos_names).tolist()
            js_velocity = self.sim.get_qvel_joints(self.sim.ctrl_qpos_names).tolist()
            self._publish_clock()
        js_msg = JointState()
        js_msg.header.stamp = stamp
        js_msg.name = js_names
        js_msg.position = js_position
        js_msg.velocity = js_velocity
        self.joint_pub.publish(js_msg)
        self.publish_model_states()

    def run(self):
        loop_dt = (1.0 / self.sim_rate) / self.realtime_factor
        # Wall-clock pacing: rospy.sleep() uses sim time when /use_sim_time is set,
        # which deadlocks here because this node is the /clock source.
        next_tick = time.monotonic()
        while not rospy.is_shutdown() and self.sim.is_viewer_alive():
            self.step()
            with self._sim_lock:
                self._plot_elastic_band()
                self.sim.render()
            if self.realtime:
                next_tick += loop_dt
                sleep_dt = next_tick - time.monotonic()
                if sleep_dt > 0:
                    time.sleep(sleep_dt)
                elif sleep_dt < -loop_dt:
                    next_tick = time.monotonic()

if __name__ == "__main__":
    rospy.init_node('mujoco_ros_node')
    node = MuJoCoRosNode()
    node.run()