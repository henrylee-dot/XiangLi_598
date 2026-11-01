#!/usr/bin/env python3
"""Real Robotis MINI bridge: joint commands -> 16 XL-320 servos through the U2D2.

Subscribes to the same topic the MuJoCo simulator listens on
(/robotis_mini/whole_body_controller/command, JointTrajectory; the last point of
each message is the target), so pose_imitation.py drives the real robot unchanged.
Publishes measured joint angles on /robotis_mini/joint_states.

Safety:
- start: goal = present position before torque on, then a slow move to DEFAULT_POSE
- every joint clipped to the simulator's range and rate limited (~max_joint_speed)
- battery voltage / servo temperature checked every second; torque off on fault
  (voltage: median over servos, low for low_voltage_secs in a row, because the
  bus sags well below the resting voltage while all servos push at once)
- Ctrl+C (or any exit) turns torque off: hold the robot before stopping

Angles follow the simulator's joint conventions; config/real_robot_calibration.yaml
maps them to servo ticks (id, sign, center).
"""
import os
import threading
import time

import numpy as np
import rospy
import yaml
from dynamixel_sdk import (COMM_SUCCESS, GroupSyncRead, GroupSyncWrite, PacketHandler,
                           PortHandler)
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory

from balance_filter import LIMITS
from pose_retarget import DEFAULT_POSE, JOINT_NAMES

# XL-320 control table (Protocol 2.0)
ADDR_TORQUE_ENABLE = 24
ADDR_LED = 25
ADDR_GOAL_POSITION = 30
ADDR_MOVING_SPEED = 32
ADDR_PRESENT_POSITION = 37
ADDR_PRESENT_VOLTAGE = 45
ADDR_PRESENT_TEMPERATURE = 46


class RealRobotBridge:
    def __init__(self):
        here = os.path.dirname(os.path.abspath(__file__))
        calib_path = os.path.expanduser(rospy.get_param(
            '~calibration', os.path.join(here, '..', 'config', 'real_robot_calibration.yaml')))
        with open(calib_path) as f:
            self.cal = yaml.safe_load(f)
        self.rate = rospy.get_param('~rate', 50.0)                # servo write rate, Hz
        self.max_speed = rospy.get_param('~max_joint_speed', 2.0)  # rad/s per joint
        self.startup_time = rospy.get_param('~startup_time', 3.0)  # s to reach DEFAULT_POSE
        # V, under load: the bus sits ~0.5-1 V below the resting battery voltage while
        # the servos hold a pose; XL-320 works from 6 V and the pack has its own cutoff
        self.min_voltage = rospy.get_param('~min_voltage', 6.3)
        self.low_voltage_secs = rospy.get_param('~low_voltage_secs', 3)  # consecutive 1 s checks
        self.low_voltage_count = 0
        self.max_temperature = rospy.get_param('~max_temperature', 65)  # degC

        self.names = list(self.cal['joints'])
        self.ids = {n: self.cal['joints'][n]['id'] for n in self.names}
        self.port = PortHandler(self.cal['port'])
        self.packet = PacketHandler(self.cal['protocol'])
        if not self.port.openPort() or not self.port.setBaudRate(self.cal['baudrate']):
            raise rospy.ROSInitException(
                f"cannot open {self.cal['port']} (is the U2D2 attached with usbipd?)")
        self.writer = GroupSyncWrite(self.port, self.packet, ADDR_GOAL_POSITION, 2)
        self.reader = GroupSyncRead(self.port, self.packet, ADDR_PRESENT_POSITION, 2)
        for n in self.names:
            self.reader.addParam(self.ids[n])

        self.lock = threading.Lock()
        self.target = None           # latest commanded angles (dict), None until first command
        self.cmd = None              # angles actually being sent (rate limited)
        self.js_pub = rospy.Publisher('/robotis_mini/joint_states', JointState, queue_size=1)
        rospy.Subscriber('/robotis_mini/whole_body_controller/command', JointTrajectory,
                         self.command_cb, queue_size=1)

    # ---------- conversions ----------
    def to_ticks(self, name, angle):
        j = self.cal['joints'][name]
        ticks = j['center'] + j['sign'] * angle * self.cal['units_per_rad']
        return int(np.clip(round(ticks), 0, 1023))

    def to_angle(self, name, ticks):
        j = self.cal['joints'][name]
        return (ticks - j['center']) / (j['sign'] * self.cal['units_per_rad'])

    # ---------- servo I/O ----------
    def _retry(self, fn, *args, tries=3):
        for _ in range(tries):
            out = fn(self.port, *args)
            if out[-2] == COMM_SUCCESS:
                return out
            time.sleep(0.005)
        raise IOError(f"{fn.__name__}{args} failed: {self.packet.getTxRxResult(out[-2])}")

    def read_positions(self):
        """Present angles of all joints (dict); joints whose read failed are left out."""
        if self.reader.txRxPacket() != COMM_SUCCESS:
            return {}
        out = {}
        for n in self.names:
            i = self.ids[n]
            if self.reader.isAvailable(i, ADDR_PRESENT_POSITION, 2):
                out[n] = self.to_angle(n, self.reader.getData(i, ADDR_PRESENT_POSITION, 2))
        return out

    def write_goals(self, angles):
        self.writer.clearParam()
        for n in self.names:
            t = self.to_ticks(n, angles[n])
            self.writer.addParam(self.ids[n], [t & 0xFF, (t >> 8) & 0xFF])
        self.writer.txPacket()  # sync write has no status reply

    def torque(self, on):
        for n in self.names:
            try:
                self._retry(self.packet.write1ByteTxRx, self.ids[n], ADDR_TORQUE_ENABLE, int(on))
            except IOError as e:
                rospy.logerr("torque %s failed for %s: %s", on, n, e)

    def check_health(self):
        """Lowest voltage and highest temperature over all servos; torque off on fault."""
        volts, temps = [], []
        for n in self.names:
            try:
                volts.append(self._retry(self.packet.read1ByteTxRx, self.ids[n], ADDR_PRESENT_VOLTAGE)[0] / 10)
                temps.append(self._retry(self.packet.read1ByteTxRx, self.ids[n], ADDR_PRESENT_TEMPERATURE)[0])
            except IOError:
                pass
        volt = float(np.median(volts)) if volts else None
        if volt is not None and volt < self.min_voltage:
            self.low_voltage_count += 1
            rospy.logwarn("real_robot_bridge: bus voltage %.1f V (low %d/%d)",
                          volt, self.low_voltage_count, self.low_voltage_secs)
            if self.low_voltage_count >= self.low_voltage_secs:
                raise RuntimeError(f"battery low: {volt:.1f} V < {self.min_voltage} V "
                                   f"for {self.low_voltage_count} s")
        else:
            self.low_voltage_count = 0
        if temps and max(temps) > self.max_temperature:
            hot = self.names[int(np.argmax(temps))]
            raise RuntimeError(f"servo too hot: {hot} {max(temps)} C")
        return volt, (max(temps) if temps else None)

    # ---------- ROS ----------
    def command_cb(self, msg):
        if not msg.points:
            return
        point = msg.points[-1]
        with self.lock:
            target = dict(self.target or DEFAULT_POSE)
            for n, v in zip(msg.joint_names, point.positions):
                if n in LIMITS and np.isfinite(v):
                    target[n] = float(np.clip(v, *LIMITS[n]))
            self.target = target

    def publish_joint_states(self, angles):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = [n for n in JOINT_NAMES if n in angles]
        msg.position = [angles[n] for n in msg.name]
        self.js_pub.publish(msg)

    def startup(self):
        present = self.read_positions()
        missing = [n for n in self.names if n not in present]
        if missing:
            raise IOError(f"cannot read servos: {missing}")
        for n in self.names:
            self._retry(self.packet.write2ByteTxRx, self.ids[n], ADDR_MOVING_SPEED, 0)  # we rate limit ourselves
        self.write_goals(present)  # torque on must not make anything jump
        self.torque(True)
        rospy.loginfo("real_robot_bridge: torque on, moving to the default pose in %.1f s", self.startup_time)
        steps = max(1, int(self.startup_time * self.rate))
        for k in range(1, steps + 1):
            a = k / steps
            self.write_goals({n: present[n] + a * (DEFAULT_POSE[n] - present[n]) for n in self.names})
            time.sleep(1.0 / self.rate)
        self.cmd = dict(DEFAULT_POSE)
        rospy.loginfo("real_robot_bridge: ready, following %s",
                      '/robotis_mini/whole_body_controller/command')

    def run(self):
        self.startup()
        dt = 1.0 / self.rate
        step = self.max_speed * dt
        next_tick = time.monotonic()
        last_health = last_read = 0.0
        while not rospy.is_shutdown():
            with self.lock:
                target = self.target
            if target is not None:
                for n in self.names:
                    self.cmd[n] += float(np.clip(target[n] - self.cmd[n], -step, step))
            self.write_goals(self.cmd)

            now = time.monotonic()
            if now - last_read > 0.1:  # joint states at 10 Hz
                last_read = now
                self.publish_joint_states(self.read_positions())
            if now - last_health > 1.0:
                last_health = now
                volt, temp = self.check_health()
                rospy.loginfo_throttle(10, "real_robot_bridge: battery %.1f V, max servo temp %s C",
                                       volt or float('nan'), temp)

            next_tick += dt
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.monotonic()

    def shutdown(self):
        self.torque(False)
        self.port.closePort()
        rospy.logwarn("real_robot_bridge: torque OFF")


def main():
    rospy.init_node('real_robot_bridge')
    bridge = RealRobotBridge()
    try:
        bridge.run()
    except Exception as e:  # noqa: BLE001 - any failure must end with torque off
        rospy.logerr("real_robot_bridge: %s", e)
    finally:
        bridge.shutdown()


if __name__ == '__main__':
    main()
