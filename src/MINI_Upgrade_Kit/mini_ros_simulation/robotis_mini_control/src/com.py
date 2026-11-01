#!/usr/bin/env python3
import rospy
import geometry_msgs.msg
import tf2_ros
import tf2_geometry_msgs as tf_geo
from urdf_parser_py.urdf import URDF
from visualization_msgs.msg import Marker

class MINI_CoM: 
    
    def __init__(self):
        rospy.init_node('mini_com', anonymous=True)
        self.base_link_frame = rospy.get_param("~base_link_frame", "world")

        self.tfBuffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tfBuffer)

        self.init_link_inertial_info()
        self.init_marker()
    
    def init_link_inertial_info(self):
        robot = URDF.from_parameter_server()
        self.link_info = {}
        for link in robot.links:
            inertial = link.inertial
            if inertial is None or inertial.mass <= 0:
                continue
            xyz = (inertial.origin.xyz if inertial.origin is not None else None)
            self.link_info[link.name] = {
                'mass': float(inertial.mass),
                'origin': {'xyz': xyz if xyz is not None else [0.0, 0.0, 0.0]}}
        self.total_mass = sum(info['mass'] for info in self.link_info.values())
        if self.total_mass <= 0:
            raise ValueError("URDF contains no positive link masses")
        rospy.loginfo("Whole-body CoM: %d inertial links, total mass %.6f kg",
                      len(self.link_info), self.total_mass)

    def init_marker(self):
        marker = Marker()
        marker.header.frame_id = self.base_link_frame
        marker.header.stamp = rospy.Time()
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.pose.orientation.w = 1.0
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.scale.x = 0.03
        marker.scale.y = 0.03
        marker.scale.z = 0.03
        self.marker = marker
        self.marker_pub = rospy.Publisher('com', Marker, queue_size=1)

    def transform_CoM(self):
        """Transform every link CoM at one common TF timestamp; never use a partial body."""
        latest = [self.tfBuffer.lookup_transform(self.base_link_frame, link,
                  rospy.Time(0)) for link in self.link_info]
        stamps = [t.header.stamp for t in latest if t.header.stamp != rospy.Time(0)]
        self.com_stamp = min(stamps) if stamps else rospy.Time(0)
        transformed = {}
        for link, info in self.link_info.items():
            trans = self.tfBuffer.lookup_transform(self.base_link_frame, link,
                                                   self.com_stamp)
            point = geometry_msgs.msg.PointStamped()
            point.header.frame_id = link
            point.header.stamp = self.com_stamp
            point.point.x, point.point.y, point.point.z = info['origin']['xyz']
            transformed[link] = tf_geo.do_transform_point(point, trans).point
        self.transformed_com = transformed

    def calculate_CoM(self):
        """c = sum(m_i * (R_i c_i + t_i)) / sum(m_i), in metres."""
        self.com_x, self.com_y, self.com_z = [
            sum(info['mass'] * getattr(self.transformed_com[link], axis)
                for link, info in self.link_info.items()) / self.total_mass
            for axis in ('x', 'y', 'z')]

    def visualize_CoM(self):
        self.marker.header.stamp = self.com_stamp
        self.marker.pose.position.x = self.com_x
        self.marker.pose.position.y = self.com_y
        self.marker.pose.position.z = self.com_z
        self.marker_pub.publish(self.marker)
             
if __name__ == '__main__':
    mini_com = MINI_CoM()
    rate = rospy.Rate(rospy.get_param("~rate", 30))
    rospy.sleep(2.0)
    
    while not rospy.is_shutdown():
        try:
            mini_com.transform_CoM()
            mini_com.calculate_CoM()
            mini_com.visualize_CoM()
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            rospy.logwarn_throttle(2.0, "Waiting for complete CoM TF: %s", exc)

        rate.sleep()
