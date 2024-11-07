import rospy
import numpy as np
from geometry_msgs.msg import Quaternion
from visualization_msgs.msg import *
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray

from robot_helpers.spatial import Rotation, Transform
from robot_helpers.conversions import grid_to_map_cloud
from robot_helpers.ros.conversions import to_cloud_msg, to_point_msg, to_pose_msg, to_vector3_msg, to_color_msg

cm = lambda s: tuple([float(1 - s), float(s), float(0)])

class Visualizer:
    def __init__(self, base_frame="panda_link0"):
        self.base_frame = base_frame
        self.create_marker_publisher()
        self.create_scene_cloud_publisher()
        self.create_map_cloud_publisher()
        self.create_quality_publisher()

    def create_marker_publisher(self, topic="visualization_marker_array"):
        self.marker_pub = rospy.Publisher(topic, MarkerArray, queue_size=1)

    def create_scene_cloud_publisher(self, topic="scene_cloud"):
        self.scene_cloud_pub = rospy.Publisher(topic, PointCloud2, queue_size=1)

    def create_map_cloud_publisher(self, topic="map_cloud"):
        self.map_cloud_pub = rospy.Publisher(topic, PointCloud2, queue_size=1)

    def create_quality_publisher(self, topic="quality"):
        self.quality_pub = rospy.Publisher(topic, PointCloud2, queue_size=1)

    def clear(self):
        self.clear_markers()
        self.clear_clouds()
        self.clear_quality()

    def clear_markers(self):
        self.draw([Marker(action=Marker.DELETEALL)])

    def clear_clouds(self):
        msg = to_cloud_msg(self.base_frame, np.array([]))
        self.scene_cloud_pub.publish(msg)
        self.map_cloud_pub.publish(msg)

    def clear_quality(self):
        msg = to_cloud_msg(self.base_frame, np.array([]))
        self.quality_pub.publish(msg)

    def clear_grasp(self):
        markers = [Marker(action=Marker.DELETE, ns="grasp", id=i) for i in range(4)]
        self.draw(markers)

    def roi(self, frame, size):
        pose = Transform.identity()
        scale = [size * 0.005, 0.0, 0.0]
        color = [0.5, 0.5, 0.5]
        lines = box_lines(np.full(3, 0), np.full(3, size))
        msg = create_line_list_marker(frame, pose, scale, color, lines, ns="roi")
        self.draw([msg])

    def scene_cloud(self, frame, points):
        msg = to_cloud_msg(frame, points)
        self.scene_cloud_pub.publish(msg)

    def map_cloud(self, frame, points, distances):
        msg = to_cloud_msg(frame, points, distances=distances)
        self.map_cloud_pub.publish(msg)

    def quality(self, frame, voxel_size, grid, threshold=0.9):
        points, values = grid_to_map_cloud(voxel_size, grid, threshold)
        msg = to_cloud_msg(frame, points, intensities=values)
        self.quality_pub.publish(msg)

    def grasp(self, frame, grasp, quality, vmin=0.5, vmax=1.0):
        color = cm((quality - vmin) / (vmax - vmin))
        self.draw(create_grasp_markers(frame, grasp, color, "grasp"))

    def grasps(self, frame, grasps, qualities, vmin=0.5, vmax=1.0):
        markers = []
        for i, (grasp, quality) in enumerate(zip(grasps, qualities)):
            color = cm((quality - vmin) / (vmax - vmin))
            markers.append(create_grasp_marker(frame, grasp, color, "grasps", i))
        self.draw(markers)

    def draw(self, markers):
        self.marker_pub.publish(MarkerArray(markers=markers))

def box_lines(lower, upper):
    x_l, y_l, z_l = lower
    x_u, y_u, z_u = upper
    return [
        ([x_l, y_l, z_l], [x_u, y_l, z_l]),
        ([x_u, y_l, z_l], [x_u, y_u, z_l]),
        ([x_u, y_u, z_l], [x_l, y_u, z_l]),
        ([x_l, y_u, z_l], [x_l, y_l, z_l]),
        ([x_l, y_l, z_u], [x_u, y_l, z_u]),
        ([x_u, y_l, z_u], [x_u, y_u, z_u]),
        ([x_u, y_u, z_u], [x_l, y_u, z_u]),
        ([x_l, y_u, z_u], [x_l, y_l, z_u]),
        ([x_l, y_l, z_l], [x_l, y_l, z_u]),
        ([x_u, y_l, z_l], [x_u, y_l, z_u]),
        ([x_u, y_u, z_l], [x_u, y_u, z_u]),
        ([x_l, y_u, z_l], [x_l, y_u, z_u]),
    ]


def create_grasp_markers(frame, grasp, color, ns, id=0, depth=0.046, radius=0.005):
    # Nicer looking grasp marker drawn with 4 Marker.CYLINDER
    w, d = grasp.width, depth
    pose = grasp.pose * Transform.t_[0.0, -w / 2, d / 2]
    scale = [radius, radius, d]
    left = create_marker(Marker.CYLINDER, frame, pose, scale, color, ns, id)
    pose = grasp.pose * Transform.t_[0.0, w / 2, d / 2]
    scale = [radius, radius, d]
    right = create_marker(Marker.CYLINDER, frame, pose, scale, color, ns, id + 1)
    pose = grasp.pose * Transform.t_[0.0, 0.0, -d / 4]
    scale = [radius, radius, d / 2]
    wrist = create_marker(Marker.CYLINDER, frame, pose, scale, color, ns, id + 2)
    pose = grasp.pose * Transform.from_rotation(Rotation.from_rotvec([np.pi / 2, 0, 0]))
    scale = [radius, radius, w]
    palm = create_marker(Marker.CYLINDER, frame, pose, scale, color, ns, id + 3)
    return [left, right, wrist, palm]


def create_grasp_marker(frame, grasp, color, ns, id=0, depth=0.05, radius=0.005):
    # Faster grasp marker using Marker.LINE_LIST
    pose, w, d, scale = grasp.pose, grasp.width, depth, [radius, 0.0, 0.0]
    points = [[0, -w / 2, d], [0, -w / 2, 0], [0, w / 2, 0], [0, w / 2, d]]
    return create_line_strip_marker(frame, pose, scale, color, points, ns, id)


def create_arrow_marker(frame, start, end, scale, color, ns="", id=0):
    pose = Transform.identity()
    marker = create_marker(Marker.ARROW, frame, pose, scale, color, ns, id)
    marker.points = [to_point_msg(start), to_point_msg(end)]
    return marker


def create_cube_marker(frame, pose, scale, color, ns="", id=0):
    return create_marker(Marker.CUBE, frame, pose, scale, color, ns, id)


def create_line_list_marker(frame, pose, scale, color, lines, ns="", id=0):
    marker = create_marker(Marker.LINE_LIST, frame, pose, scale, color, ns, id)
    marker.points = [to_point_msg(point) for line in lines for point in line]
    return marker


def create_line_strip_marker(frame, pose, scale, color, points, ns="", id=0):
    marker = create_marker(Marker.LINE_STRIP, frame, pose, scale, color, ns, id)
    marker.points = [to_point_msg(point) for point in points]
    return marker


def create_sphere_marker(frame, pose, scale, color, ns="", id=0):
    return create_marker(Marker.SPHERE, frame, pose, scale, color, ns, id)


def create_sphere_list_marker(frame, pose, scale, color, centers, ns="", id=0):
    marker = create_marker(Marker.SPHERE_LIST, frame, pose, scale, color, ns, id)
    marker.points = [to_point_msg(center) for center in centers]
    return marker


def create_mesh_marker(frame, mesh, pose, scale=None, color=None, ns="", id=0):
    marker = create_marker(Marker.MESH_RESOURCE, frame, pose, scale, color, ns, id)
    marker.mesh_resource = mesh
    return marker


def create_marker(type, frame, pose, scale=None, color=None, ns="", id=0):
    if scale is None:
        scale = [1, 1, 1]
    elif np.isscalar(scale):
        scale = [scale, scale, scale]
    if color is None:
        color = (1, 1, 1)
    msg = Marker()
    msg.header.frame_id = frame
    msg.header.stamp = rospy.Time()
    msg.ns = ns
    msg.id = id
    msg.type = type
    msg.action = Marker.ADD
    msg.pose = to_pose_msg(pose)
    msg.scale = to_vector3_msg(scale)
    msg.color = to_color_msg(color)
    return msg


MOVE_AXIS = InteractiveMarkerControl.MOVE_AXIS
ROTATE_AXIS = InteractiveMarkerControl.ROTATE_AXIS


def create_6dof_ctrl(frame, name, pose, scale, markers):
    im = InteractiveMarker()
    im.header.frame_id = frame
    im.name = name
    im.pose = to_pose_msg(pose)
    im.scale = scale
    im.controls = [
        InteractiveMarkerControl(markers=markers, always_visible=True),
        InteractiveMarkerControl(
            orientation=Quaternion(w=1, x=1), interaction_mode=MOVE_AXIS
        ),
        InteractiveMarkerControl(
            orientation=Quaternion(w=1, y=1), interaction_mode=MOVE_AXIS
        ),
        InteractiveMarkerControl(
            orientation=Quaternion(w=1, z=1), interaction_mode=MOVE_AXIS
        ),
        InteractiveMarkerControl(
            orientation=Quaternion(w=1, x=1), interaction_mode=ROTATE_AXIS
        ),
        InteractiveMarkerControl(
            orientation=Quaternion(w=1, y=1), interaction_mode=ROTATE_AXIS
        ),
        InteractiveMarkerControl(
            orientation=Quaternion(w=1, z=1), interaction_mode=ROTATE_AXIS
        ),
    ]
    return im
