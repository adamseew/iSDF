
# Edits for standard ROS1 wrapper by GRAB Lab, Yale University

# Original copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import queue
import numpy as np
from scipy.spatial.transform import Rotation
import rospy
import trimesh
import cv2
import imgviz
# from time import perf_counter

# from orb_slam3_ros_wrapper.msg import frame
from geometry_msgs.msg import Quaternion, Pose
from sensor_msgs.msg import Image

import message_filters


class iSDFNode:

    def __init__(self, queue, crop=False) -> None:
        print("iSDF Node: starting", os.getpid())
        print("Waiting for first frame...")

        self.queue = queue

        self.crop = crop

        # self.first_pose_inv = None
        # self.world_transform = trimesh.transformations.rotation_matrix(
        #         np.deg2rad(-90), [1, 0, 0]) @ trimesh.transformations.rotation_matrix(
        #         np.deg2rad(90), [0, 1, 0])

        rospy.init_node("isdf", anonymous=True)
        _subscriber = message_filters.Subscriber("/orb_slam3_ros_wrapper/pose", Pose) #Pose, self.callback_pose)
        __subscriber = message_filters.Subscriber("/orb_slam3_ros_wrapper/depth", Image) #Image, self.callback_depth)
        ___subscriber = message_filters.Subscriber("/orb_slam3_ros_wrapper/rgb", Image) #Image, self.callback_rgb)

        ts = message_filters.ApproximateTimeSynchronizer([_subscriber, __subscriber, ___subscriber], 10, 0.1, allow_headerless=True)
        ts.registerCallback(self.callback)
        rospy.spin()

    def callback(self, pose, depth, rgb):
        if self.queue.full():
            return

        # start = perf_counter()

        rgb_np = np.frombuffer(rgb.data, dtype=np.uint8)
        rgb_np = rgb_np.reshape(rgb.height, rgb.width, 3)
        rgb_np = rgb_np[..., ::-1]

        depth_np = np.frombuffer(depth.data, dtype=np.uint16)
        depth_np = depth_np.reshape(depth.height, depth.width)

        # Formatting camera pose as a transformation matrix w.r.t. the world frame
        position = pose.position
        quat = pose.orientation
        trans = np.asarray([[position.x], [position.y], [position.z]])
        rot = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()
        camera_transform = np.concatenate((rot, trans), axis=1)
        camera_transform = np.vstack((camera_transform, [0.0, 0.0, 0.0, 1.0]))

        camera_transform = np.linalg.inv(camera_transform)

        # Crop images to remove the black edges after calibration
        if self.crop:
            w = rgb.width
            h = rgb.height
            mw = 40
            mh = 20
            rgb_np = rgb_np[mh:(h - mh), mw:(w - mw)]
            depth_np = depth_np[mh:(h - mh), mw:(w - mw)]

        try:
            self.queue.put(
                (rgb_np.copy(), depth_np.copy(), camera_transform.copy()),
                block=False,
            )
        except queue.Full:
            pass

        del rgb_np
        del depth_np
        del camera_transform

class iSDFFrankaNode:
    def __init__(self, queue, crop=False, ext_calib = None) -> None:
        print("iSDF Franka Node: starting", os.getpid())
        print("Waiting for first frame...")

        self.queue = queue
        self.crop = crop
        self.camera_transform = None 
        self.cal = ext_calib
        self.rgb, self.depth, self.pose = None, None, None
        self.first_pose_inv = None

        rospy.init_node("isdf_franka")
        rospy.Subscriber("/franka/rgb", Image, self.main_callback, queue_size=1)
        rospy.Subscriber("/franka/depth", Image, self.depth_callback, queue_size=1)
        rospy.Subscriber("/franka/pose", Pose, self.pose_callback, queue_size=1)
        rospy.spin()

    def main_callback(self, msg):
        # main callback is RGB, and uses the latest depth + pose 
        # TODO: subscribe to single msg type that contains (image, depth, pose)
        rgb_np = np.frombuffer(msg.data, dtype=np.uint8)
        rgb_np = rgb_np.reshape(msg.height, msg.width, 3)
        rgb_np = rgb_np[..., ::-1]
        self.rgb = cv2.resize(rgb_np, (1280, 720), interpolation=cv2.INTER_AREA)

        del rgb_np

        if self.depth is None or self.pose is None: 
            return
        # self.show_rgbd(self.rgb, self.depth, 0)

        try:
            self.queue.put(
                (self.rgb.copy(), self.depth.copy(), self.pose.copy()),
                block=False,
            )
        except queue.Full:
            pass

    def depth_callback(self, msg):
        depth_np = np.frombuffer(msg.data, dtype=np.uint16)
        depth_np = depth_np.reshape(msg.height, msg.width)
        self.depth = cv2.resize(depth_np, (1280, 720), interpolation=cv2.INTER_AREA)
        del depth_np

    def pose_callback(self, msg):
        position = msg.position
        quat = msg.orientation
        trans = np.asarray([position.x, position.y, position.z])
        rot = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()
        trans, rot = self.ee_to_cam(trans, rot)
        camera_transform = np.concatenate((rot, trans.reshape((3, 1))), axis=1)
        camera_transform = np.vstack((camera_transform, [0.0, 0.0, 0.0, 1.0]))
        self.pose = camera_transform

        del camera_transform

    def ee_to_cam(self, trans, rot):
        # transform the inverse kinematics EE pose to the realsense pose
        cam_ee_pos = np.array(self.cal[0]['camera_ee_pos'])
        cam_ee_rot = np.array(self.cal[0]['camera_ee_ori_rotvec'])
        cam_ee_rot = Rotation.from_rotvec(cam_ee_rot).as_matrix()

        camera_world_pos = trans + rot @ cam_ee_pos
        camera_world_rot = rot @ cam_ee_rot
        return camera_world_pos, camera_world_rot

def show_rgbd(rgb, depth, timestamp):
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.imshow(rgb)
    plt.title('RGB ' + str(timestamp))
    plt.subplot(2, 1, 2)
    plt.imshow(depth)
    plt.title('Depth ' + str(timestamp))
    plt.draw()
    plt.pause(1e-6)

def get_latest_frame(q):
    # Empties the queue to get the latest frame
    message = None
    while True:
        try:
            message_latest = q.get(block=False)
            if message is not None:
                del message
            message = message_latest

        except queue.Empty:
            break

    return message


