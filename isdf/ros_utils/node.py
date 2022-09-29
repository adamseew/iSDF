import os
import queue
import numpy as np
from scipy.spatial.transform import Rotation
import rospy
import trimesh
# import cv2
# import imgviz
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


