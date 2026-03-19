#!/usr/bin/env python3
"""
Main grasp processing

:Author: Lukas Rustler
"""

import multiprocessing
import time
import signal
import os
from pybullet_grasper.bullet_classes import Client, Robotiq_2f_85, Barrett
from pybullet_grasper.utils import Logger
from pybullet_grasper.grasp import Grasp
from pybullet_grasper.visualization import Visualizer
import sys
from subprocess import call
import multiprocessing as mp
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as ts
from typing import Any, Optional, List, Union


Vector3 = Union[List[float], np.ndarray]


def signal_handler(signal: int, frame: Any) -> None:
    """
    Handler for killing the program
    :param signal: signal type
    :type signal: int
    :param frame: current frame where signal interrupted execution
    :type frame: Any
    :return: None
    :rtype: None
    """
    print("Killing on user request")
    for _ in os.popen("pgrep -f grasp_generator.py").read().strip().splitlines():
        call("kill -9 "+_, shell=True)
    sys.exit(0)


class ClientProcess(mp.Process):
    def __init__(self, client: Client, id: int, gripper_poses: List[Any], score: Any, after_grasp_poses: Any) -> None:
        """
        Worker process that evaluates a batch of candidate grasp poses.

        :param client: shared simulation client template
        :type client: Client
        :param id: worker/process identifier
        :type id: int
        :param gripper_poses: pose candidates assigned to this worker
        :type gripper_poses: list[Any]
        :param score: shared array storing pose quality scores
        :type score: Any
        :param after_grasp_poses: shared array storing object poses after grasping
        :type after_grasp_poses: Any
        :return: None
        :rtype: None
        """
        super().__init__()
        self.client = client
        self.id = id
        self.gripper_poses = gripper_poses
        self.client.grasper = Grasp(self.client)
        self.score = score
        self.after_grasp_poses = after_grasp_poses
        self.axes = [2, 1, 0]

    def test_grasp(self, gripper_pose: Any, pose_id: int) -> int:
        """
        Simulates one grasp attempt and stores its score.

        :param gripper_pose: candidate end-effector pose
        :type gripper_pose: Any
        :param pose_id: index into shared result arrays
        :type pose_id: int
        :return: 0 after evaluation completes
        :rtype: int
        """

        if self.client.config.gripper == "barrett":
            gripper_f = Barrett
        elif self.client.config.gripper == "robotiq":
            gripper_f = Robotiq_2f_85

        self.client.gripper = gripper_f(self.client,
                                            [gripper_pose.position.__getattribute__(_) for _ in ["x", "y", "z"]],
                                            [gripper_pose.orientation.__getattribute__(_) for _ in ["x", "y", "z", "w"]])

        self.client.step_simulation()

        collisions = self.client.getContactPoints(self.client.gripper.id, self.client.plane)
        for c in collisions:
            if c[self.client.contactPoints["DISTANCE"]] < -1e-2:
                self.client.logger.log_debug("Gripper is in collision with the plane")
                return 0

        if self.client.config.contacts.move_to_grasp_pose:
                self.client.gripper.move(1, 0.1, 1)

        collisions = self.client.getContactPoints(self.client.gripper.id, self.client.object.id)
        for c in collisions:
            if c[self.client.contactPoints["DISTANCE"]] < -1e-2:
                self.client.logger.log_debug("Gripper is in collision with the object")
                return 0

        before_grasp_pos, before_grasp_ori = self.client.getBasePositionAndOrientation(self.client.object.id)
        ret = self.client.gripper.set_gripper_pose(1, wait=True)

        if ret == 1:
            if self.client.config.visualization.mode in [1, 2]:
                self.client.visualizer.window.close()
            return 0

        self.client.gripper.stop()

        if self.client.config.analytical.enabled and self.client.config.visualization.mode in [1, 2]:
            self.client.grasper.evaluate_contact()
            self.client.logger.log_info("Press Q to continue")
            while self.client.is_alive() and self.client.visualizer.is_alive:
                self.client.step_simulation()
                if self.client.visualizer.last_key is not None and self.client.visualizer.last_key == "q":
                    self.client.visualizer.last_key = None
                    break

        start_pos, start_ori = self.client.getBasePositionAndOrientation(self.client.object.id)
        start_pos_gripper = np.array(self.client.getLinkState(self.client.gripper.id, 1)[0])
        initial_difference = np.abs(start_pos - start_pos_gripper)

        self.client.lower_the_plane(0.2)

        lift_pos_gripper = np.array(self.client.getLinkState(self.client.gripper.id, 1)[0])

        lift_pos, lift_ori = self.client.getBasePositionAndOrientation(self.client.object.id)
        for angle in [np.pi/2, -np.pi/2, 0]:
            r = self.client.gripper.move(0, angle, 2)
            if r == -1:
                self.client.logger.log_debug("Move timeout")
                return 0

        steps = self.client.steps_done

        # wait for one simulated second
        while self.client.is_alive():
            self.client.step_simulation(0)
            if self.client.steps_done - steps == self.client.config.contacts.time_step:
                break

        end_pos, end_ori = self.client.getBasePositionAndOrientation(self.client.object.id)

        # if the object is still in the gripper it can be considered stable grasp
        stable_grasp = len(self.client.getContactPoints(self.client.gripper.id, self.client.object.id)) > 0

        # Difference between start and end of gripper rotation
        height_dif = np.abs(lift_pos[2] - end_pos[2])*1000
        position_distance = np.linalg.norm(end_pos - np.array(lift_pos))*1000

        # difference of change of height of gripper and the object
        obj_gripper_height_diff = np.abs(np.abs(start_pos[2]-lift_pos_gripper[2])-initial_difference[2])*1000

        # if the object slided less than 2.5cm
        firm_grasp = np.logical_and(obj_gripper_height_diff < 25,  position_distance < 25)

        # change of position during grasping
        grasp_position_change = np.linalg.norm(np.array(before_grasp_pos) - start_pos)*1000

        Gr_R = np.eye(4)
        Gr_R[:3, :3] = ts.from_quat([gripper_pose.orientation.x, gripper_pose.orientation.y, gripper_pose.orientation.z, gripper_pose.orientation.w]).as_matrix()
        direction = np.matmul(Gr_R, [0, 0, 1, 1])[:3]  # Rotate Z-axis vector to point in direction of gripper
        direction /= np.linalg.norm(direction)  # normalize
        # distance_from_top_grasp = np.arctan2(np.linalg.norm(np.cross(direction, [0, 0, -1])), np.dot(direction, [0, 0, -1]))
        distance_from_top_grasp = 1 - np.min([np.abs(np.dot(direction, [0, 0, 1])), 1])

        self.score[pose_id] = 1/(height_dif + position_distance + obj_gripper_height_diff) + firm_grasp + stable_grasp + distance_from_top_grasp
        self.after_grasp_poses[pose_id*7:(pose_id+1)*7] = list(start_pos) + list(start_ori)

        self.client.logger.log_debug(f"""Total score: {self.score[pose_id]}
                                    Angle from top grasp: {distance_from_top_grasp:.2f}
                                    Position change after grasp {grasp_position_change:.2f}mm
                                    Height difference after rotation: {height_dif:.2f}mm
                                    Position distance after rotation: {position_distance:.2f}mm
                                    Relative height change after lift: {obj_gripper_height_diff:.2f}mm
                                    Firm grasp: {firm_grasp}, Stable grasp: {stable_grasp}""")

        if self.client.config.visualization.mode == 1:
            self.client.logger.log_info("Press Q to end")
            while self.client.is_alive() and self.client.visualizer.is_alive:
                self.client.step_simulation()
                if self.client.visualizer.last_key is not None and self.client.visualizer.last_key == "q":
                    self.client.visualizer.last_key = None
                    break
        return 0

    def handle_timeout(self, signum: int, frame: Any) -> None:
        """
        Raises timeout exception from signal alarm callback.

        :param signum: POSIX signal number
        :type signum: int
        :param frame: interrupted frame
        :type frame: Any
        :return: None
        :rtype: None
        """
        raise TimeoutError

    def run(self) -> int:
        """
        Executes grasp evaluation over all assigned poses.

        :return: 0 when worker exits
        :rtype: int
        """
        if self.client.visualizer is None:
            signal.signal(signal.SIGALRM, self.handle_timeout)
        for pose_id, pose in enumerate(self.gripper_poses):
            self.client.reset()
            if self.client.config.visualization.mode not in [1, 2]:
                signal.alarm(2)

            try:
                self.test_grasp(pose, pose_id)
            except TimeoutError:
                self.client.logger.log_debug("Timeout")
                continue
            finally:
                signal.alarm(0)

        return 0


class GraspGenerator:
    def __init__(self, object_name: str = "object_0.obj", init_position: Vector3 = [0, 0, 0],
                 config_name: str = "default.yaml", ros: bool = True) -> None:
        """
        Creates grasp candidates and prepares simulation resources.

        :param object_name: object mesh name or path
        :type object_name: str, optional, default="object_0.obj"
        :param init_position: initial object position in world coordinates
        :type init_position: list[float] | np.ndarray, optional, default=[0, 0, 0]
        :param config_name: configuration file name or path
        :type config_name: str, optional, default="default.yaml"
        :param ros: whether ROS message types should be used
        :type ros: bool, optional, default=True
        :return: None
        :rtype: None
        """

        if ros:
            from geometry_msgs.msg import Pose, Point, Quaternion
            from pybullet_grasper.srv import GenerateGraspsResponse
        else:
            from pybullet_grasper.utils import Pose, Point, Quaternion, GenerateGraspsResponse

        self.PoseImport = Pose
        self.PointImport = Point
        self.QuaternionImport = Quaternion
        self.GenerateGraspsResponseImport = GenerateGraspsResponse

        self.file_dir = os.path.dirname(os.path.abspath(__file__))
        self.client = Client(config_name,object_name=object_name, init_position=init_position)

        self.client.config.visualization.mode = 0 # Fixed value without config setting

        self.client.logger = Logger(ros, debug=self.client.debug)
        self.ros = ros

        if self.client.config.gripper == "barrett":
            gripper_f = Barrett
        elif self.client.config.gripper == "robotiq":
            gripper_f = Robotiq_2f_85

        self.client.gripper = gripper_f(self.client, [0, 0, 0], [0, 0, 0, 1])
        self.gripper_mesh = o3d.geometry.TriangleMesh()
        self.fingers_mesh = o3d.geometry.TriangleMesh()
        self.plane_mesh = None
        self.load_gripper()

        self.gripper_poses = []

        # self.client.visualizer = Visualizer(self.client, True)
        points = np.asarray(self.client.object.pc.points)
        R_old = np.eye(4)
        for point, normal in zip(points, np.asarray(self.client.object.pc.normals)):

            a = np.array([0, 0, 1])
            b = np.array(-normal)

            if np.linalg.norm(b - [0, 0, -1]) < 1e-5:
                self.client.logger.log_debug("This happened")
                Q = [0, 1, 0, 0]
            else:
                rotAngle = np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))
                rotAxis = np.cross(a, b)
                rotAxis /= np.linalg.norm(rotAxis)
                rot_vec = np.array(rotAxis) * rotAngle
                Q = ts.from_rotvec(rot_vec).as_quat()
            rot_vec = b * (np.pi/2)
            rot = ts.from_rotvec(rot_vec).as_quat()
            Q2 = quaternion_multiply(rot, Q)

            rot_vec = b * np.pi
            rot = ts.from_rotvec(rot_vec).as_quat()
            Q3 = quaternion_multiply(rot, Q)


            pose_added = False
            for q in [Q, Q2, Q3]:
                if pose_added:
                    break
                for offset in [0.1, 0.11, 0.12]:  # [0.10, 0.11, 0.12]
                    R = np.eye(4)
                    R[:3, :3] = ts.from_quat(q).as_matrix()
                    R[:3, 3] = point + offset * normal
                    self.gripper_mesh.transform(R @ np.linalg.inv(R_old))

                    R_old = R

                    if not self.gripper_mesh.is_intersecting(self.client.object.mesh) and not self.gripper_mesh.is_intersecting(self.plane_mesh):
                        if self.client.config.contacts.move_to_grasp_pose:
                            offset += 0.1
                        point_ = point + offset * normal
                        self.gripper_poses.append(self.PoseImport(self.PointImport(*point_), self.QuaternionImport(*q)))
                        pose_added = True
                        break

        # this is just pure guest that when the poses are randomized it will be better distributed along multiple cores
        np.random.shuffle(self.gripper_poses)

        self.num_poses = len(self.gripper_poses)
        self.client.logger.log_debug(f"Number of poses: {self.num_poses} from {points.shape[0]} points")

        self.visualization = False
        if self.client.config.visualization.mode in [1, 2]:
            self.client.visualizer = Visualizer(self.client, self.client.config.visualization.mode == 1)
            self.visualization = True
            if self.num_poses > 1:
                self.client.logger.log_info("Visualization enabled -> only one proces is used")
                self.num_poses = 1

    def run(self, custom_pose: Optional[Any] = None) -> Any:
        """
        Runs grasp evaluation either for all generated poses or one custom pose.

        :param custom_pose: optional pose override for one-off evaluation
        :type custom_pose: Any, optional, default=None
        :return: grasp response with poses and quality metrics
        :rtype: Any
        """
        processes = []

        if custom_pose is not None:
            self.gripper_poses = [custom_pose]
            self.num_poses = 1
            self.client.config.visualization.mode = 1
            self.client.config.visualization.debug_sleep = self.client.config.visualization.debug_sleep_back
            self.client.visualizer = Visualizer(self.client, True)
        else:
            self.client.config.visualization.debug_sleep_back = self.client.config.visualization.debug_sleep
            self.client.config.visualization.debug_sleep = 0.0

        out = self.GenerateGraspsResponseImport()

        num_proc = np.min([mp.cpu_count()-2, self.num_poses])
        num_per_process = (self.num_poses // num_proc) + 1
        break_point = num_proc - (num_per_process * num_proc - self.num_poses)
        last = 0
        for i in range(num_proc):
            if i < break_point:
                shift = num_per_process
            else:
                shift = num_per_process - 1
            poses = self.gripper_poses[last:last + shift]
            last = last + shift
            p = ClientProcess(self.client, i, poses, multiprocessing.Array('d', np.repeat(-1, len(poses))), multiprocessing.Array('d', np.repeat(-1, 7*len(poses))))
            if self.num_poses != 1:
                p.start()
            else:
                p.run()
                out.poses = [p.gripper_poses[0]]
                out.qualities = [p.score[0]]
                return out
            processes.append(p)

        for p in processes:
            p.join()
            for idx in range(len(p.gripper_poses)):
                out.poses.append(p.gripper_poses[idx])
                out.qualities.append(p.score[idx])
                out.poses_after_grasps.append(p.after_grasp_poses[idx*7:(idx+1)*7])

        sort_ids = np.argsort(out.qualities)[::-1]
        out.poses = np.array(out.poses)[sort_ids]
        out.qualities = np.array(out.qualities)[sort_ids]
        out.poses_after_grasps = (np.array(out.poses_after_grasps)[sort_ids]).ravel()
        return out

    def load_gripper(self) -> None:
        """
        Loads and composes simplified gripper collision geometry for pre-checks.

        :return: None
        :rtype: None
        """
        visualData = self.client.getVisualShapeData(self.client.gripper.id)

        for m in visualData:
            # Get information about individual parts of the object
            f_path = m[self.client.visualShapeData["FILE"]].decode("utf-8").replace("visual", "collision").replace(".obj", "_vhacd.obj")
            if not ("robotiq_arg2f_85_pad" in f_path or "contactile_vhacd" in f_path or "finger_tip" in f_path):
                continue
            link = m[self.client.visualShapeData["LINK"]]

            # non-base links
            if link != -1:
                # get link info
                linkState = self.client.getLinkState(self.client.gripper.id, link, computeLinkVelocity=0,
                                                     computeForwardKinematics=0)
                # get orientation and position wrt URDF
                ori = linkState[self.client.linkInfo["URDFORI"]]
                pos = linkState[self.client.linkInfo["URDFPOS"]]
            # link == -1 is base. For that, getBasePosition... needs to be used
            else:
                pos, ori = self.client.getBasePositionAndOrientation(self.client.gripper.id)

            m = o3d.io.read_triangle_mesh(f_path)
            R = np.eye(4)
            R[:3, :3] = ts.from_quat(ori).as_matrix()
            R[:3, 3] = pos
            m.transform(R)

            if "contactile_vhacd.obj" in f_path or "finger_tip" in f_path:
                bbox = m.get_oriented_bounding_box()
                bbox = bbox.scale(0.75, bbox.get_center())
                m = o3d.geometry.TriangleMesh().create_from_oriented_bounding_box(bbox)

            self.gripper_mesh += m
        self.plane_mesh = o3d.io.read_triangle_mesh(os.path.join(self.client.data_folder, "objects", "plane.obj"))#, [-5, -5, self.plane_lift]
        self.plane_mesh.translate([-5, -5, self.client.plane_lift], relative=True)


def quaternion_multiply(q1: Vector3, q2: Vector3) -> np.ndarray:
    """
    Multiplies two quaternions [x, y, z, w].

    :param q1: first quaternion
    :type q1: list[float] | np.ndarray
    :param q2: second quaternion
    :type q2: list[float] | np.ndarray
    :return: quaternion product [x, y, z, w]
    :rtype: np.ndarray
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    return np.array([
        x1 * w2 + y1 * z2 - z1 * y2 + w1 * x2,
        -x1 * z2 + y1 * w2 + z1 * x2 + w1 * y2,
        x1 * y2 - y1 * x2 + z1 * w2 + w1 * z2,
        -x1 * x2 - y1 * y2 - z1 * z2 + w1 * w2
    ])

def str_to_bool(str_bool: str) -> bool:
    """
    Converts a string value to boolean.

    :param str_bool: string representation of boolean
    :type str_bool: str
    :return: True for common truthy values, otherwise False
    :rtype: bool
    """
    if str_bool.lower() in ["true", "t", "1"]:
        return True
    return False