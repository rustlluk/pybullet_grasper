"""
Classes for pyBullet grasper containing the Client, Object and Robotiq 2F-85 Gripper and Barrett Hand

:Author: Lukas Rustler
"""


from pybullet_utils.bullet_client import BulletClient
import os
import numpy as np
import time
import open3d as o3d
from pybullet_grasper.utils import Config, get_data_folder, suppress_c_output
import inspect
from typing import Optional, Any, List, Tuple, Union


Vector3 = Union[List[float], np.ndarray]
Quaternion = Union[List[float], np.ndarray]


class Client(BulletClient):
    """
    Wrapper around :class:`pybullet_utils.bullet_client.BulletClient` used by the grasper pipeline.

    The class initializes the simulation world, loads the active graspable object,
    and stores convenience indices for pyBullet tuple-based APIs.
    """

    # As dict, because IntEnum is about 1.5-2x slower
    jointInfo = {name: i for i, name in enumerate(["INDEX", "NAME", "TYPE", "QINDEX", "UINDEX", "FLAGS", "DAMPING",
                                                   "FRICTION", "LOWERLIMIT", "UPPERLIMIT", "MAXFORCE", "MAXVELOCITY",
                                                   "LINKNAME", "AXIS", "PARENTPOS", "PARENTORN", "PARENTINDEX"])}
    jointStates = {name: i for i, name in enumerate(["POSITION", "VELOCITY", "FORCES", "TORQUE"])}

    linkInfo = {name: i for i, name in enumerate(["WORLDPOS", "WORLDORI", "INERTIAPOS", "INERTIAORI", "URDFPOS",
                                                  "URDFORI", "LINVEL", "ANGVEL"])}

    contactPoints = {name: i for i, name in enumerate(["FLAG", "IDA", "IDB", "INDEXA", "INDEXB", "POSITIONA",
                                                       "POSITIONB", "NORMAL", "DISTANCE", "FORCE",
                                                       "FRICTION1", "FRICTIONDIR1", "FRICTION2", "FRICTIONDIR2"])}
    dynamicsInfo = {name: i for i, name in enumerate(["MASS", "FRICTION", "INTERTIADIAGONAL", "INERTIAPOS", "INERTIAOR",
                                                      "RESTITUTION", "ROLLINGFRICTION", "SPINNINGFRICTION", "DAMPING",
                                                      "STIFFNESS", "BODYTYPE", "MARGIN"])}
    visualShapeData = {name: i for i, name in enumerate(["ID", "LINK", "GEOMTYPE", "DIMS", "FILE", "POS", "ORI",
                                                         "COLOR", "TEXTURE"])}

    def __init__(self, config_path: str, object_name: str = "", init_position: Vector3 = [0, 0, 0]) -> None:
        """
        Client class which inherits from BulletClient. Runs the physics client with desired mode and prepares dicts
        for easier access to things

        :param config_path: path to the config file
        :type config_path: str
        :param object_name: name of the object; can be relative to the calling script; relative to source_code/objects or an absolute path
        :type object_name: str, optional, default=""
        :param init_position: initial position of the object
        :type init_position: list[float] | np.ndarray, optional, default=[0, 0, 0]
        :return: None
        :rtype: None
        """

        super().__init__(self.DIRECT)
        self.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
        self.configureDebugVisualizer(self.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        self.configureDebugVisualizer(self.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        self.configureDebugVisualizer(self.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        self.setGravity(0, 0, -9.81)

        self.gripper = None
        self.visualizer = None
        self.grasper = None
        self.logger = None
        self.last_step = time.time()
        self.last_render = time.time()
        self.steps_done = 0
        self.msg = None
        self.state = None
        self.plane_lift = 0

        self.data_folder = get_data_folder()

        self.file_dir = os.path.dirname(os.path.realpath(__file__))
        self.plane = self.loadURDF(os.path.join(self.data_folder, "urdf", "plane.urdf"), [-5, -5, self.plane_lift])
        if not os.path.exists(os.path.join(self.data_folder, "urdf")):
            os.makedirs(os.path.join(self.data_folder, "urdf"))
        self.object = None

        if not os.path.isabs(config_path):
            for c_path in [config_path, os.path.join(os.getcwd(), os.path.dirname(inspect.stack()[-1].filename), config_path), os.path.join(self.data_folder, "configs", config_path)]:
                if os.path.exists(c_path):
                    config_path = c_path
                    break

        self.config = Config(config_path)
        self.debug = self.config.debug

        self.visualization_objects = [self.plane]

        if not os.path.isabs(object_name):
            for o_path in [object_name,
                           os.path.join(os.getcwd(), os.path.dirname(inspect.stack()[-1].filename), object_name),
                           os.path.join(self.data_folder, "objects", object_name)]:
                if os.path.exists(o_path):
                    object_name = o_path
                    break

        self.load_objects(object_name, init_position)
        # default 1/240
        self.setTimeStep(1 / self.config.contacts.time_step)

    def step_simulation(self, sleep_duration: Optional[float] = None) -> None:
        """
        Updates the simulation

        :param sleep_duration: duration to sleep before the next simulation step
        :type sleep_duration: float, optional, default=None
        :return: None
        :rtype: None
        """

        if self.debug and sleep_duration is None and self.config.visualization.debug_sleep != 0:
            sleep_duration = self.config.visualization.debug_sleep

        # This is here to keep events and everything in open3D work even if we want slower simulation
        if sleep_duration is None or (time.time() - self.last_step) > sleep_duration:
            self.stepSimulation()
            self.last_step = time.time()
            self.steps_done += 1

        if self.visualizer is not None and time.time() - self.last_render > 0.01 and self.visualizer.is_alive:
            self.visualizer.render()
            self.last_render = time.time()

    def reset(self) -> None:
        """
        Function to reset simulation to default state

        :return: None
        :rtype: None
        """
        self.resetSimulation()
        self.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
        self.setGravity(0, 0, -9.81)
        self.setTimeStep(1 / self.config.contacts.time_step)
        self.plane = self.loadURDF(os.path.join(self.data_folder, "urdf", "plane.urdf"), [-5, -5, self.plane_lift])
        self.visualization_objects = [self.plane]
        self.load_objects()

    def is_alive(self) -> bool:
        """
        Checks whether the engine is still running

        :return: True when running
        :rtype: bool
        """
        return True if self._client >= 0 else False

    def load_objects(self, path: str = "", init_position: Vector3 = [0, 0, 0]) -> None:
        """
        Load graspable objects. With possibility of convex decomposition with V-HACD

        :param path: path to the mesh
        :type path: str, optional, default=""
        :param init_position: initial position of the object
        :type init_position: list[float] | np.ndarray, optional, default=[0, 0, 0]
        :return: None
        :rtype: None
        """
        if path != "":
            g = GraspableBody(self, init_position)
            g.prepare_body(path, True)
            self.object = g
        else:
            self.object.prepare_body(path, False)
        self.visualization_objects.append(self.object.id)

    def read_info(self, obj_id: int) -> int:
        """
        Add mesh to the visualizer

        :param obj_id: id of the object; given by pybullet
        :type obj_id: int
        :return: 0 for success
        :rtype: int
        """
        visualData = self.getVisualShapeData(obj_id)
        for m in visualData:
            # Get information about individual parts of the object
            f_path = m[self.visualShapeData["FILE"]].decode("utf-8")#.replace("visual", "collision").replace(".obj", "_vhacd.obj")
            col = m[self.visualShapeData["COLOR"]]
            link = m[self.visualShapeData["LINK"]]

            # non-base links
            if link != -1:
                # get link info
                linkState = self.getLinkState(obj_id, link, computeLinkVelocity=0,
                                                     computeForwardKinematics=0)
                # get orientation and position wrt URDF
                ori = linkState[self.linkInfo["URDFORI"]]
                pos = linkState[self.linkInfo["URDFPOS"]]
            # link == -1 is base. For that, getBasePosition... needs to be used
            else:
                pos, ori = self.getBasePositionAndOrientation(obj_id)

            self.msg.pos += pos
            self.msg.ori += ori
            self.msg.colors += col[:-1]
            self.msg.paths.append(f_path)

        return 0

    def lower_the_plane(self, target: float) -> int:
        """
        Moves the support plane joint to the requested position.

        :param target: desired prismatic joint position of the plane
        :type target: float
        :return: 0 when target is reached, -1 when timeout is reached
        :rtype: int
        """
        self.setJointMotorControl2(self.plane, 0, self.POSITION_CONTROL, targetPosition=target,
                                   maxVelocity=10)
        start_steps = self.steps_done
        while True:
            position = self.getJointState(self.plane, 0)[0]
            if np.abs(position - target) < 1e-3:
                return 0
            if self.steps_done - start_steps > self.config.contacts.time_step:
                return -1
            self.step_simulation()


class GraspableBody:
    """
    Stores geometry, sampling and runtime metadata for one graspable object.
    """

    def __init__(self, client: Client, init_position: Vector3) -> None:
        """
        Initializes an empty graspable object container.

        :param client: active physics client
        :type client: Client
        :param init_position: initial translation used when loading the object
        :type init_position: list[float] | np.ndarray
        :return: None
        :rtype: None
        """
        self.client = client
        self.id = None
        self.mesh = None
        self.pc = None
        self.vertices = None
        self.mesh_original = None
        self.center = None
        self.r = None
        self.name = "object"
        self.obj_path = None
        self.init_pos = init_position
        self.init_ori = [0, 0, 0, 1]

    def prepare_body(self, model: str, do_init: bool = True) -> None:
        """
        Load graspable objects. With possibility of convex decomposition with V-HACD

        :param model: path to the .obj file
        :type model: str
        :param do_init: whether to regenerate VHACD/URDF assets before loading
        :type do_init: bool, optional, default=True
        :return: None
        :rtype: None
        """
        if do_init:
            model_name = model.split(".obj")[0].split("/")[-1]
            model_vhacd = model.replace(".obj", "_vhacd.obj")
            #if not os.path.exists(model.replace(".obj", "_vhacd.obj")):
            with suppress_c_output():
                self.client.vhacd(model, model_vhacd, "", depth=15, resolution=100000)#, resolution=10000, maxNumVerticesPerCH=1, gamma=0.0005, concavity=0)
            self.prepare_mesh(model)

            with open(os.path.join(self.client.data_folder, "urdf", "object_default.urdf"), "r") as f:
                urdf = f.read()
            urdf = urdf.replace("OBJECTNAME", model_name).replace("LATERALFRICTION", "1") \
                .replace("ROLLINGFRICTION", "0").replace("MASS", "0.5").replace("FILENAMEVHACD", model_vhacd) \
                .replace("FILENAME", model_vhacd)

            with open(model.replace("objects", "urdf").replace(".obj", ".urdf"), "w") as f:
                f.write(urdf)

            self.obj_path = model.replace("objects", "urdf").replace(".obj", ".urdf") # this is only needed for easier reset
        self.id = self.client.loadURDF(self.obj_path, self.init_pos, self.init_ori, useFixedBase=False)#, flags=self.client.URDF_USE_INERTIA_FROM_FILE)

    def prepare_mesh(self, model: str) -> None:
        """
        Prepares information about the object

        :param model: path to the .obj file
        :type model: str
        :return: None
        :rtype: None
        """

        self.mesh_original = o3d.io.read_triangle_mesh(model)
        self.mesh = o3d.io.read_triangle_mesh(model.replace(".obj", "_vhacd.obj"))
        self.mesh_original.translate(self.init_pos, relative=True)
        self.mesh.translate(self.init_pos, relative=True)
        # self.vertices = np.asarray(self.mesh.vertices)
        # self.client.plane_lift = np.min(self.vertices[:, 2])-0.01
        self.client.plane_lift = self.mesh.get_axis_aligned_bounding_box().get_min_bound()[2]-0.005
        box_points = np.asarray(self.mesh_original.get_axis_aligned_bounding_box().get_box_points())
        # box_points[np.abs(box_points[:, 2] - np.min(box_points[:, 2])) < 1e-5, 2] += np.min([0.075, height/2])

        crop_box = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(box_points))
        points = np.max([np.min([500, int(self.mesh_original.get_surface_area()*10000)]), 250])
        self.pc = self.mesh_original.sample_points_uniformly(number_of_points=points)
        self.pc.estimate_normals()
        self.pc.orient_normals_consistent_tangent_plane(10)
        self.pc.normalize_normals()
        self.pc = self.pc.crop(crop_box)

        self.vertices = np.asarray(self.mesh.vertices)
        bounding_points_indexes = np.hstack((np.argmax(self.vertices, 0), np.argmin(self.vertices, 0)))
        bounding_points = self.vertices[bounding_points_indexes, :]
        self.center = self.mesh.get_center()
        self.r = np.max(np.linalg.norm(bounding_points - self.center, axis=1))


class Robotiq_2f_85:
    """
    Runtime controller for the Robotiq 2F-85 gripper loaded in pyBullet.
    """

    JOINTS = ["left_inner_knuckle_joint", "left_inner_finger_joint", "right_outer_knuckle_joint",
              "right_inner_knuckle_joint", "right_inner_finger_joint"]
    SIGNS = [1, -1, 1, 1, -1, 1, -1]

    def __init__(self, client: Client, position: Vector3 = [0, 0, 1], orientation: Quaternion = [0, 0, 0, 1]) -> None:
        """
        Class for the Robotiq 2f-85 gripper

        :param client: instance of Client, holding info about the scene
        :type client: Client
        :param position: spawn position of the gripper base in world frame
        :type position: list[float] | np.ndarray, optional, default=[0, 0, 1]
        :param orientation: spawn orientation quaternion of the gripper base
        :type orientation: list[float] | np.ndarray, optional, default=[0, 0, 0, 1]
        :return: None
        :rtype: None
        """
        self.client = client
        self.position = position
        self.orientation = orientation
        self.id = self.load_gripper()
        self.constraints, self.finger_joint_id, self.finger_info = self.prepare_gripper()
        self.current_goal = -1
        self.timeout = -1
        self.name = "gripper"
        self.max_force = 185
        self.client.visualization_objects.append(self.id)
        self.set_gripper_pose(0)
        #TODO: just test
        self.height_diff = 0

    def load_gripper(self) -> int:
        """
        Loads the grippers URDF into the scene

        :return: body id of the loaded gripper
        :rtype: int
        """
        # Path needs to be changed to load the meshes correctly

        gripper = self.client.loadURDF(os.path.join(self.client.data_folder, "grippers/robotiq_2f_85/robotiq_2f_85_contactile.urdf"),
                                       self.position, self.orientation,
                                       useFixedBase=False,
                                       flags=self.client.URDF_USE_INERTIA_FROM_FILE)
        return gripper

    def prepare_gripper(self) -> Tuple[List[int], int, Tuple[Any, ...]]:
        """
        Sets constraints for joints to works as mimic joint in URDF

        :return: created constraints, driver joint id, and driver joint info tuple
        :rtype: tuple[list, int, tuple]
        """

        # find ids of the 6 articulated joints
        joints_ids = []
        finger_joint_id = -1
        for idx in range(self.client.getNumJoints(self.id)):
            j_name = self.client.getJointInfo(self.id, idx)[1].decode("UTF-8")
            if "finger_joint" == j_name:
                finger_joint_id = idx
            if j_name in self.JOINTS:
                joints_ids.append((j_name, idx))

        assert finger_joint_id != -1, "Finger joint not found in the URDF"
        finger_info = self.client.getJointInfo(self.id, finger_joint_id)

        # Set constraints -> only one joint (finger_joint) is articulated in reality and 5 others should follow it
        # Set a gear constraint
        constraints = []
        for joint in joints_ids:
            idx = joint[1]
            s = self.SIGNS[self.JOINTS.index(joint[0])]
            constraints.append(self.client.createConstraint(self.id, finger_joint_id,
                                                            self.id, idx,
                                                            jointType=self.client.JOINT_GEAR,
                                                            jointAxis=[1, 0, 0],
                                                            parentFramePosition=[0, 0, 0],
                                                            childFramePosition=[0, 0, 0]))

            # Some have different direction, because of axes settings etc.
            self.client.changeConstraint(constraints[-1], gearRatio=-s,  # +1 in gear ratio means reverse direction
                                         maxForce=185, erp=1)

        # This needs to be done, otherwise the joints are not moving properly
        for i in range(2, self.client.getNumJoints(self.id)):
            self.client.setJointMotorControl2(self.id, i, self.client.POSITION_CONTROL, targetVelocity=0, force=0)

        return constraints, finger_joint_id, finger_info

    def move(self, joint: int, target: float, velocity: float = 0.1) -> int:
        """
        Move joint to target position

        :param joint: id of the joint
        :type joint: int
        :param target: target position
        :type target: float
        :param velocity: maximum joint speed during the move command
        :type velocity: float, optional, default=0.1
        :return: 0 when target is reached, -1 when timeout is reached
        :rtype: int
        """
        self.client.setJointMotorControl2(self.id, joint, self.client.POSITION_CONTROL, targetPosition=target,
                                          maxVelocity=velocity, force=self.max_force)

        start_steps = self.client.steps_done
        while True:
            position = self.client.getJointState(self.id, joint)[0]
            if np.abs(position - target) < 1e-3:
                return 0
            if self.client.steps_done - start_steps > self.client.config.contacts.time_step*2:
                return -1
            self.client.step_simulation()

    def reset_joints(self) -> None:
        """
        Reset gripper joints to default

        :return: None
        :rtype: None
        """
        for _ in range(0, self.client.getNumJoints(self.id)):
            self.client.resetJointState(self.id, _, 0)

    def reset_constraints(self) -> None:
        """
        Remove all constraints

        :return: None
        :rtype: None
        """
        for c in self.constraints:
            self.client.removeConstraint(c)

    def stop(self) -> None:
        """
        Holds the current finger joint position using position control.

        :return: None
        :rtype: None
        """
        position = self.client.getJointState(self.id, self.finger_joint_id)[0]
        self.client.setJointMotorControl2(self.id, self.finger_joint_id,
                                          controlMode=self.client.POSITION_CONTROL, targetPosition=position,
                                          force=self.max_force,
                                          maxVelocity=self.finger_info[self.client.jointInfo["MAXVELOCITY"]])

    def set_gripper_pose(self, position: float, wait: bool = False) -> int:
        """
        Set goal position of gripper

        :param position: position of the gripper, 0 - closed, 1-open
        :type position: float
        :param wait: whether to wait for the motion to finish
        :type wait: bool, optional, default=False
        :return: motion result code, or 0 when command is issued without waiting
        :rtype: int
        """
        assert 0 <= position <= 1, "Gripper position must be between 0 and 1"

        # fitted from data
        position = 0.00499187 + 0.68547417 * position + 0.10344773 * np.power(position, 2)
        self.client.setJointMotorControl2(self.id, self.finger_joint_id,
                                          controlMode=self.client.POSITION_CONTROL, targetPosition=position,
                                          force=self.max_force, #self.finger_info[self.client.jointInfo["MAXFORCE"]],
                                          maxVelocity=self.finger_info[self.client.jointInfo["MAXVELOCITY"]])
        self.current_goal = position
        if wait:
            while True:
                r = self.motion_done(object_id=self.client.object.id,
                                     method=self.client.config.contacts.detection_method,
                                     contact_threshold=self.client.config.contacts.contact_threshold)
                if r != 0:
                    break
                self.client.step_simulation()
            return r
        return 0

    def motion_done(self, eps: float = 1e-2, object_id: int = -1, method: str = "force", contact_threshold: float = 10) -> int:
        """
        Checks whether gripper is in goal position

        :param eps: tolerance around the commanded joint value
        :type eps: float, optional, default=1e-2
        :param object_id: kept for API compatibility with other detectors
        :type object_id: int, optional, default=-1
        :param method: kept for API compatibility with other detectors
        :type method: str, optional, default="force"
        :param contact_threshold: torque threshold used to classify object contact
        :type contact_threshold: float, optional, default=10
        :return: 1 for normal closure, 2 for collision, 3 for timeout, 0 for no closure
        :rtype: int
        """

        if self.timeout == -1:
            self.timeout = self.client.steps_done
            return 0
        elif self.client.steps_done - self.timeout < 5:  # sow start of the gripper jaws
            return 0

        info = self.client.getJointState(self.id, self.finger_joint_id)
        if self.client.steps_done - self.timeout > self.client.config.contacts.time_step*2:  # do one second of simulated time only
            self.timeout = -1
            return 3

        if np.abs(info[0] - self.current_goal) < eps:
            self.timeout = -1
            return 1
        if info[-1] < self.max_force and np.abs(info[-1]) > contact_threshold:
            self.timeout = -1
            return 2
        return 0


class Barrett:
    """
    Runtime controller for the Barrett Hand gripper loaded in pyBullet.
    """

    JOINTS = ["bh_j32_joint", "bh_j12_joint", "bh_j22_joint", "bh_j11_joint"] # moveable joints
    MIMICS = {"bh_j32_joint": [0.3442622950819672, "bh_j33_joint"], "bh_j12_joint": [0.3442622950819672, "bh_j13_joint"],
              "bh_j22_joint": [0.3442622950819672, "bh_j23_joint"], "bh_j11_joint": [1, "bh_j21_joint"]}

    def __init__(self, client: Client, position: Vector3 = [0, 0, 1], orientation: Quaternion = [0, 0, 0, 1]) -> None:
        """
        Class for the Barrett Hand gripper

        :param client: instance of Client, holding info about the scene
        :type client: Client
        :param position: spawn position of the gripper base in world frame
        :type position: list[float] | np.ndarray, optional, default=[0, 0, 1]
        :param orientation: spawn orientation quaternion of the gripper base
        :type orientation: list[float] | np.ndarray, optional, default=[0, 0, 0, 1]
        :return: None
        :rtype: None
        """
        self.client = client
        self.position = position
        self.orientation = orientation
        self.id = self.load_gripper()
        self.client.visualization_objects.append(self.id)
        self.constraints, self.finger_joint_ids, self.fingers_info, self.finger_joints_names = self.prepare_gripper()
        self.name = "gripper"
        self.max_force = 120
        self.current_goal = -1
        self.timeout = -1

    def load_gripper(self) -> int:
        """
        Loads the grippers URDF into the scene

        :return: body id of the loaded gripper
        :rtype: int
        """
        # Path needs to be changed to load the meshes correctly

        gripper = self.client.loadURDF(os.path.join(self.client.data_folder, "grippers/barrett_hand/barrett_hand.urdf"),
                                       self.position, self.orientation,
                                       useFixedBase=False,
                                       flags=self.client.URDF_USE_INERTIA_FROM_FILE)

        return gripper

    def prepare_gripper(self) -> Tuple[List[int], List[int], List[Tuple[Any, ...]], List[str]]:
        """
        Sets constraints for joints to works as mimic joint in URDF

        :return: created constraints, moveable joint ids, joint info, and joint names
        :rtype: tuple[list, list, list, list]
        """

        # find ids of the 6 articulated joints
        joints_ids = []
        moveable_joint_ids = []
        fingers_info = []
        finger_joints_names = []
        for idx in range(self.client.getNumJoints(self.id)):
            j_name = self.client.getJointInfo(self.id, idx)[1].decode("UTF-8")
            if j_name in self.JOINTS:
                moveable_joint_ids.append(idx)
                finger_joints_names.append(j_name)
                fingers_info.append(self.client.getJointInfo(self.id, idx))
                for idxx in range(self.client.getNumJoints(self.id)):
                    j_namee = self.client.getJointInfo(self.id, idxx)[1].decode("UTF-8")
                    if j_namee == self.MIMICS[j_name][1]:
                        break
                joints_ids.append((idx, idxx, self.MIMICS[j_name][0]))

        # Set constraints -> only one joint (finger_joint) is articulated in reality and 5 others should follow it
        # Set a gear constraint
        constraints = []
        for joint1, joint2, s in joints_ids:
            constraints.append(self.client.createConstraint(self.id, joint1,
                                                            self.id, joint2,
                                                            jointType=self.client.JOINT_GEAR,
                                                            jointAxis=[1, 0, 0],
                                                            parentFramePosition=[0, 0, 0],
                                                            childFramePosition=[0, 0, 0]))

            # Some have different direction, because of axes settings etc.
            self.client.changeConstraint(constraints[-1], gearRatio=-1/s,  # +1 in gear ratio means reverse direction
                                         maxForce=10, erp=1)
            self.client.setJointMotorControl2(self.id, joint2, self.client.POSITION_CONTROL, targetVelocity=0, force=0)

        return constraints, moveable_joint_ids, fingers_info, finger_joints_names

    def reset_constraints(self) -> None:
        """
        Remove all constraints

        :return: None
        :rtype: None
        """
        for c in self.constraints:
            self.client.removeConstraint(c)

    def stop(self) -> None:
        """
        Holds all actuated finger joints at their current positions.

        :return: None
        :rtype: None
        """
        for idx, finger_joint_id in enumerate(self.finger_joint_ids):
            position = self.client.getJointState(self.id, finger_joint_id)[0]
            self.client.setJointMotorControl2(self.id, finger_joint_id,
                                              controlMode=self.client.POSITION_CONTROL, targetPosition=position,
                                              force=self.fingers_info[idx][self.client.jointInfo["MAXFORCE"]],
                                              maxVelocity=self.fingers_info[idx][self.client.jointInfo["MAXVELOCITY"]])

    def set_gripper_pose(self, position: float, pose: int = 0, velocity: Optional[float] = None, wait: bool = False) -> int:
        """
        Set goal position of gripper

        :param position: position of the gripper
        :type position: float
        :param pose: 0 for two-finger mode, 1 to fold one finger out of the way
        :type pose: int, optional, default=0
        :param velocity: unused placeholder for API compatibility
        :type velocity: float | None, optional, default=None
        :param wait: whether to wait for the motion to finish
        :type wait: bool, optional, default=False
        :return: motion result code, or 0 when command is issued without waiting
        :rtype: int
        """
        assert 0 <= position <= 1, "Gripper width must be between 0 and 1"
        joints_min = 0
        joint_max = 2.44

        # pose 0 = two finger
        # pose 1 = 'one' finger

        info_id = self.finger_joints_names.index("bh_j11_joint")
        finger_joint_id = self.finger_joint_ids[info_id]
        if pose == 0:
            third_finger_position = 0.01
        elif pose == 1:
            third_finger_position = 3.14
        self.client.setJointMotorControl2(self.id, finger_joint_id,
                                          controlMode=self.client.POSITION_CONTROL, targetPosition=third_finger_position,
                                          force=self.max_force,
                                          maxVelocity=self.fingers_info[info_id][self.client.jointInfo["MAXVELOCITY"]])

        position = ((position - 0) / (1 - 0)) * (joint_max - joints_min) + joints_min
        for finger_joint_id, finger_name in zip(self.finger_joint_ids, self.finger_joints_names):
            if finger_name == "bh_j11_joint":
                continue
            info_id = self.finger_joints_names.index(finger_name)
            self.client.setJointMotorControl2(self.id, finger_joint_id,
                                              controlMode=self.client.POSITION_CONTROL, targetPosition=position,
                                              force=self.max_force, #self.finger_info[self.client.jointInfo["MAXFORCE"]],
                                              maxVelocity=self.fingers_info[info_id][self.client.jointInfo["MAXVELOCITY"]])
        self.current_goal = position
        if wait:
            while True:
                r = self.motion_done(object_id=self.client.object.id,
                                     method=self.client.config.contacts.detection_method,
                                     contact_threshold=self.client.config.contacts.contact_threshold)
                if r != 0:
                    break
                self.client.step_simulation()
            return r
        return 0

    def motion_done(self, eps: float = 1e-2, object_id: int = -1, method: str = "force", contact_threshold: float = 10) -> int:
        """
        Checks whether gripper is in goal position

        :param eps: tolerance around the commanded joint value
        :type eps: float, optional, default=1e-2
        :param object_id: kept for API compatibility with other detectors
        :type object_id: int, optional, default=-1
        :param method: kept for API compatibility with other detectors
        :type method: str, optional, default="force"
        :param contact_threshold: torque threshold used to classify object contact
        :type contact_threshold: float, optional, default=10
        :return: 1 for normal closure, 2 for collision, 3 for timeout, 0 for no closure
        :rtype: int
        """

        if self.timeout == -1:
            self.timeout = self.client.steps_done
            return 0
        elif self.client.steps_done - self.timeout < 5:  # sow start of the gripper jaws
            return 0


        if self.client.steps_done - self.timeout > self.client.config.contacts.time_step*2:  # do one second of simulated time only
            self.timeout = -1
            return 3


        for finger_joint_id, finger_joint_name in zip(self.finger_joint_ids, self.finger_joints_names):
            if finger_joint_name == "bh_j11_joint":
                continue
            info = self.client.getJointState(self.id, finger_joint_id)
            if np.abs(info[0] - self.current_goal) < eps:
                self.timeout = -1
                return 1
            if info[-1] < self.max_force and np.abs(info[-1]) > contact_threshold:
                self.timeout = -1
                return 2
        return 0

    def move(self, joint: int, target: float, velocity: float = 0.1) -> int:
        """
        Move joint to target position

        :param joint: id of the joint
        :type joint: int
        :param target: target position
        :type target: float
        :param velocity: maximum joint speed during the move command
        :type velocity: float, optional, default=0.1
        :return: 0 when target is reached, -1 when timeout is reached
        :rtype: int
        """
        self.client.setJointMotorControl2(self.id, joint, self.client.POSITION_CONTROL, targetPosition=target,
                                          maxVelocity=velocity, force=self.max_force)

        start_steps = self.client.steps_done
        while True:
            position = self.client.getJointState(self.id, joint)[0]
            if np.abs(position - target) < 1e-3:
                return 0
            if self.client.steps_done - start_steps > self.client.config.contacts.time_step*2:
                return -1
            self.client.step_simulation()
