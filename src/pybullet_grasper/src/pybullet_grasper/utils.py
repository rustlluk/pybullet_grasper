"""
Small utils

:Author: Lukas Rustler
"""
import logging
import yaml
from collections import namedtuple
import os
import sys
from typing import Any
from contextlib import contextmanager


class Logger:
    """
    Class to handle logging for both ROS and non-ROS behaviours
    """
    def __init__(self, ros: bool, debug: bool = False) -> None:
        """
        Initializes logger backend for ROS or standard Python logging.

        :param ros: whether ROS logging API should be used
        :type ros: bool
        :param debug: enables debug-level logs
        :type debug: bool, optional, default=False
        :return: None
        :rtype: None
        """
        # Get info whether we are using ros and whether to show debug info
        self.ros = ros
        if self.ros:
            import rospy
            self.rospy = rospy
        self.debug = debug

        # init classic python logger
        if not self.ros:
            logging.basicConfig(level=logging.DEBUG if self.debug else logging.INFO,
                                format='[%(levelname)s] [%(asctime)s]: %(message)s',
                                force=True)  # force is needed to overwrite settings from other libraries

    def log_info(self, msg: Any) -> None:
        """
        Logs an info message.

        :param msg: message payload
        :type msg: Any
        :return: None
        :rtype: None
        """

        if self.ros:
            self.rospy.loginfo(msg)
        else:
            logging.info(msg)

    def log_debug(self, msg: Any) -> None:
        """
        Logs a debug message.

        :param msg: message payload
        :type msg: Any
        :return: None
        :rtype: None
        """

        if self.ros:  # #!#§! ROS cant print debug to stdout, so use INFO with DEBUg "decorator"
            if self.debug:
                self.log_info("DEBUG: "+str(msg))
        else:
            logging.debug(msg)

    def log_warning(self, msg: Any) -> None:
        """
        Logs a warning message.

        :param msg: message payload
        :type msg: Any
        :return: None
        :rtype: None
        """

        if self.ros:
            self.rospy.logwarn(msg)
        else:
            logging.warning(msg)

    def log_error(self, msg: Any) -> None:
        """
        Logs an error message.

        :param msg: message payload
        :type msg: Any
        :return: None
        :rtype: None
        """

        if self.ros:
            self.rospy.logerr(msg)
        else:
            logging.error(msg)

    def log_critical(self, msg: Any) -> None:
        """
        Logs a critical/fatal message.

        :param msg: message payload
        :type msg: Any
        :return: None
        :rtype: None
        """

        if self.ros:
            self.rospy.logfatal(msg)
        else:
            logging.critical(msg)


class Config:
    """
    Class to parse and keep the config loaded from yaml file
    """
    def __init__(self, config_path: str) -> None:
        """
        Loads YAML config and exposes keys as object attributes.

        :param config_path: path to YAML config file
        :type config_path: str
        :return: None
        :rtype: None
        """
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        for attr, value in config_dict.items():
            self.set_attribute(attr, value, self)

    def set_attribute(self, attr: str, value: Any, reference: Any) -> int:
        """
        Function to recursively fill the instance variables from dictionary. When value is non-dict, it is directly
        assigned to a variable. Else, the dict is recursively parsed.
        :param attr: name of the attribute
        :type attr: str
        :param value: value of the attribute
        :type value: str, float, int, dict, list, ... - and other that can be loaded from yaml
        :param reference: reference to the parent class. "self" for the upper attributes, pointer to namedtuple for
                          inner attributes
        :type reference: pointer or whatever it is called in Python
        :return: 0
        :rtype: int
        """

        # Parse non-dict directly to the attribute
        if not isinstance(value, dict):
            setattr(reference, attr, value)
            return 0
        # prepare named tuple for the dict attribute and populate in recursively
        else:
            setattr(reference, attr, namedtuple(attr, list(value.keys())))
            for inner_attr, inner_value in value.items():
                self.set_attribute(inner_attr, inner_value, getattr(reference, attr))
        return 0


class Point:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


class Quaternion:
    def __init__(self, x: float, y: float, z: float, w: float) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.w = w


class Pose:
    def __init__(self, position: Point, orientation: Quaternion) -> None:
        self.position = position
        self.orientation = orientation


class GenerateGraspsResponse:
    def __init__(self) -> None:
        self.poses = []
        self.qualities = []
        self.poses_after_grasps = []


class MeshInfo:
    def __init__(self) -> None:
        self.pos = []
        self.ori = []
        self.colors = []
        self.paths = []


def get_data_folder() -> str:
    if os.path.isfile(os.path.join(os.path.realpath(os.path.dirname(__file__)), "../..", "urdf/plane.urdf")):
        return os.path.join(os.path.realpath(os.path.dirname(__file__)), "../..")
    else:
        return os.path.join(sys.prefix, 'share', 'pybullet_grasper')


@contextmanager
def suppress_c_output():
    """Context manager to suppress C/C++ level stdout and stderr."""
    # Open the null device
    devnull = os.open(os.devnull, os.O_WRONLY)

    # Save the original file descriptors
    old_stdout_fd = os.dup(sys.stdout.fileno())
    old_stderr_fd = os.dup(sys.stderr.fileno())

    try:
        # Overwrite the standard file descriptors with the null device
        os.dup2(devnull, sys.stdout.fileno())
        os.dup2(devnull, sys.stderr.fileno())
        yield
    finally:
        # Restore the original file descriptors
        os.dup2(old_stdout_fd, sys.stdout.fileno())
        os.dup2(old_stderr_fd, sys.stderr.fileno())

        # Clean up by closing the duplicated descriptors
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)
        os.close(devnull)