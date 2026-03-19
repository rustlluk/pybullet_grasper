#!/usr/bin/env python3
"""
Main CLI/API file

:Author: Lukas Rustler
"""

import signal
from pybullet_grasper.grasp_generator import signal_handler
from pybullet_grasper.utils import GenerateGraspsResponse
import argparse
import time
import numpy as np
from typing import Tuple, Union


def prepare_parser() -> Tuple[bool, bool, str, str, Union[str, None]]:
    """
    Parses CLI arguments for ROS/service/object/config options.

    :return: tuple with ROS flag, service flag, config name, and object name
    :rtype: tuple[bool, bool, str, str]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--ros", action="store_true")
    parser.add_argument("-s", "--service", action="store_true")
    parser.add_argument("-c", "--config_name", default="default.yaml")
    parser.add_argument("-o", "--object_name", default="object_0.obj")
    parser.add_argument("-sp","--save_path", default=None)
    args = parser.parse_args()
    return args.ros, args.service, args.config_name, args.object_name, args.save_path


def main(config_name: str, object_name: str, ROS: bool = False, service: bool = False, save_path: Union[str, None] = None) -> Union[None, GenerateGraspsResponse]:
    """
    Runs the grasp generator either through ROS service client or direct API call.

    :param config_name: configuration file name or path
    :type config_name: str
    :param object_name: object mesh name or path
    :type object_name: str
    :param ROS: whether ROS node mode is enabled
    :type ROS: bool, optional, default=False
    :param service: whether to call the ROS service instead of direct execution
    :type service: bool, optional, default=False
    :param save_path: path where to save grasp result. If None -> do not save
    :type save_path: str | None, optional, default=None
    :return: result
    :rtype: None | GenerateGraspsResponse
    """

    result = None
    if service:
        import rospy
        from pybullet_grasper.srv import GenerateGrasps, GenerateGraspsRequest
        rospy.init_node("grasp_generator_node")
        rospy.wait_for_service("generate_grasps")
        rospy.set_param("config_name", config_name)

        generate_grasps = rospy.ServiceProxy("generate_grasps", GenerateGrasps)

        request = GenerateGraspsRequest()
        request.object_name.data = object_name
        request.init_position = [0, 0, 0]
        result = generate_grasps.call(request)
    else:
        from pybullet_grasper.grasp_generator import GraspGenerator
        signal.signal(signal.SIGINT, signal_handler)

        if ROS:
            import rospy
            rospy.init_node("grasp_generator_node")

        t = time.time()
        g = GraspGenerator(object_name, init_position=[0, 0, 0], config_name=config_name, ros=ROS)
        result = g.run()

        if g.client.debug:
            g.client.logger.log_debug(
                f"{np.count_nonzero(result.qualities != -1)} grasps found from {len(g.gripper_poses)} poses")
            g.client.logger.log_debug(f"{np.count_nonzero(result.qualities > 1)} grasps with quality better than  1 found")
            g.client.logger.log_debug(f"Time: {time.time() - t}")
            g.client.logger.log_debug(f"Best Score: {result.qualities[0]}")
            g.client.logger.log_debug(f"Pose of best grasp: {result.poses_after_grasps[:7]}")

            if g.client.config.visualization.show_last:
                p = result.poses[0]
                g.run(p)

    if save_path is not None:
        with open(save_path, "w") as f:
            f.write("quality;position;orientation\n")
            for q, p in zip(result.qualities, result.poses):
                f.write(f'{q};{[p.position.__getattribute__(_) for _ in ["x", "y", "z"]]};{[p.orientation.__getattribute__(_) for _ in ["x", "y", "z", "w"]]}\n')

    return result


if __name__ == "__main__":
    ROS, service, config_name, object_name, save_path = prepare_parser()
    main(config_name, object_name, ROS, service, save_path)