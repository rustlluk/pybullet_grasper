#!/usr/bin/env python3
"""
Service file

:Author: Lukas Rustler
"""
import rospy
from pybullet_grasper.srv import GenerateGrasps, GenerateGraspsResponse, GenerateGraspsRequest
from pybullet_grasper.grasp_generator import GraspGenerator
import numpy as np
import time


def generate_grasps(request: GenerateGraspsRequest) -> GenerateGraspsResponse:
    """
    ROS service callback that runs grasp generation for requested object.

    :param request: incoming ROS service request
    :type request: GenerateGraspsRequest
    :return: generated grasps response
    :rtype: GenerateGraspsResponse
    """
    config_name = rospy.get_param("config_name", "default.yaml")

    g = GraspGenerator(request.object_name.data, request.init_position, config_name, ros=True)
    debug = g.client.debug
    if debug:
        t = time.time()
    response = g.run()

    if debug:
        qualities = np.array(response.qualities)
        g.client.logger.log_debug(f"{np.count_nonzero(qualities != -1)} grasps found from {len(g.gripper_poses)} poses")
        g.client.logger.log_debug(f"{np.count_nonzero(qualities > 1)} grasps with quality better than  1 found")
        g.client.logger.log_debug(f"Time: {time.time() - t}")
        g.client.logger.log_debug(f"Best Score: {qualities[0]}")
        if g.client.config.visualization.show_last:
            g.run(response.poses[0])
        del g.client.visualizer
    return response


if __name__ == "__main__":
    rospy.init_node("grasp_generator_service_node")

    rospy.Service('generate_grasps', GenerateGrasps, generate_grasps)
    rospy.loginfo("Grasp generator successfully started")
    rospy.spin()
