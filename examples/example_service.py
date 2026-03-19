import rospy
from pybullet_grasper.srv import GenerateGrasps, GenerateGraspsRequest
import numpy as np
import time


if __name__ == "__main__":
    # Here the path must be absolute or the .yaml must be in ws/src/pybullet_grasper/configs and .obj in ws/src/pybullet_grasper/objects
    config_name = "example.yaml"
    object_name = "example.obj"

    rospy.init_node("grasp_generator_node")

    rospy.set_param("config_name", config_name)

    rospy.wait_for_service("generate_grasps")
    generate_grasps = rospy.ServiceProxy("generate_grasps", GenerateGrasps)

    request = GenerateGraspsRequest()
    request.object_name.data = object_name
    request.init_position = [0, 0, 0]
    start_time = time.time()
    result = generate_grasps.call(request)
    end_time = time.time()
    print(f"Found {np.count_nonzero(np.array(result.qualities) > 0)} grasp with quality better than 0 in {end_time - start_time} seconds")