from pybullet_grasper.grasper_main import main
import os
import time
import numpy as np


if __name__ == '__main__':
    file_dir = os.path.realpath(os.path.dirname(__file__))

    # The path can be absolute or relative
    config_name = "example.yaml"
    object_name = "example.obj"
    save_path = "example.csv"

    start_time = time.time()
    result = main(config_name, object_name, save_path=save_path)
    end_time = time.time()
    print(f"Found {np.count_nonzero(result.qualities > 0)} grasp with quality better than 0 in {end_time - start_time} seconds")