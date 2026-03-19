# Grasping with PyBullet
This package provides an interface for grasp planning using PyBullet. It includes a grasp planner that can 
generate grasp candidates for a given object.

![img](https://rustlluk.github.io/pybullet_grasper/documentation/_images/img.png)

## Installation
### Pure Python
  - You can install the package using pip: `python3 -m pip install pybullet_grasper`
    - when using python3.8 you might need to install upgrade pip and install development version of open3d  
  - Or clone this repository `git clone https://github.com/rustlluk/pybullet_grasper.git`; go to src/pybullet_grasper and install it locally: `python3 -m pip install -e .`  

### ROS
 - You can install the package in ROS1 Noetic workspace
   - Clone the repository or put the pybullet_grasper folder (inside src of this repo) to src of your ROS workspace
     - should be as `catkin_ws/src/pybullet_grasper`
       - this includes setup.py, src, objects, etc. 
     - `catkin_build` it
 - See the branch [ShapeGrasp repo](https://github.com/rustlluk/ShapeGrasp) for Dockerfile and Docker installation instructions

## Usage
### Pure Python
  - The easier option is to either:
    - **CLI** in command line run `python3 -m pybullet_grasper.grasper_main` to see a demo of the grasp planner. This
      will print the results (if debug enabled) and/or save them to a file (if save_path provided)
      - possible command line options:
        - `--object_name/-o`: path to the .obj file
          - can be a relative path with respect to current folder, an absolute path, or a path related to installed path
            and be in pybullet_grasper/objects (when installed through pip it will be in 
            sys.prefix/share/pybullet_grasper/objects)
        - `--config_name/-c`: path to .yaml config file
          - can be a relative path with respect to current folder, an absolute path, or a path related to installed path
            and be in pybullet_grasper/configs (when installed through pip it will be in 
            sys.prefix/share/pybullet_grasper/config)
        - `--ros/-r`: whether to use ROS or not
          - if True, it will run the grasper as ROS node
          - requires roscore running and rospy installed
        - `--service/-s`: call the ROS service
          - requires node with rosservice running 
          - here the object and config paths must be absolute, or the files must be located in ws/src/pybullet_grasper/configs and ..../objects, respectively
        - `--save_path/-sp`: path where to save the generated grasps; str or None
          - will be saved as file with values seperated by ";", so it is recommended to use .csv format
          - if None -> nothing will be saved
    - **API** you can import the package main as `from pybullet_grasper.grasper_main import main` and call it
      `result = main(object_name, config_name, ros=False, service=False)`, where:
      -  object_name: path to the .obj file
          - can be a relative path with respect to current folder, an absolute path, or a path related to installed path
            and be in pybullet_grasper/objects (when installed through pip it will be in 
            sys.prefix/share/pybullet_grasper/objects)
      - config_name: path to .yaml config file
          - can be a relative path with respect to current folder, an absolute path, or a path related to installed path
            and be in pybullet_grasper/configs (when installed through pip it will be in 
            sys.prefix/share/pybullet_grasper/config)
      - ros: boolean, whether to run as a ROS node or not
      - service: boolean, whether to call as ROS service or not
      - save_path: str or None; path where to save the generated grasps
          - will be saved as file with values seperated by ";", so it is recommended to use .csv format
          - if None -> nothing will be saved
      - result: pybullet_grasper.utils.GenerateGraspsResponse object
        - see [utils.py](https://raw.githubusercontent.com/rustlluk/pybullet_grasper/refs/main/src/pybullet_grasper/utils.py) for details on the response object and its attributes
      - see [examples/example.py](https://raw.githubusercontent.com/rustlluk/pybullet_grasper/refs/main/examples/example.py) for usage
    - **Low-level API**  you can import directly `from pybullet_grasper.grasp_generator import GraspGenerator`
      - see [grasper_main.py](https://raw.githubusercontent.com/rustlluk/pybullet_grasper/refs/main/src/pybullet_grasper/grasper_main.py) for usage

### ROS
 - When you have ROS installed and the package installed, you can:
   - **ROS Node** run the script as node (works the same as pure python basically, but uses ROS messages etc.)
     - use either the CLI or API options described above, but with `--ros/-r` (in CLI) or ros argument (API) set to True
       - see also [grasper_main.py](https://raw.githubusercontent.com/rustlluk/pybullet_grasper/refs/main/src/pybullet_grasper/grasper_main.py) for low-level usage
     - roscore must be running for this
   - **ROS Service** call the ROS service to get grasp candidates
     - to just test it, you can use the CLI (with `--service/-s` option) or API (with `service=True` argument) options described above
     - roscore and [grasp_generator_service.py](https://raw.githubusercontent.com/rustlluk/pybullet_grasper/refs/main/src/pybullet_grasper/grasp_generator_service.py) node must be running for this
       - you can run  `roslaunch pybullet_grasper main.launch` to run the roscore and service
     - Otherwise, the [service file](https://raw.githubusercontent.com/rustlluk/pybullet_grasper/refs/main/src/pybullet_grasper/srv/GenerateGrasps.srv) is defined as:
       ```     
       std_msgs/String object_name
       std_msgs/String[] grasp_type
       float32[] init_position
       ---
       geometry_msgs/Pose[] poses
       float32[] qualities
       float32[] poses_after_grasps
       ```
       and can be called as (see also [example_service.py](https://raw.githubusercontent.com/rustlluk/pybullet_grasper/refs/main/examples/example_service.py)):
       ```
       import rospy
       from pybullet_grasper.srv import GenerateGrasps, GenerateGraspsRequest


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
           result = generate_grasps.call(request)
       ```
       or from the command line as:
       ```
       rosservice call /generate_grasps "{object_name: {data: 'example.obj'}, grasp_type: [{data: 'example.yaml'}], init_position: [0.0, 0.0, 0.0]}"
       ```  
       
## Config files
The pacakge uses .yaml config files to specify both parameters of the grasp planner and visualization/debug options.
The default one can be found in [src/pybullet_grasper/configs/default.yaml](https://raw.githubusercontent.com/rustlluk/pybullet_grasper/refs/main/src/pybullet_grasper/configs/default.yaml)
The main things to change are:
  - gripper: for now "robotiq" or "barrett"
    - users can add new grippers adding the API in a similar form to the existing ones, available in
      [bullet_classes.py](https://raw.githubusercontent.com/rustlluk/pybullet_grasper/refs/main/src/pybullet_grasper/src/pybullet_grasper/bullet_classes.py)   
  - debug: boolean; when True additional prints are shown. Also neccessary when visualization.show_last enabled
  - visualization.show_last: boolean; when True, the best generated grasp candidate is visualized. 
    - Requires debug to be True as well.
  - visualization.debug_sleep: float; when show_last is enabled, it will most probably run super fast. debug_sleep can
    make the rendering slower (0.01) is a good value on most machines
  - analytical.enabled: boolean; when True, friction cones and GWS are computed (with epsilon-quality and GWS volume) and
    can be shown in the visualizer if show_last is enabled
    - very preliminary version, will most like not work every time 

## Input Meshes and Processing
 - the input meshes must be in .obj format (Wavefront OBJ format). You can use third-party libraries like [Meshlab](https://www.meshlab.net/),
   [open3D](http://www.open3d.org/), or [Blender](https://www.blender.org/) to convert your meshes to .obj format if they are in a different format.
 - We further employ the [VHACD](https://github.com/Unity-Technologies/VHACD) library to perform convex decomposition of
   the meshes. This steps takes some time in the beginning. But it is necessary for detection collision of object and gripper properly  

## License

[![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

See licences of the used libraries like PyBullet, Open3D, etc. for more details.

## Citing
When used, please cite this repository as follows:
```
@inproceedings{rustler2026ShapeGrasp,
      title={},
      author={},
      year={2026},
      booktitle={},
      organization={}}
```