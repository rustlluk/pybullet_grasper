"""
Visualzation class

:Author: Lukas Rustler
"""
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import numpy as np
from scipy.spatial.transform import Rotation as ts
from pybullet_grasper.utils import MeshInfo
from typing import Any, Optional, List, Union


Vector3 = Union[List[float], np.ndarray]


class Visualizer:
    """Class to help with custom rendering"""

    def __init__(self, client: Optional[Any] = None, visible: int = 0) -> None:
        """
        Initializes Open3D-based visualization utilities.

        :param client: active simulation client used to query geometry states
        :type client: Any, optional, default=None
        :param visible: 1 to show UI window, 0 for headless render loop
        :type visible: int, optional, default=0
        :return: None
        :rtype: None
        """
        self.client = client
        self.meshes = {}
        self.visible = visible
        self.is_alive = True
        self.rendered_geometries = []
        self.mat = None
        self.img_counter = 0
        self.file_dir = os.path.dirname(os.path.abspath(__file__))
        self.last_msg = -1
        self.gws_vis = {}
        self.last_image = None
        self.last_key = None

        # create the instance, window and OpenDScene widget to visualize geometries
        self.gui = gui.Application.instance
        self.gui.initialize()
        self.window = gui.Application.instance.create_window("BulletGrasper",
                                                             self.client.config.visualization.resolution[0],
                                                             self.client.config.visualization.resolution[1])
        if not self.visible:
            self.window.show(False)

        self.window.set_on_key(self._on_key_event)

        self.gui.menubar = gui.Menu()
        self.menu = self.gui.menubar
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)

        self.vis = self.scene.scene
        self.scene.background_color = o3d.visualization.gui.Color(1, 1, 1, 1)
        self.vis.set_background([1, 1, 1, 1])
        self.vis.set_lighting(self.vis.NO_SHADOWS, [0, 0, 0])

        # prepare default material
        self.mat = rendering.MaterialRecord()
        self.mat.shader = 'defaultLit'

        self.show_menu = gui.Menu()
        for show_idx, show_obj, is_shown in zip(np.arange(0, 7),
                                                ["Gripper", "Object", "Cones", "Torques", "Normals", "Axes", "GWS"],
                                                [True, True, False, False, False, False, False]):
            self.show_menu.add_item(show_obj, show_idx)
            self.window.set_on_menu_item_activated(show_idx, self.KeyCallback(show_obj.lower(), self, show_idx))
            self.show_menu.set_checked(show_idx, is_shown)
            self.show_menu.set_enabled(show_idx, is_shown)
        self.menu.add_menu("Show", self.show_menu)
        self.show_menu.set_enabled(5, True) # Axes

        # add to window
        self.window.add_child(self.scene)

        self.show_mesh()

        # compute center of all objects
        scene_mesh = o3d.geometry.TriangleMesh()
        for f_path, m in self.meshes.items():
            if "plane.obj" not in f_path and "pc" not in f_path:
                scene_mesh += m
        bbox = scene_mesh.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        # look at the center
        self.scene.look_at(center, center+[-0.4, 0, 0], [0, 0, 1])
        self.scene.center_of_rotation = center

    def _on_key_event(self, event: gui.KeyEvent) -> bool:
        """
        Callback to handle keyboard events in the Open3D window.
        """
        # We usually only want to trigger an action when the key is pressed DOWN
        if event.type == gui.KeyEvent.DOWN:
            # Example: Pressing 'Q' to quit
            if event.key == gui.KeyName.Q:
                self.last_key = "q"
                return True

        # Return False if the event wasn't handled here,
        # allowing default Open3D behaviors (like camera movement) to still work.
        return False

    def show_mesh(self, obj_type: str = "") -> None:
        """
        Function to parse message with information to insert objects to window
        :param obj_type: which type of objects is being shown - gripper, object
        :type obj_type: str, optional, default=""
        :return: None
        :rtype: None
        """
        self.client.msg = MeshInfo()
        for idx in self.client.visualization_objects:
            self.client.read_info(idx)
        for mesh_id in range(len(self.client.msg.pos) // 3):
            # get correct values for given mesh
            pos = self.client.msg.pos[mesh_id * 3:(mesh_id + 1) * 3]
            ori = self.client.msg.ori[mesh_id * 4:(mesh_id + 1) * 4]
            col = self.client.msg.colors[mesh_id * 3:(mesh_id + 1) * 3]
            f_path = self.client.msg.paths[mesh_id]

            # check the type based on the information whether the files are located in grippers folder
            if "grippers" in f_path:
                mesh_type = "gripper"
            elif "plane" in f_path:
                mesh_type = "plane"
            else:
                mesh_type = "object"
            # skin when types do not agree
            if obj_type != "" and obj_type != mesh_type:
                continue

            if f_path not in self.meshes:
                self.meshes[f_path] = o3d.io.read_triangle_mesh(f_path)
                self.meshes[f_path].paint_uniform_color(col[:3])

                # Just for visualization
                if not self.meshes[f_path].has_triangle_normals():
                    self.meshes[f_path].compute_triangle_normals()
                if not self.meshes[f_path].has_vertex_normals():
                    self.meshes[f_path].compute_vertex_normals()

            if mesh_type + "_" + str(mesh_id) not in self.rendered_geometries:
                self.vis.add_geometry(mesh_type + "_" + str(mesh_id), geometry=self.meshes[f_path], material=self.mat)
                self.rendered_geometries.append(mesh_type + "_" + str(mesh_id))

            # get ori and position as 4x4 transformation matrix
            R = np.eye(4)
            R[:3, :3] = ts.from_quat(ori).as_matrix()
            R[:3, 3] = pos

            self.vis.set_geometry_transform(mesh_type + "_" + str(mesh_id), R)

    def render(self) -> None:
        """
        Interactive render
        :return: None
        :rtype: None
        """

        self.show_mesh()

        self.window.post_redraw()
        if not self.gui.run_one_tick():
            self.is_alive = False
            self.gui.quit()

        if not self.visible:
            img_path = os.path.join(self.client.data_folder, "imgs", "img_" + str(self.img_counter) + ".png")
            self.vis.scene.render_to_image(self.save_image)
            if self.last_image is not None:
                o3d.io.write_image(img_path, self.last_image)
                self.img_counter += 1

    def save_image(self, im: Any) -> None:
        """
        Stores the last rendered image from asynchronous Open3D callback.

        :param im: rendered image instance provided by Open3D
        :type im: Any
        :return: None
        :rtype: None
        """
        self.last_image = im

    def draw_normal_forces(self, points: np.ndarray, normals: np.ndarray, world_origin: Vector3 = [0, 0, 0]) -> None:
        """
        Draws line segments for contact normals.

        :param points: contact point positions
        :type points: np.ndarray
        :param normals: normal vectors at contact points
        :type normals: np.ndarray
        :param world_origin: origin offset applied to rendered vectors
        :type world_origin: list[float] | np.ndarray, optional, default=[0, 0, 0]
        :return: None
        :rtype: None
        """
        ls = o3d.geometry.LineSet()
        points = np.vstack((points+world_origin, points+normals+world_origin))
        lines = np.hstack((np.arange(0, len(points) // 2).reshape(-1, 1),
                           np.arange(len(points) // 2, len(points)).reshape(-1, 1)))
        ls.points = o3d.utility.Vector3dVector(points)
        ls.lines = o3d.utility.Vector2iVector(lines)
        indexes = np.arange(0, normals.shape[0])
        colors = self.client.grasper.rgb_colors[indexes]
        ls.colors = o3d.utility.Vector3dVector(colors)
        self.vis.add_geometry(name="normals", geometry=ls, material=self.mat)
        self.vis.show_geometry("normals", False)
        self.rendered_geometries.append("normals")
        self.show_menu.set_enabled(4, True) # Normals

    def draw_force_torques(self, points: np.ndarray, forces: np.ndarray, torques: np.ndarray,
                           cone_vertices: int, magnitudes: np.ndarray, world_origin: Vector3 = [0, 0, 0]) -> None:
        """
        Draws friction cones and torque vectors for currently active contacts.

        :param points: contact point positions
        :type points: np.ndarray
        :param forces: discretized friction cone forces
        :type forces: np.ndarray
        :param torques: corresponding torques for plotted forces
        :type torques: np.ndarray
        :param cone_vertices: number of vertices per friction cone approximation
        :type cone_vertices: int
        :param magnitudes: normal force magnitudes used for cone scaling
        :type magnitudes: np.ndarray
        :param world_origin: origin offset applied to rendered vectors
        :type world_origin: list[float] | np.ndarray, optional, default=[0, 0, 0]
        :return: None
        :rtype: None
        """

        # create LineSet
        ls = o3d.geometry.LineSet()

        torque_origins = np.tile(world_origin, (points.shape[0]*cone_vertices, 1))
        # Stack points and point with normals
        torques = np.vstack((torque_origins, torque_origins + torques))

        # Make lines between individual points and points with normals
        lines = np.hstack((np.arange(0, len(torques) // 2).reshape(-1, 1),
                           np.arange(len(torques) // 2, len(torques)).reshape(-1, 1)))
        ls.points = o3d.utility.Vector3dVector(torques)
        ls.lines = o3d.utility.Vector2iVector(lines)

        # Paint by different color
        indexes = np.repeat(np.arange(0, lines.shape[0] // cone_vertices), cone_vertices)
        colors = self.client.grasper.rgb_colors[indexes]
        ls.colors = o3d.utility.Vector3dVector(colors)

        self.vis.add_geometry(name="torques", geometry=ls, material=self.mat)
        self.vis.show_geometry("torques", False)
        self.rendered_geometries.append("torques")
        self.show_menu.set_enabled(3, True)

        scale = 0.1/np.max(magnitudes)
        for point_id, point in enumerate(points):
            cone_forces = forces[point_id*cone_vertices:(point_id+1)*cone_vertices]
            # cone_forces = (cone_forces-point)/5000 + point + world_origin
            cone_forces *= magnitudes[point_id]*scale
            cone_forces += point + world_origin

            cone_points = np.vstack((point+world_origin, cone_forces))
            pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cone_points))
            mesh, _ = pc.compute_convex_hull()
            mesh.orient_triangles()
            mesh.compute_triangle_normals()
            mesh.paint_uniform_color(self.client.grasper.rgb_colors[point_id])
            self.vis.add_geometry(name="cones_"+str(point_id), geometry=mesh, material=self.mat)
            self.vis.show_geometry("cones_"+str(point_id), False)
            self.vis.add_geometry(name="cones_pc_"+str(point_id), geometry=pc, material=self.mat)
            self.vis.show_geometry("cones_pc_"+str(point_id), False)
            self.rendered_geometries.append("cones_"+str(point_id))
            self.rendered_geometries.append("cones_pc_"+str(point_id))
        self.show_menu.set_enabled(2, True)

    def draw_gws(self) -> None:
        """
        Opens per-hull windows with grasp wrench space meshes.

        :return: None
        :rtype: None
        """

        for gws_type, gws_mesh in self.client.grasper.gws_meshes.items():
            window = gui.Application.instance.create_window(gws_type, 640, 480)

            scene = gui.SceneWidget()
            scene.scene = rendering.Open3DScene(window.renderer)
            vis = scene.scene
            scene.background_color = o3d.visualization.gui.Color(1, 1, 1, 1)
            scene.frame = gui.Rect(0, 0, 640, 480)
            vis.set_background([1, 1, 1, 1])
            vis.set_lighting(self.vis.NO_SHADOWS, [0, 0, 0])
            vis.add_geometry(name=gws_type, geometry=gws_mesh, material=self.mat)

            window.add_child(scene)
            window.show_menu(False)

            window.set_on_close(self.OnCloseGWS(gws_type, self))
            self.gws_vis[gws_type] = (window, scene, gws_type)

    class OnCloseGWS:
        """
        Help class delete information about the given GWS window from visualizer
        """
        def __init__(self, gws_type: str, parent: "Visualizer") -> None:
            """
            Init
            :param gws_type: type of GWS, by default something like L1_fixed_force; used as key to dictionary
            :type gws_type: str
            :param parent: pointer to parent instance
            :type parent: Visualizer
            :return: None
            :rtype: None
            """
            self.gws_type = gws_type
            self.parent = parent

        def __call__(self) -> bool:
            """
            When callback to close window is received, delete the thing from dictionary
            :return: True so Open3D finalizes window close
            :rtype: bool
            """
            del self.parent.gws_vis[self.gws_type]
            if self.parent.gws_vis == {}:
                self.parent.menu.set_checked(6, False)
            # return True so the window is really closed
            return True

    class KeyCallback:
        """
        Help function to unify key callbacks for toggling thingies in the windows
        """
        def __init__(self, geom_type: str, parent: Any, menu_id: int) -> None:
            """
            Init
            :param geom_type: name of geometric type: cone, object, gripper
            :type geom_type: str
            :param parent: pointer to the parent instance
            :type parent: Any
            :param menu_id: menu item identifier in Open3D menu
            :type menu_id: int
            :return: None
            :rtype: None
            """
            self.geom_type = geom_type
            self.parent = parent
            self.menu_id = menu_id

        def __call__(self) -> int:
            """
            Does something when callback is called

            :return: 0 on handled action
            :rtype: int
            """

            show = not self.parent.menu.is_checked(self.menu_id)
            self.parent.menu.set_checked(self.menu_id, show)
            # if axes button, show them and return
            if self.geom_type == "axes":
                if isinstance(self.parent, Visualizer):
                    self.parent.vis.show_axes(show)
                else:
                    self.parent.show_axes(show)
                return 0
            elif self.geom_type == "gws":
                if self.parent.client.grasper.gws_meshes == {}:
                    for hull_type in self.parent.client.grasper.hulls:
                        self.parent.client.grasper.compute_gws_mesh([0, 1, 2], hull_type+"_fixed_force")
                        self.parent.client.grasper.compute_gws_mesh([3, 4, 5], hull_type+"_fixed_torque")
                if show:
                    self.parent.client.visualizer.draw_gws()
                else:
                    windows = [window for window, *_ in self.parent.client.visualizer.gws_vis.values()]
                    for window in windows:
                        window.close()
                    del windows

            # iterate over all geometries
            for geom in self.parent.rendered_geometries:
                # if the geometry is one of the needed types show/hide it
                if self.geom_type in geom:
                    if self.geom_type in ["cones", "torques", "normals"]:
                        pos, ori = self.parent.client.getBasePositionAndOrientation(self.parent.client.object.id)
                        R = np.eye(4)
                        R[:3, :3] = np.reshape(self.parent.client.getMatrixFromQuaternion(ori), (3, 3))
                        R[:3, 3] = pos
                        R = R @ np.linalg.inv(self.parent.client.grasper.pose)
                        self.parent.vis.set_geometry_transform(geom, R)
                    self.parent.vis.show_geometry(geom, show)
            return 0
