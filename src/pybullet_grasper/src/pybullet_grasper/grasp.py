#!/usr/bin/env python3
"""
Main class for analytic metrics

:Author: Lukas Rustler
"""

import copy
import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection
from sklearn.cluster import DBSCAN
import matplotlib as mpl
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.optimize import linprog
from scipy.linalg import null_space
from typing import Any, Optional, Tuple, List, Dict


class Grasp:
    """
    Class to encapsulate functions regarding grasp quality.
    Based on: https://ieeexplore.ieee.org/document/1371616/ and https://ieeexplore.ieee.org/document/772531/
    """
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    rgb_colors = np.array([mpl.colors.to_rgb(c) for c in colors])

    def __init__(self, client: Any) -> None:
        self.client = client
        self.gws_meshes = {}
        self.hulls = {}
        self.pose = None

    def get_contact_info(self, object_id: int, torque_origin: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Obtain contact information and divide it into arrays
        :param object_id: id of the grasped object
        :type object_id: int
        :return: contact points, contact normals, contact forces, mutual frictions
        :rtype: tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]
        """

        # get contact from PyBullet
        contacts = self.client.getContactPoints(object_id, self.client.gripper.id)
        if len(contacts) == 0:
            return None, None, None, None

        # prepare lists
        normals = []
        points = []
        forces = []
        frictions = []

        # get normals, points, forces, lateral frictions
        for contact_id, contact in enumerate(contacts):
            force = contact[self.client.contactPoints["FORCE"]]
            self.client.logger.log_debug(f"Force for contact {contact_id}: {force}")
            if force > self.client.config.analytical.force_threshold:  # sometimes the force is too low -> wrong contact
                normals.append(contact[self.client.contactPoints["NORMAL"]])
                points.append(contact[self.client.contactPoints["POSITIONB"]]-torque_origin)
                forces.append(force)

                # mutual friction = friction1*friction2
                friction = self.client.getDynamicsInfo(object_id, -1)[self.client.dynamicsInfo["FRICTION"]] * \
                           self.client.getDynamicsInfo(self.client.gripper.id, contact[self.client.contactPoints["INDEXB"]])[self.client.dynamicsInfo["FRICTION"]]
                frictions.append(friction)

        # make them numpy
        normals = np.array(normals)
        points = np.array(points)
        forces = np.array(forces)
        frictions = np.array(frictions)

        # cluster points to get rid of some unwanted, e.g., close points with similar normal
        dbscan_angle = DBSCAN(min_samples=1, eps=0.001, metric="cosine").fit(normals)
        dbscan_distance = DBSCAN(min_samples=1, eps=0.005).fit(points)

        final_indexes = []
        for cl in np.unique(dbscan_distance.labels_):
            temp_idx = dbscan_distance.labels_ == cl
            if np.all(dbscan_angle.labels_[temp_idx] == dbscan_angle.labels_[temp_idx][0]):
                p_temp = points[temp_idx]
                # TODO: check if taking the maximum force one makes sense
                max_force_idx = np.argmax(forces[temp_idx])
                if len(p_temp) < 3:
                    final_indexes.append(np.nonzero(temp_idx)[0][max_force_idx])
                else:  # TODO: find collinear and/or plane convex hull
                    final_indexes.append(np.nonzero(temp_idx)[0][max_force_idx])
            else:
                for _ in np.nonzero(temp_idx)[0]:
                    final_indexes.append(_)

        # select only final ones
        normals = normals[final_indexes, :]
        points = points[final_indexes, :]
        forces = forces[final_indexes]
        frictions = frictions[final_indexes]

        return points, normals, forces, frictions

    @staticmethod
    def compute_cone(mus: np.ndarray, forces: np.ndarray, normals: np.ndarray, points: np.ndarray, num_edges: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Function to compute the friction cone for given normal force
        :param mus: friction coefficient
        :type mus: array of floats
        :param forces: normal forces in the touch points
        :type forces: Nx3 np.array()
        :param normals: the size of the force
        :type normals: Nx1 np.array()
        :param points: array with contact points
        :type points: Nx3 np.array/
        :param num_edges: number of edges to approximate the cone
        :type num_edges: int, optional, default=8
        :return: approximated forces on the edge of the friction cones
        :rtype: tuple[np.ndarray, np.ndarray]
        """

        forces_Linf = []
        forces_L1 = []
        sum_forces = np.sum(forces)
        for idx, mu, force, normal, point in zip(np.arange(0, forces.shape[0]), mus, forces, normals, points):
            # force = 1

            # r = mu*force
            # # split the circle in num_edges parts (for default=8, it will split by 45 degrees)
            # thetas = np.arange(0, 2*np.pi, 2*np.pi/num_edges)
            #
            # # compute forces on circle, with the same height
            # forces_local = np.vstack((r*np.cos(thetas), r*np.sin(thetas), np.tile(force, (num_edges, )))).T


            # # Rotation axis is cross product of original z-axis (0, 0, 1) to the new z-axis (normal_force)
            # rotation_axis = np.cross(np.array([0, 0, 1]), normal)
            #
            # sin = np.linalg.norm(rotation_axis)
            # cos = np.dot(np.array([0, 0, 1]), normal)
            #
            # # Normalize the rotation axis
            # rotation_axis = rotation_axis/sin
            #
            # # Skew matrix from the rotation axis
            # skew = np.array([[0, -rotation_axis[2], rotation_axis[1]],
            #                  [rotation_axis[2], 0, -rotation_axis[0]],
            #                  [-rotation_axis[1], rotation_axis[0], 0]])
            # # Rodriguez formula
            # R = np.eye(3) + sin*skew + (1-cos)*(skew @ skew)
            #
            # forces_local = (R @ forces_local.T).T
            #
            ns = null_space(normal.reshape(1, 3))
            u = ns[:, 0]
            v = ns[:, 1]

            f_len = np.linalg.norm(normal + mu * np.cos(0) * u + mu * np.sin(0) * v)
            scales = [1/f_len * 1/num_edges, 1/f_len * 1/num_edges * force/sum_forces]

            for scale, forces_temp in zip(scales, [forces_Linf, forces_L1]):
                forces_local = np.zeros((num_edges, 3))

                for j in range(num_edges):
                    forces_local[j, :] = scale*normal + \
                                         scale*mu*np.cos(2*np.pi*j/num_edges)*u + scale*mu*np.sin(2*np.pi*j/num_edges)*v

                for f in forces_local:
                    forces_temp.append(f)

        return np.array(forces_Linf), np.array(forces_L1)

    def compute_gws_mesh(self, fixed_set: List[int], hull_type: str) -> None:
        # TODO: This works only when full 6D hull is computed
        hyperplanes = np.unique(self.hulls[hull_type.split("_")[0]].equations, axis=0)
        self.project_to_3d(np.zeros(6), fixed_set, hyperplanes, hull_type=hull_type)

    class MinkowskiSum:
        """
        Recursively compute Minkowski sum of individual wrenches.
        """
        def __init__(self, wrenches: np.ndarray, cone_vertices: int, contacts: int, dims: int = 6) -> None:
            self.wrench_id = 0
            self.contacts = contacts
            self.dims = dims

            # m^n elements
            self.wrenches_out = np.zeros((cone_vertices ** contacts, dims))

            self.cone_vertices = cone_vertices
            self.wrenches = wrenches

        def minkowski_sum(self, contact_id: int, prev_sum: Optional[np.ndarray] = None, used_wrenches: int = 0) -> int:
            """
            Recursive function itself
            :param contact_id: id of contact
            :type contact_id: int
            :param prev_sum: previous sum; used to sum more than two wrenches
            :type prev_sum: np.ndarray, optional, default=None
            :param used_wrenches: number of currently selected wrenches
            :type used_wrenches: int, optional, default=0
            :return: 0
            :rtype: int
            """

            # so we can use class variable in argument
            if prev_sum is None:
                prev_sum = np.zeros(self.dims)

            if contact_id == self.contacts:
                if used_wrenches == self.contacts:
                    self.wrenches_out[self.wrench_id, :] = prev_sum
                    self.wrench_id += 1
                return 0

            self.minkowski_sum(contact_id + 1, copy.deepcopy(prev_sum))

            for m in range(self.cone_vertices):
                cur_sum = prev_sum + self.wrenches[contact_id * self.cone_vertices + m]
                self.minkowski_sum(contact_id + 1, cur_sum, used_wrenches+1)

            return 0

    def evaluate_contact(self, hull_types: List[str] = ["L1", "Linf"]) -> Dict[str, ConvexHull]:
        """
        Call necessary function and computes the convex hull of wrenches

        :param hull_types: types of hull: L1 or Linf
        :type hull_types: list[str], optional, default=["L1", "Linf"]
        :return: computed convex hulls for requested metrics
        :rtype: dict[str, scipy.spatial.ConvexHull]
        """
        graspable_object = self.client.object
        cone_vertices = self.client.config.analytical.cone_vertices
        position, orientation = self.client.getBasePositionAndOrientation(graspable_object.id)
        R = np.eye(4)
        R[:3, :3] = np.reshape(self.client.getMatrixFromQuaternion(orientation), (3, 3))
        R[:3, 3] = position
        self.pose = R

        # torque origin is the center of gravity of the object
        # it must be recomputed every time, as it changes with the change in position of the body
        torque_origin, _ = self.client.getBasePositionAndOrientation(graspable_object.id)
        torque_origin = np.array(torque_origin)

        # get contact information
        points, normals, forces, frictions = self.get_contact_info(graspable_object.id, torque_origin)
        if points is None:
            self.client.logger.log_debug(
                f"No contact found")
            return {}

        if self.client.config.visualization.mode in [1, 2]:
            self.client.visualizer.draw_normal_forces(points, normals, torque_origin)

        # Compute friction cones
        forces_from_cones_Linf, forces_from_cones_L1 = self.compute_cone(frictions, forces, normals, points, cone_vertices)

        # torque_multiplier = 1/r; r is max radius from torque origin
        # we use r precomputed in the beginning, as it is not changing given the position/orientation of the body
        r = graspable_object.r

        # torque = torque_multiplier * ((position - torque_origin) x force_vector)
        points_concat = np.tile(points, (1, cone_vertices)).flatten().reshape((-1, 3))
        d = points_concat#-torque_origin
        # d /= np.linalg.norm(d, axis=1).reshape(-1, 1)
        torques_Linf = np.cross(d, forces_from_cones_Linf)/r
        torques_L1 = np.cross(d, forces_from_cones_L1)/r

        # # Visualize
        if self.client.config.visualization.mode in [1, 2]:
            self.client.visualizer.draw_force_torques(points, copy.deepcopy(forces_from_cones_Linf), torques_Linf, cone_vertices,
                                                      forces, torque_origin)

        if points.shape[0] < 2:
            self.client.logger.log_debug(f"Only {points.shape[0]} contacts found. At least 2 contacts are needed to create GWS.")
            return {}


        output = {}
        wrenches_Linf = np.hstack((forces_from_cones_Linf, torques_Linf))
        wrenches_Linf = np.vstack((wrenches_Linf, [0, 0, 0, 0, 0, 0]))

        wrenches_L1 = np.hstack((forces_from_cones_L1, torques_L1))
        wrenches_L1 = np.vstack((wrenches_L1, [0, 0, 0, 0, 0, 0]))
        wrenches = {"Linf": wrenches_Linf, "L1": wrenches_L1}
        for hull_type in hull_types:
            wrenches_temp = wrenches[hull_type]
            if hull_type == "Linf":  # do minkowski sum
                ms = self.MinkowskiSum(wrenches_temp, cone_vertices, points.shape[0])
                ms.minkowski_sum(0)
                wrenches_temp = ms.wrenches_out

            if self.client.config.analytical.gws.scale:
                wrenches_temp = (((wrenches_temp - np.min(wrenches_temp, axis=0)) *
                                  (self.client.config.analytical.gws.scale_interval[1] - self.client.config.analytical.gws.scale_interval[0]))
                                 / (np.max(wrenches_temp, axis=0) - np.min(wrenches_temp, axis=0))) \
                                + self.client.config.analytical.gws.scale_interval[0]
            try:
                output[hull_type] = ConvexHull(wrenches_temp, qhull_options="n Qx C-"+self.client.config.analytical.gws.merging_radius)
                h = output[hull_type]
                q = np.min(-h.equations[:, -1])
                volume = h.volume
                self.client.logger.log_debug(f"Hull {hull_type} created with volume {volume} and epsilon {q}")
            except Exception as e:
                self.client.logger.log_debug(f"Hull {hull_type} cannot be created.")
                self.client.logger.log_debug(f"With an exception: {e}")

        if output != {} and self.client.config.visualization.mode in [1, 2]:
            self.client.visualizer.show_menu.set_enabled(6, True)
        self.hulls = output

    def project_to_3d(self, where_to_project: np.ndarray, fixed_coordinates: List[int], hyperplanes: np.ndarray, hull_type: str = "L1_fixed_force") -> None:
        """
        Projects selected dimensions of a 6D hull to 3D and builds a mesh for visualization.

        :param where_to_project: values used for fixed dimensions during projection
        :type where_to_project: np.ndarray
        :param fixed_coordinates: indices of coordinates that stay fixed during projection
        :type fixed_coordinates: list[int]
        :param hyperplanes: hyperplane representation of the source hull
        :type hyperplanes: np.ndarray
        :param hull_type: name used as key in visualized GWS meshes
        :type hull_type: str, optional, default="L1_fixed_force"
        :return: None
        :rtype: None
        """
        # where_to_project = [0, 0, 0, 0, 0, 0]
        # fixedSet = [0, 1, 2] or [3, 4, 5] depending whether fixing force or torque

        # Get free coordinates as opposite to the fixed ones
        free_coordinates = [_ for _ in range(len(where_to_project)) if _ not in fixed_coordinates]

        planes = []
        # Project the hyperplanes to 3D
        for hyperplane in hyperplanes:
            hyperplane_len = np.linalg.norm(hyperplane[free_coordinates])

            # To check non-zero hyperplanes
            if hyperplane_len > 1e-11:
                temp_plane = hyperplane[free_coordinates] / hyperplane_len
                # offset of the plane from 3D origin
                offset = (hyperplane[-1] + hyperplane[fixed_coordinates] @ where_to_project[
                    fixed_coordinates]) / hyperplane_len
                temp_plane = np.append(temp_plane, offset)
                planes.append(temp_plane)

        planes = np.array(planes)

        # Get feasible point by linear programming magic
        norm_vector = np.reshape(np.linalg.norm(planes[:, :-1], axis=1), (planes.shape[0], 1))
        c = np.zeros((planes.shape[1],))
        c[-1] = -1
        A = np.hstack((planes[:, :-1], norm_vector))
        b = - planes[:, -1:]
        res = linprog(c, A_ub=A, b_ub=b, bounds=(None, None), method="highs-ipm")

        interior_point = res.x[:-1]
        cleared_planes = []
        wrong_counter = 0
        for plane in planes:
            dist = plane[-1] + plane[:3] @ interior_point
            if dist <= 0:
                cleared_planes.append(plane)
            else:
                wrong_counter += 1
        if wrong_counter > 0:
            self.client.logger.log_warning(f"{wrong_counter} hyperplanes were wrong for {hull_type}. The hull be will be most probably incorrect.")
        cleared_planes = np.array(cleared_planes)

        # planes, interior point from linear programming
        hsi = HalfspaceIntersection(cleared_planes, interior_point, qhull_options="Pp Qbb")

        # normal/offset
        vertices = hsi.dual_equations[:, :3] / -hsi.dual_equations[:, -1:]

        # just for better visualization
        vertices -= np.mean(vertices, axis=0)

        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices))
        hull_mesh, _ = pc.compute_convex_hull()

        hull_mesh.orient_triangles()
        hull_mesh.compute_triangle_normals()
        hull_mesh.paint_uniform_color([0, 0, 1])

        # Check for inversion -> if angle between triangle normal and vector between triangle center and center is <90
        #                        - this works because the hulls are convex!
        triangle_normal = np.asarray(hull_mesh.triangle_normals)[0]
        triangle_to_center = hull_mesh.get_center() - np.mean(
            np.asarray(hull_mesh.vertices)[np.asarray(hull_mesh.triangles)[0]], axis=0)
        angle = np.arctan2(np.linalg.norm(np.cross(triangle_normal, triangle_to_center)),
                           np.dot(triangle_normal, triangle_to_center))
        if 0 < angle < np.pi / 2 or angle < -270:
            hull_mesh.triangles = o3d.utility.Vector3iVector(np.fliplr(hull_mesh.triangles))
            hull_mesh.compute_triangle_normals()
        self.gws_meshes[hull_type] = hull_mesh
