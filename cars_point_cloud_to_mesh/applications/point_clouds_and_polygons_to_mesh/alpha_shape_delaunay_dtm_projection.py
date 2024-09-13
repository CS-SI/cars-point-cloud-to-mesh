#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2023 CS Group.
#
# This file is part of cars_point_cloud_to_mesh
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
this module contains an implementation of a
point cloud to polygons application class.
"""
import json
import logging
import os

import cars.orchestrator.orchestrator as ocht
import numpy as np
from cars.core import projection, tiling
from cars.data_structures import cars_dataset
from json_checker import And, Checker, Or

from ..create_dtm_mesh import percentile_of_unclassified_points_tools as pupt
from . import alpha_shape_delaunay_dtm_projection_tools as asdt
from .point_clouds_and_polygons_to_mesh import PointCloudsAndPolygonsToMesh


def mesh_point_cloud(
    point_clouds,
    grouping_polys,
    groups,
    dtm_mesh,
    simplification_method,
    alpha_shape_radius,
    visvalingam_area,
    douglas_threshold,
    douglas_search_offset,
    height_extraction,
    height_detection_min_building_points,
    height_subgroups_filtering_radius,
    height_hist_smooth_ratio,
    height_hist_min_peaks_dist_m,
    height_hist_min_peak_prominence,
    min_cluster_area,
    cluster_reassign_search_radius,
    out_mesh_mode,
    out_epsg,
    out_folder,
    id_thread,
    building_index,
):
    """
    Outputs a mesh representing the point clouds' buildings in files
    """

    if len(point_clouds) <= 0:
        return None

    grid = asdt.get_grid(point_clouds, alpha_shape_radius * 2, building_index)
    if grid is None:
        return None

    pts_groups = asdt.group_points(
        grid, grouping_polys, groups, alpha_shape_radius
    )

    groups_per_point = np.zeros(grid.points.shape[0]) - 1
    for i, pts_group in enumerate(pts_groups):
        groups_per_point[pts_group] = i

    offset = grid.grid_base.copy()
    grid.points[:, :2] -= offset
    grid.grid_base -= offset

    if height_extraction:
        pts_groups, groups_per_point = asdt.refine_buildings_by_height(
            groups_per_point,
            grid,
            alpha_shape_radius,
            height_detection_min_building_points,
            height_subgroups_filtering_radius,
            height_hist_smooth_ratio,
            height_hist_min_peaks_dist_m,
            height_hist_min_peak_prominence,
            min_cluster_area,
            cluster_reassign_search_radius,
        )

    buildingsids = asdt.get_building_contours(
        grid, pts_groups, alpha_shape_radius, groups_per_point
    )

    triangulations = {
        "vertices": [],
        "uvs": [],
        "triangles": [],
        "classes": ["roof", "wall"],
        "triangles_classification": [],
    }

    for building_key, building_ids in buildingsids.items():

        # bk is an int representing the group id.
        # get the average z of all points in the building
        group_mean_height = np.median(
            grid.points[pts_groups[building_key]][:, 2]
        )

        # simplification
        for i, ctr in enumerate(building_ids):
            building_ids[i] = asdt.simplify(
                ctr,
                grid,
                simplification_method,
                douglas_threshold,
                douglas_search_offset,
                visvalingam_area,
            )

        triangulation = asdt.delaunay_triangulate(building_ids, grid)

        # No triangulation created (too few points)
        if triangulation is None:
            continue

        nb_top_verts = len(triangulation["vertices"])

        # add the z component to vertices
        triangulation["vertices"] = np.hstack(  # noqa: B909
            (
                triangulation["vertices"] + offset,
                np.zeros((nb_top_verts, 1)) + group_mean_height,
            )
        )

        triangulation["vertices"] = projection.points_cloud_conversion(
            triangulation["vertices"], 32631, dtm_mesh["crs"]
        )

        lower_vertices = asdt.project_on_dtm(
            triangulation["vertices"][:, :2], dtm_mesh
        )
        height_diff = triangulation["vertices"][:, 2] - lower_vertices[:, 2]
        # TODO: give control of this setting to the user
        if height_diff.mean() < 3:
            continue

        triangulation["vertices"] = np.vstack(
            (triangulation["vertices"], lower_vertices)
        )

        triangulation["classes"] = ["roof", "wall"]

        triangulation["triangles_classification"] = [
            0 for _ in triangulation["triangles"]
        ]

        # add triangles that make up the walls
        new_tris = []
        for seg in triangulation["segments"]:
            id_tl = seg[0]
            id_tr = seg[1]
            id_bl = id_tl + nb_top_verts
            id_br = id_tr + nb_top_verts

            new_tris.append([id_tl, id_br, id_bl])
            new_tris.append([id_br, id_tl, id_tr])

        triangulation["triangles_classification"] += [1 for _ in new_tris]

        triangulation["triangles"] = np.vstack(
            (triangulation["triangles"], np.array(new_tris, dtype=int))
        )

        # reproject into the output epsg
        uv_maps = []
        for point in triangulation["vertices"]:

            point = np.array([point[0], point[1], 1])  # homogenous 2d coord

            uv_map = np.matmul(dtm_mesh["uv_matrix"], point)
            uv_map /= uv_map[2]
            uv_map = uv_map[:2]  # back to non-homogenous coords
            uv_map[1] = 1 - uv_map[1]

            uv_maps.append(uv_map)

        triangulation["vertices"] = projection.points_cloud_conversion(
            triangulation["vertices"], dtm_mesh["crs"], out_epsg
        )

        triangulation["uvs"] = uv_maps

        voffset = len(triangulations["vertices"])
        triangulations["vertices"] += list(triangulation["vertices"])
        triangulations["uvs"] += triangulation["uvs"]
        triangulations["triangles"] += list(
            triangulation["triangles"] + voffset
        )
        triangulations["triangles_classification"] += triangulation[
            "triangles_classification"
        ]

    if out_mesh_mode == "texture":
        pupt.export_obj(
            os.path.join(out_folder, "buildings/"),
            f"bldgs_{id_thread}",
            "../color.png",
            triangulations["vertices"],
            triangulations["uvs"],
            triangulations["triangles"],
            id_thread=id_thread,
        )
    elif out_mesh_mode == "classification":
        pupt.export_obj(
            os.path.join(out_folder, "buildings/"),
            f"bldgs_{id_thread}",
            None,
            triangulations["vertices"],
            triangulations["uvs"],
            triangulations["triangles"],
            classif_data={
                "classes": triangulations["classes"],
                # default colors (a shade of darker
                # orange and a shade of pinkier orange)
                "classes_color": [[204, 102, 0], [255, 153, 102]],
                "classes_texture": [None, None],
                "triangles_classification": triangulations[
                    "triangles_classification"
                ],
            },
            id_thread=id_thread,
        )

    return None


class AlphaShapeDelaunayDtmProjection(
    PointCloudsAndPolygonsToMesh,
    short_name="alpha_shape_delaunay_dtm_projection",
):
    """
    PointCloudsAndPolygonsToMesh
    """

    def __init__(self, conf=None):
        """
        Init function of PointCloudToPolygons

        :param conf: configuration for matching
        :return: a application_to_use object
        """

        super().__init__(conf=conf)

        self.used_config = self.check_conf(conf)

        # check conf
        self.used_method = self.used_config["method"]

        # Init orchestrator
        self.orchestrator = None

    def check_conf(self, conf):
        """
        Check configuration

        :param conf: configuration to check
        :type conf: dict

        :return: overloaded configuration
        :rtype: dict

        """

        # init conf
        if conf is not None:
            overloaded_conf = conf.copy()
        else:
            conf = {}
            overloaded_conf = {}

        self.simplification_methods = [
            "none",
            "douglas",
            "visvalingam",
            "douglas then visvalingam",
        ]
        self.out_mesh_modes = ["texture", "classification"]

        # Overload conf
        overloaded_conf["method"] = conf.get(
            "method", "alpha_shape_delaunay_dtm_projection"
        )
        overloaded_conf["alpha_shape_radius"] = conf.get(
            "alpha_shape_radius", 1.5
        )
        overloaded_conf["simplification_method"] = conf.get(
            "simplification_method", "douglas then visvalingam"
        )
        overloaded_conf["douglas_threshold"] = conf.get(
            "douglas_threshold", 1.3
        )
        overloaded_conf["douglas_search_offset"] = conf.get(
            "douglas_search_offset", 5
        )
        overloaded_conf["visvalingam_area"] = conf.get("visvalingam_area", 25)

        overloaded_conf["height_extraction"] = conf.get(
            "height_extraction", False
        )
        overloaded_conf["height_detection_min_building_points"] = conf.get(
            "height_detection_min_building_points", 20
        )
        overloaded_conf["height_subgroups_filtering_radius"] = conf.get(
            "height_subgroups_filtering_radius", 2.5
        )
        overloaded_conf["height_hist_smooth_ratio"] = conf.get(
            "height_hist_smooth_ratio", 0.33
        )
        overloaded_conf["height_hist_min_peaks_dist_m"] = conf.get(
            "height_hist_min_peaks_dist_m", 1.5
        )
        overloaded_conf["height_hist_min_peak_prominence"] = conf.get(
            "height_hist_min_peak_prominence", 0.2
        )

        overloaded_conf["min_cluster_area"] = conf.get("min_cluster_area", 50)
        overloaded_conf["cluster_reassign_search_radius"] = conf.get(
            "cluster_reassign_search_radius", 2.5
        )

        overloaded_conf["out_mesh_mode"] = conf.get(
            "out_mesh_mode", self.out_mesh_modes[0]
        )

        polygon_schema = {
            "method": str,
            # common params
            "alpha_shape_radius": And(Or(float, int), lambda x: x > 0),
            "simplification_method": lambda x: x in self.simplification_methods,
            "douglas_threshold": And(Or(float, int), lambda x: x > 0),
            "douglas_search_offset": And(int, lambda x: x >= 0),
            "visvalingam_area": And(Or(float, int), lambda x: x > 0),
            # LOD2 params
            "height_extraction": bool,
            "height_detection_min_building_points": And(int, lambda x: x >= 0),
            "height_subgroups_filtering_radius": And(
                Or(float, int), lambda x: x > 0
            ),
            "height_hist_smooth_ratio": And(
                Or(float, int), lambda x: 1 > x >= 0
            ),
            "height_hist_min_peaks_dist_m": And(
                Or(float, int), lambda x: x >= 0
            ),
            "height_hist_min_peak_prominence": And(
                Or(float, int), lambda x: 1 >= x >= 0
            ),
            "min_cluster_area": And(Or(float, int), lambda x: x >= 0),
            "cluster_reassign_search_radius": And(
                Or(float, int), lambda x: x > 0
            ),
            # Out params
            "out_mesh_mode": And(str, lambda x: x in self.out_mesh_modes),
        }
        checker = Checker(polygon_schema)

        checker.validate(overloaded_conf)

        return overloaded_conf

    def run(
        self,
        point_clouds,
        tiles,
        groups,
        dtm_mesh,
        out_dir,
        out_epsg=4326,
        classification_buildings_description=None,
        orchestrator=None,
    ):
        """
        Executes the meshing algorithm
        """

        # Check if input data is supported
        data_valid = False
        if isinstance(point_clouds, list):
            if isinstance(point_clouds[0], cars_dataset.CarsDataset):
                if point_clouds[0].dataset_type in ("arrays", "points"):
                    data_valid = True

        if not data_valid:
            message = (
                "PointsCloudRasterisation application doesn't support "
                "this input data format : "
                f"type : {type(point_clouds)}"
            )
            logging.error(message)
            raise RuntimeError(message)

        if orchestrator is None:
            # Create default orchestrator for current application
            self.orchestrator = ocht.Orchestrator()
        else:
            self.orchestrator = orchestrator

        # placeholder cars dataset
        meshes = cars_dataset.CarsDataset(dataset_type="dict")
        meshes.tiling_grid = tiling.generate_tiling_grid(
            0, 0, 1, len(tiles), 1, 1
        )
        self.orchestrator.add_to_replace_lists(
            meshes,
            "Generating meshes from groups of points",
        )

        # multiple point clouds may have been given
        # only use the first point cloud given as input
        pcd = point_clouds[0]

        broadcasted_dtm_mesh = self.orchestrator.cluster.scatter(dtm_mesh)

        for i, tilekey in enumerate(tiles):
            meshes[0, i] = self.orchestrator.cluster.create_task(
                mesh_point_cloud
            )(
                # list of all point clouds concerned by this thread
                [
                    pcd[r, c]
                    for r, c in [
                        (tilekey[j], tilekey[j + 1])
                        for j in range(0, len(tilekey) - 1, 2)
                    ]
                ],
                # list of all polys concerned by this thread
                tiles[tilekey],
                # list with the group of each poly in this thread
                groups[tilekey],
                # mesh to project vertices onto
                broadcasted_dtm_mesh,
                # config parameters
                self.used_config["simplification_method"],
                self.used_config["alpha_shape_radius"],
                self.used_config["visvalingam_area"],
                self.used_config["douglas_threshold"],
                self.used_config["douglas_search_offset"],
                self.used_config["height_extraction"],
                self.used_config["height_detection_min_building_points"],
                self.used_config["height_subgroups_filtering_radius"],
                self.used_config["height_hist_smooth_ratio"],
                self.used_config["height_hist_min_peaks_dist_m"],
                self.used_config["height_hist_min_peak_prominence"],
                self.used_config["min_cluster_area"],
                self.used_config["cluster_reassign_search_radius"],
                self.used_config["out_mesh_mode"],
                out_epsg,
                out_dir,
                i,
                classification_buildings_description,
            )

        with open(
            os.path.join(out_dir, "buildings_attrs.json"),
            "w",
            encoding="utf8",
        ) as desc:
            json.dump({"EPSG": out_epsg}, desc)
