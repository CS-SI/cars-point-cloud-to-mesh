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
this module contains the dense_matching application class.
"""

# pylint: disable=C0302
# pylint: disable=line-too-long
# pylint: disable=import-error

import json

# Standard imports
import logging
import os

import cars.orchestrator.orchestrator as ocht
import numpy as np
import triangle as libtr
from cars.core import projection, tiling

# CARS imports
from cars.data_structures import cars_dataset
from cars.data_structures.cars_dict import CarsDict

# Third party imports
from json_checker import And, Checker, Or

from . import percentile_of_unclassified_points_tools as pupt
from .create_dtm_mesh import CreateDtmMesh


def get_dtm_mesh_vertices(
    point_cloud, percentile, search_offset, building_index, saving_info
):
    """
    Fuction wrapping the algorithm returning vertices of the dtm
    """

    # get only unclassified points
    unclassified_selector = pupt.get_unclassified_selector(
        point_cloud, building_index
    )

    # prepare output
    out_dict = CarsDict(
        {
            "vertices": [],
            "bb_points": {
                "left": None,
                "right": None,
                "up": None,
                "down": None,
            },
            "crs": point_cloud.attrs["epsg"],
        }
    )

    # if there are no points to sample
    if np.sum(unclassified_selector) <= 0:
        cars_dataset.fill_dict(out_dict, saving_info=saving_info)
        return out_dict

    valid_pts = np.argwhere(unclassified_selector)

    width = point_cloud["x"].values.shape[0]
    height = point_cloud["x"].values.shape[1]

    center = pupt.get_closest_point_id(valid_pts, [width // 2, height // 2])

    to_eval_ids = [center]

    for ptid in to_eval_ids:

        current_point = valid_pts[ptid]

        pcdx = point_cloud["x"].values[current_point[0], current_point[1]]
        pcdy = point_cloud["y"].values[current_point[0], current_point[1]]
        pcdz = pupt.evaluate_z_around(
            point_cloud,
            unclassified_selector,
            current_point,
            percentile,
            search_offset,
        )

        out_dict.data["vertices"].append([pcdx, pcdy, pcdz])

    leftest = valid_pts[np.argmin(valid_pts[:, 0])]
    rightest = valid_pts[np.argmax(valid_pts[:, 0])]
    lowest = valid_pts[np.argmin(valid_pts[:, 1])]
    highest = valid_pts[np.argmax(valid_pts[:, 1])]

    out_dict.data["bb_points"] = {
        "left": [
            point_cloud["x"].values[leftest[0], leftest[1]],
            point_cloud["y"].values[leftest[0], leftest[1]],
            pupt.evaluate_z_around(
                point_cloud,
                unclassified_selector,
                leftest,
                percentile,
                search_offset,
            ),
        ],
        "right": [
            point_cloud["x"].values[rightest[0], rightest[1]],
            point_cloud["y"].values[rightest[0], rightest[1]],
            pupt.evaluate_z_around(
                point_cloud,
                unclassified_selector,
                rightest,
                percentile,
                search_offset,
            ),
        ],
        "up": [
            point_cloud["x"].values[highest[0], highest[1]],
            point_cloud["y"].values[highest[0], highest[1]],
            pupt.evaluate_z_around(
                point_cloud,
                unclassified_selector,
                highest,
                percentile,
                search_offset,
            ),
        ],
        "down": [
            point_cloud["x"].values[lowest[0], lowest[1]],
            point_cloud["y"].values[lowest[0], lowest[1]],
            pupt.evaluate_z_around(
                point_cloud,
                unclassified_selector,
                lowest,
                percentile,
                search_offset,
            ),
        ],
    }

    cars_dataset.fill_dict(out_dict, saving_info=saving_info)

    return out_dict


class PercentileOfUnclassifiedPoints(
    CreateDtmMesh, short_name="percentile_of_unclassified_points"
):
    """
    CreateDtmMesh
    """

    def __init__(self, conf=None):
        """
        Init function of CreateDtmMesh subclass

        :param conf: configuration
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

        # Overload conf
        overloaded_conf["method"] = conf.get(
            "method", "percentile_of_unclassified_points"
        )

        overloaded_conf["percentile"] = conf.get("percentile", 5)
        overloaded_conf["dtm_precision"] = conf.get("dtm_precision", 250)
        overloaded_conf["search_offset"] = conf.get("search_offset", 75)
        overloaded_conf["filter_edges"] = conf.get("filter_edges", True)
        overloaded_conf["filter_max_length_to_median_edge_ratio"] = conf.get(
            "filter_max_length_to_median_edge_ratio", 3
        )
        overloaded_conf["texture_scale_factor"] = conf.get(
            "texture_scale_factor", 1
        )

        polygon_schema = {
            "method": str,
            "percentile": And(int, lambda x: 0 <= x <= 100),
            # 50 -> a point for each tile with a size of 50x50 (in pixels)
            "dtm_precision": And(int, lambda x: 50 <= x <= 10_000),
            # 500 -> 1000 x 1000 search zone, plenty
            # for just one single sample point
            "search_offset": And(int, lambda x: 1 <= x <= 500),
            # remove edges thay might be too long (because of delaunay)
            # using the ratio edge length / median edge length
            "filter_edges": bool,
            "filter_max_length_to_median_edge_ratio": And(
                Or(int, float), lambda x: 0 < x
            ),
            # allow upscaling (>1) if needed
            "texture_scale_factor": And(Or(int, float), lambda x: 0 < x <= 10),
        }
        checker = Checker(polygon_schema)

        checker.validate(overloaded_conf)

        return overloaded_conf

    def run(
        self,
        point_clouds,
        out_dir,
        out_epsg=4978,
        dsm_color=None,
        classification_buildings_description=None,
        orchestrator=None,
    ):
        """
        Executes the dtm mesh creation algorithm
        """

        # Check if input data is supported
        data_valid = False
        if isinstance(point_clouds, list):
            if isinstance(point_clouds[0], cars_dataset.CarsDataset):
                data_valid = point_clouds[0].dataset_type in (
                    "arrays",
                    "points",
                )

        if not data_valid:
            message = (
                "CreateDtmMesh application doesn't support "
                "this input data "
                f"format : type : {type(point_clouds)}"
            )
            logging.error(message)
            raise RuntimeError(message)

        self.orchestrator = orchestrator
        if self.orchestrator is None:
            self.orchestrator = ocht.Orchestrator()

        vertices = cars_dataset.CarsDataset(dataset_type="dict")

        pcd = point_clouds[0]  # only use the first point cloud given as input

        vertices.tiling_grid = tiling.generate_tiling_grid(
            0, 0, pcd.shape[0], pcd.shape[1], 1, 1
        )

        [saving_info] = self.orchestrator.get_saving_infos([vertices])

        self.orchestrator.add_to_replace_lists(
            vertices,
            "Computing the dtm mesh vertices for each tile",
        )

        for row in range(pcd.shape[0]):
            for col in range(pcd.shape[1]):
                # update saving infos  for potential replacement
                full_saving_info = ocht.update_saving_infos(
                    saving_info, row=row, col=col
                )
                vertices[row, col] = self.orchestrator.cluster.create_task(
                    get_dtm_mesh_vertices
                )(
                    pcd[row, col],
                    self.used_config["percentile"],
                    self.used_config["search_offset"],
                    classification_buildings_description,
                    saving_info=full_saving_info,
                )

        self.orchestrator.breakpoint()

        crs_vertices = vertices[0, 0].data["crs"]

        list_xys = []
        list_zs = []

        lrud = {
            "l": None,
            "r": None,
            "u": None,
            "d": None,
        }

        for row in range(pcd.shape[0]):
            for col in range(pcd.shape[1]):

                lrud = pupt.update_lrud(
                    lrud,
                    vertices[row, col].data["bb_points"],
                )

                for point in vertices[row, col].data["vertices"]:
                    if point[:2] not in list_xys:
                        list_xys.append(point[:2])
                        list_zs.append(point[2])

        list_xys += [
            [current_x, current_y]
            for current_x, current_y, _z in [
                lrud["l"],
                lrud["r"],
                lrud["u"],
                lrud["d"],
            ]
        ]
        list_zs += [
            z for _x, _y, z in [lrud["l"], lrud["r"], lrud["u"], lrud["d"]]
        ]

        dtm_mesh = libtr.triangulate({"vertices": list_xys})

        assert len(dtm_mesh["vertices"]) == len(list_zs)

        if self.used_config["filter_edges"]:
            # remove "bad" triangles from the dtm
            dtm_mesh["triangles"] = pupt.filter_triangles_of_mesh(
                dtm_mesh,
                self.used_config["filter_max_length_to_median_edge_ratio"],
            )

        # stitch back xys and zs together
        dtm_mesh["vertices"] = [
            [xy[0], xy[1], list_zs[i]] for i, xy in enumerate(list_xys)
        ]

        # compute uvs in color file
        uv_maps = []

        inv_mat_tr, clr_epsg = pupt.get_relevant_info(dsm_color)

        for point_x, point_y, _point_z in dtm_mesh["vertices"]:
            point = projection.points_cloud_conversion(
                np.array([point_x, point_y]), crs_vertices, clr_epsg
            )
            # homogenous coords
            point = np.array([point[0], point[1], 1])

            uv_map = np.matmul(inv_mat_tr, point)
            uv_map /= uv_map[2]
            uv_map = uv_map[:2]  # back to normal coords
            uv_map[1] = 1 - uv_map[1]
            # uv = np.clip(uv, 0, 1)
            uv_maps.append(uv_map)

        dtm_mesh["vertices"] = projection.points_cloud_conversion(
            np.array(dtm_mesh["vertices"]), crs_vertices, clr_epsg
        )
        dtm_mesh["uvs"] = uv_maps

        dtm_mesh["crs"] = clr_epsg
        dtm_mesh["uv_matrix"] = inv_mat_tr

        pupt.tif_to_png(
            in_file=dsm_color,
            out_file=f"{out_dir}/color.png",
            scale_factor=self.used_config["texture_scale_factor"],
        )

        with open(
            os.path.join(out_dir, "dtm_mesh_attrs.json"),
            "w",
            encoding="utf8",
        ) as desc:
            json.dump({"EPSG": out_epsg}, desc)

        pupt.export_obj(
            out_dir,
            "dtm_mesh",
            "color.png",
            projection.points_cloud_conversion(
                np.array(dtm_mesh["vertices"]),
                clr_epsg,
                out_epsg,
            ),
            dtm_mesh["uvs"],
            dtm_mesh["triangles"],
        )

        return dtm_mesh
