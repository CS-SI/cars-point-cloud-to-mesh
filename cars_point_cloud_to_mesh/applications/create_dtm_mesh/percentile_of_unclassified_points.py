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
from cars.core import tiling
from cars_point_cloud_to_mesh.core import projection

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

    # Pre-extract numpy arrays once to avoid repeated attribute access
    x_vals = point_cloud["x"].values
    y_vals = point_cloud["y"].values

    width = x_vals.shape[0]
    height = x_vals.shape[1]

    center_idx = pupt.get_closest_point_id(valid_pts, [width // 2, height // 2])
    center = valid_pts[center_idx]

    pcdz = pupt.evaluate_z_around(
        point_cloud, unclassified_selector, center, percentile, search_offset
    )
    out_dict.data["vertices"].append(
        [x_vals[center[0], center[1]], y_vals[center[0], center[1]], pcdz]
    )

    # Compute boundary point indices once
    idx_left    = valid_pts[np.argmin(valid_pts[:, 0])]
    idx_right   = valid_pts[np.argmax(valid_pts[:, 0])]
    idx_lowest  = valid_pts[np.argmin(valid_pts[:, 1])]
    idx_highest = valid_pts[np.argmax(valid_pts[:, 1])]

    def _make_bb_point(idx):
        return [
            x_vals[idx[0], idx[1]],
            y_vals[idx[0], idx[1]],
            pupt.evaluate_z_around(
                point_cloud, unclassified_selector, idx, percentile, search_offset
            ),
        ]

    out_dict.data["bb_points"] = {
        "left":  _make_bb_point(idx_left),
        "right": _make_bb_point(idx_right),
        "up":    _make_bb_point(idx_highest),
        "down":  _make_bb_point(idx_lowest),
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
                data_valid = point_clouds[0].dataset_type in ("arrays", "points")

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

        # Use a set for O(1) duplicate detection instead of O(n) list scan
        seen_xys = set()
        list_xys = []
        list_zs = []
        lrud = {"l": None, "r": None, "u": None, "d": None}

        for row in range(pcd.shape[0]):
            for col in range(pcd.shape[1]):
                tile_data = vertices[row, col].data  # cache dict lookup

                lrud = pupt.update_lrud(lrud, tile_data["bb_points"])

                for point in tile_data["vertices"]:
                    key = (point[0], point[1])
                    if key not in seen_xys:
                        seen_xys.add(key)
                        list_xys.append([point[0], point[1]])
                        list_zs.append(point[2])

        # Append LRUD boundary points
        for pt in [lrud["l"], lrud["r"], lrud["u"], lrud["d"]]:
            list_xys.append([pt[0], pt[1]])
            list_zs.append(pt[2])

        dtm_mesh = libtr.triangulate({"vertices": list_xys})

        assert len(dtm_mesh["vertices"]) == len(list_zs)

        if self.used_config["filter_edges"]:
            dtm_mesh["triangles"] = pupt.filter_triangles_of_mesh(
                dtm_mesh,
                self.used_config["filter_max_length_to_median_edge_ratio"],
            )

        list_xys_arr = np.array(list_xys)
        list_zs_arr = np.array(list_zs)

        # Stitch xys and zs as a numpy array (avoids list comprehension)
        dtm_mesh["vertices"] = np.column_stack([list_xys_arr, list_zs_arr])

        inv_mat_tr, clr_epsg = pupt.get_relevant_info(dsm_color)

        # Single batch CRS conversion
        pts_for_uv = projection.points_cloud_conversion(
            list_xys_arr, crs_vertices, clr_epsg
        )
        # Homogeneous coords: (N, 3)
        pts_h = np.hstack([pts_for_uv, np.ones((len(pts_for_uv), 1))])
        # Single batched matmul for all points
        uv_maps_h = pts_h @ inv_mat_tr.T           # (N, 3)
        uv_maps_h /= uv_maps_h[:, 2:3]             # normalize by w
        uv_maps = uv_maps_h[:, :2].copy()           # (N, 2)
        uv_maps[:, 1] = 1.0 - uv_maps[:, 1]        # flip V

        dtm_mesh["uvs"] = uv_maps.tolist()

        # Single batch CRS conversion for mesh vertices
        dtm_mesh["vertices"] = projection.points_cloud_conversion(
            dtm_mesh["vertices"], crs_vertices, clr_epsg
        )

        dtm_mesh["crs"] = clr_epsg
        dtm_mesh["uv_matrix"] = inv_mat_tr

        pupt.tif_to_png(
            in_file=dsm_color,
            out_file=f"{out_dir}/color.png",
            scale_factor=self.used_config["texture_scale_factor"],
        )

        with open(
            os.path.join(out_dir, "dtm_mesh_attrs.json"), "w", encoding="utf8"
        ) as desc:
            json.dump({"EPSG": out_epsg}, desc)

        pupt.export_obj(
            out_dir,
            "dtm_mesh",
            "color.png",
            projection.points_cloud_conversion(
                np.array(dtm_mesh["vertices"]), clr_epsg, out_epsg
            ),
            dtm_mesh["uvs"],
            dtm_mesh["triangles"],
        )

        return dtm_mesh
