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

# Standard imports
import logging

import cars.orchestrator.orchestrator as ocht
import numpy as np
from cars.applications.holes_detection.holes_detection_tools import (
    classif_to_stacked_array,
)
from cars.core import projection, tiling

# CARS imports
from cars.data_structures import cars_dataset
from cars.data_structures.cars_dict import CarsDict

# Third party imports
from json_checker import And, Checker, Or

from . import external_alpha_shape_tools as east  # noqa: E501,B950

# noqa are for line too long
from .point_cloud_to_polygons import PointCloudToPolygons  # noqa: E501,B950


def trace_all_contours_wrapper(
    point_cloud, diameter, simplify, building_index, saving_info
):
    """
    Fuction wrapping the algorithm tracing the contours of a
    point cloud by creating its associated alpha shape
    """

    # -- format input --
    classif = classif_to_stacked_array(point_cloud, building_index)
    classified_selector = np.logical_and(
        classif.flatten() > 0,
        np.logical_not(np.isnan(point_cloud["x"].values.flatten())),
    )
    pcd = np.column_stack(
        (
            point_cloud["x"].values.flatten()[classified_selector],
            point_cloud["y"].values.flatten()[classified_selector],
            point_cloud["z"].values.flatten()[classified_selector],
        )
    )

    if len(pcd) <= 0:
        contours_dict = CarsDict({"contours": [], "aabbs": []})
        cars_dataset.fill_dict(contours_dict, saving_info=saving_info)

        return contours_dict

    pcd = projection.points_cloud_conversion(
        pcd, point_cloud.attrs["epsg"], 32631
    )

    # -- core algorihtm --

    pcd = pcd[np.argsort(pcd[:, 0])]
    mmin = pcd.min(axis=0)

    # Normalize pcd, to make precision much higher.
    # This is needed because positions that are compared
    # using an angle are computed, and the angles need to
    # be very precise
    pcd = pcd - mmin

    mask = np.zeros(pcd.shape[0], dtype=int)

    # create grid
    grid = east.Grid(pcd, [diameter, diameter])

    # create contours
    contours = []

    # while there are still points that don't belong to an outer contour
    while len(pt0 := np.where(mask == 0)[0]) > 0:

        # get the unmasked point with the lowest x
        pt0 = pt0[0]

        # trace his contour
        contour = east.trace_contour(
            pcd[:, :2],
            mask=mask,
            id_start_pt=pt0,
            start_vector=np.array([-1, 0], dtype=np.float32),
            alpha_shape_diameter=diameter,
            grid=grid,
        )

        # no contour found - this point is alone
        if contour is None:
            mask[pt0] = 1
            contours.append(np.array([pcd[pt0][:2]]) + mmin[:2])
            continue

        # contour found - mask off all points inside of it
        contour_as_pts = pcd[contour][:, :2]
        contours.append(contour_as_pts + mmin[:2])
        mask[contour] = 1

        wmin = contour_as_pts[:, :2].min(axis=0)
        wmax = contour_as_pts[:, :2].max(axis=0)

        gmin = grid.grid_pos(wmin)
        gmax = grid.grid_pos(wmax)

        for ptlist in grid.grid[
            gmin[0] : gmax[0] + 1, gmin[1] : gmax[1] + 1
        ].flat:
            for point in ptlist:
                if mask[point] != 0:
                    continue
                if not east.point_in_poly(pcd[point], contour_as_pts):
                    continue

                # noqa because mask is supposed
                # to be modified in this loop, it's
                # not a bug
                mask[point] = 1  # noqa: B909

    # -- simplification, if needed --

    if simplify:

        # remove small contours
        i = 0
        while i < len(contours):
            if len(contours[i]) > 5:
                i += 1
            else:
                del contours[i]

        # simplify remaining contours
        contours = east.simplify(contours)

    # format output?
    contours_dict = CarsDict(
        {
            "contours": contours,
            "aabbs": [(ctr.min(axis=0), ctr.max(axis=0)) for ctr in contours],
        }
    )
    cars_dataset.fill_dict(contours_dict, saving_info=saving_info)

    return contours_dict


class ExternalAlphaShape(
    PointCloudToPolygons, short_name="external_alpha_shape"
):
    """
    PointCloudToPolygons
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

        # Overload conf
        overloaded_conf["method"] = conf.get("method", "external_alpha_shape")
        overloaded_conf["simplify"] = conf.get("simplify", False)
        overloaded_conf["radius"] = conf.get("radius", 1.5)

        polygon_schema = {
            "method": str,
            "simplify": bool,
            "radius": And(Or(float, int), lambda x: x > 0),
        }
        checker = Checker(polygon_schema)

        checker.validate(overloaded_conf)

        return overloaded_conf

    def run(
        self,
        point_clouds,
        classification_buildings_description=None,
        orchestrator=None,
    ):
        """
        Executes the contouring algorithm on the point clouds given
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
                "this input data "
                f"format : type : {type(point_clouds)}"
            )
            logging.error(message)
            raise RuntimeError(message)

        if orchestrator is None:
            # Create default orchestrator for current application
            self.orchestrator = ocht.Orchestrator()
        else:
            self.orchestrator = orchestrator

        polygons = cars_dataset.CarsDataset(dataset_type="dict")

        pcd = point_clouds[0]  # only use the first point cloud given as input

        polygons.tiling_grid = tiling.generate_tiling_grid(
            0, 0, pcd.shape[0], pcd.shape[1], 1, 1
        )

        [saving_info] = self.orchestrator.get_saving_infos([polygons])

        self.orchestrator.add_to_replace_lists(
            polygons,
            "Computing the contour polygons of buildings for each tile",
        )

        for row in range(pcd.shape[0]):
            for col in range(pcd.shape[1]):
                # update saving infos  for potential replacement
                full_saving_info = ocht.update_saving_infos(
                    saving_info, row=row, col=col
                )
                polygons[row, col] = self.orchestrator.cluster.create_task(
                    trace_all_contours_wrapper
                )(
                    pcd[row, col],
                    self.used_config["radius"] * 2,  # pass diameter
                    self.used_config["simplify"],
                    classification_buildings_description,
                    saving_info=full_saving_info,
                )

        return polygons
