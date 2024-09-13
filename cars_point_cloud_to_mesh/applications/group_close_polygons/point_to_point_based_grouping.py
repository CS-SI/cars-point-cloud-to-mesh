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

# pylint: disable= C0302

# Standard imports

# Third party imports
from json_checker import And, Checker

# CARS imports
from . import point_to_point_based_grouping_tools as ppgt
from .group_close_polygons import GroupClosePolygons


def group_polygons(point_cloud_path, radius):
    return [point_cloud_path, radius]


class PointToPointBasedGrouping(
    GroupClosePolygons, short_name="point_to_point_based_grouping"
):
    """
    GroupClosePolygons
    """

    def __init__(self, conf=None):
        """
        Init function of GroupClosePolygons

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
        """

        # init conf
        if conf is not None:
            overloaded_conf = conf.copy()
        else:
            conf = {}
            overloaded_conf = {}

        app_conf_schema = {
            "method": str,
            "radius": And(float, lambda x: x > 0),
        }
        checker = Checker(app_conf_schema)

        # Overload conf
        overloaded_conf["method"] = conf.get(
            "method", "point_to_point_based_grouping"
        )
        overloaded_conf["radius"] = conf.get("radius", 1.1)

        checker.validate(overloaded_conf)

        return overloaded_conf

    def run(self, polygons, **kwargs):  # no orchestrator needed
        """
        Run the polygon grouping algorithm
        """

        # polygons has a list of polygons and a list of associated AABBs
        # compute overall AABB for each tile

        intersecting_polygons = ppgt.compute_intersecting_polygons(
            polygons, self.used_config["radius"]
        )

        new_tiles, groups = ppgt.compute_new_tiles(
            polygons, intersecting_polygons
        )

        return new_tiles, groups
