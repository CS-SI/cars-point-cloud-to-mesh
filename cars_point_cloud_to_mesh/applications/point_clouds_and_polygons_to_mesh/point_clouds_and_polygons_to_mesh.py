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
this module contains the abstract point cloud to polygons application class.
"""

import logging
from abc import ABCMeta, abstractmethod
from typing import Dict

from cars.applications.application import Application
from cars.applications.application_template import ApplicationTemplate


@Application.register("point_clouds_and_polygons_to_mesh")
class PointCloudsAndPolygonsToMesh(ApplicationTemplate, metaclass=ABCMeta):
    """
    Abstract class for the meshing of point
    clouds based on a spatial classification
    given by groups of polygons.
    """

    available_applications: Dict = {}
    default_application = "alpha_shape_delaunay_dtm_projection"

    def __new__(cls, conf=None):  # pylint: disable=W0613
        """
        Return the required application
        :raises:
         - KeyError when the required application is not registered

        :param conf: configuration for matching
        :return: a application_to_use object
        """

        meshing_method = cls.default_application
        if bool(conf) is False:
            logging.info(
                "Meshing method not specified, default {} is used",
                meshing_method,
            )
        else:
            meshing_method = conf.get("method", cls.default_application)

        if meshing_method not in cls.available_applications:
            logging.error("No meshing application named {} registered")
            raise KeyError(
                "No meshing application "
                "named {} registered".format(meshing_method)
            )

        logging.info(
            "The PointCloudToPolygons"
            "({}) application will be used".format(meshing_method)
        )

        return super(PointCloudsAndPolygonsToMesh, cls).__new__(
            cls.available_applications[meshing_method]
        )

    def __init_subclass__(cls, short_name, **kwargs):  # pylint: disable=E0302
        super().__init_subclass__(**kwargs)
        cls.available_applications[short_name] = cls

    @abstractmethod
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
        Run polygonalization application.

        Creates groups of polygons representing the buildings,
        using a point-wise classification.

        :param point_cloud: point cloud to create polygons from
        :type point_cloud: Dict

        :return: left matches, right matches
        :rtype: Tuple(CarsDataset, CarsDataset)
        """
