#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2020 Centre National d'Etudes Spatiales (CNES).
# Copyright (C) 2023 CS Group.
#
# This file is part of CARS
# (see https://github.com/CNES/cars).
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
Projection module:
contains some general purpose functions using polygons and data projections
"""

import numpy as np
import pandas
import pyproj
from cars.core import constants as cst


def points_cloud_conversion(
    cloud_in: np.ndarray, epsg_in: int, epsg_out: int
) -> np.ndarray:
    """
    Convert a point cloud from a SRS to another one.

    :param cloud_in: cloud to project
    :param epsg_in: EPSG code of the input SRS
    :param epsg_out: EPSG code of the output SRS
    :return: Projected point cloud
    """
    # Get CRS from input EPSG codes
    crs_in = pyproj.CRS.from_epsg(epsg_in)
    crs_out = pyproj.CRS.from_epsg(epsg_out)

    # Project point cloud between CRS (keep always_xy for compatibility)
    cloud_in = np.array(cloud_in).T
    transformer = pyproj.Transformer.from_crs(crs_in, crs_out, always_xy=True)
    cloud_in = transformer.transform(*cloud_in)
    cloud_in = np.array(cloud_in).T

    return cloud_in


def points_cloud_conversion_dataframe(
    cloud: pandas.DataFrame, epsg_in: int, epsg_out: int
):
    """
    Convert a point cloud as a panda.DataFrame to another epsg (inplace)

    :param cloud: cloud to project
    :param epsg_in: EPSG code of the input SRS
    :param epsg_out: EPSG code of the output SRS
    """
    xyz_in = cloud.loc[:, [cst.X, cst.Y, cst.Z]].values

    if xyz_in.shape[0] != 0:
        xyz_in = points_cloud_conversion(xyz_in, epsg_in, epsg_out)
        cloud[cst.X] = xyz_in[:, 0]
        cloud[cst.Y] = xyz_in[:, 1]
        cloud[cst.Z] = xyz_in[:, 2]