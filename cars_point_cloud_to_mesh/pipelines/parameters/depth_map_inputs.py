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
CARS depth map inputs module for point cloud to mesh pipeline
"""


import logging

from json_checker import Checker, Or

# CARS imports
import cars_point_cloud_to_mesh.pipelines.parameters.depth_map_inputs_constants as dm_cst
from cars.core import constants as cst
from cars.core import inputs
from cars.core.utils import make_relative_path_absolute


def check_depth_map_inputs(conf, config_dir=None):
    """
    TODO: add docstring
    """

    overloaded_conf = {}

    # Overload some optional parameters
    overloaded_conf[cst.INDEX_DEPTH_MAP_EPSG] = conf.get(cst.INDEX_DEPTH_MAP_EPSG, None)
    overloaded_conf[cst.ROI] = conf.get(cst.ROI, None)
    overloaded_conf[dm_cst.DEPTH_MAP] = {}

    # Validate inputs
    inputs_schema = {
        dm_cst.DEPTH_MAP: dict,
        cst.INDEX_DEPTH_MAP_EPSG: Or(int, None),
        cst.ROI: Or(str, dict, None),
    }

    checker_inputs = Checker(inputs_schema)
    checker_inputs.validate(overloaded_conf)

    # Validate depth maps

    dm_schema = {
        cst.INDEX_DEPTH_MAP_X: str,
        cst.INDEX_DEPTH_MAP_Y: str,
        cst.INDEX_DEPTH_MAP_Z: str,
        cst.INDEX_DEPTH_MAP_COLOR: str,
        cst.INDEX_DEPTH_MAP_MASK: Or(str, None),
        cst.INDEX_DEPTH_MAP_CLASSIFICATION: Or(str, None),
        cst.INDEX_DEPTH_MAP_PERFORMANCE_MAP: Or(str, None),
        cst.INDEX_DEPTH_MAP_AMBIGUITY: Or(str, None),
        cst.INDEX_DEPTH_MAP_FILLING: Or(str, None),
        cst.INDEX_DEPTH_MAP_EPSG: Or(str, int, None),
    }
    checker_dm = Checker(dm_schema)
    for depth_map_key in conf[dm_cst.DEPTH_MAP]:
        # Get depth maps with default
        overloaded_conf[dm_cst.DEPTH_MAP][depth_map_key] = {}
        overloaded_conf[dm_cst.DEPTH_MAP][depth_map_key][cst.INDEX_DEPTH_MAP_X] = conf[
            dm_cst.DEPTH_MAP
        ][depth_map_key].get("x", None)
        overloaded_conf[dm_cst.DEPTH_MAP][depth_map_key][cst.INDEX_DEPTH_MAP_Y] = conf[
            dm_cst.DEPTH_MAP
        ][depth_map_key].get("y", None)
        overloaded_conf[dm_cst.DEPTH_MAP][depth_map_key][cst.INDEX_DEPTH_MAP_Z] = conf[
            dm_cst.DEPTH_MAP
        ][depth_map_key].get("z", None)
        overloaded_conf[dm_cst.DEPTH_MAP][depth_map_key][
            cst.INDEX_DEPTH_MAP_COLOR
        ] = conf[dm_cst.DEPTH_MAP][depth_map_key].get("image", None)
        overloaded_conf[dm_cst.DEPTH_MAP][depth_map_key][
            cst.INDEX_DEPTH_MAP_MASK
        ] = conf[dm_cst.DEPTH_MAP][depth_map_key].get("mask", None)
        overloaded_conf[dm_cst.DEPTH_MAP][depth_map_key][
            cst.INDEX_DEPTH_MAP_CLASSIFICATION
        ] = conf[dm_cst.DEPTH_MAP][depth_map_key].get(
            "classification", None
        )
        overloaded_conf[dm_cst.DEPTH_MAP][depth_map_key][
            cst.INDEX_DEPTH_MAP_PERFORMANCE_MAP
        ] = conf[dm_cst.DEPTH_MAP][depth_map_key].get(
            "performance_map", None
        )
        overloaded_conf[dm_cst.DEPTH_MAP][depth_map_key][
            cst.INDEX_DEPTH_MAP_AMBIGUITY
        ] = conf[dm_cst.DEPTH_MAP][depth_map_key].get(
            "ambiguity", None
            )
        overloaded_conf[dm_cst.DEPTH_MAP][depth_map_key][
            cst.INDEX_DEPTH_MAP_FILLING
        ] = conf[dm_cst.DEPTH_MAP][depth_map_key].get("filling", None)


        
        overloaded_conf[dm_cst.DEPTH_MAP][depth_map_key][cst.INDEX_DEPTH_MAP_EPSG] = (
            conf[dm_cst.DEPTH_MAP][depth_map_key].get("epsg", 4326)
        )
        # validate
        checker_dm.validate(
            overloaded_conf[dm_cst.DEPTH_MAP][depth_map_key]
        )


    # Modify to absolute path
    if config_dir is not None:
        modify_to_absolute_path(config_dir, overloaded_conf)
    else:
        logging.debug(
            "path of config file was not given,"
            "relative path are not transformed to absolute paths"
        )

    for depth_map_key in conf[dm_cst.DEPTH_MAP]:
        # check sizes
        check_input_size(
            overloaded_conf[dm_cst.DEPTH_MAP][depth_map_key][cst.INDEX_DEPTH_MAP_X],
            overloaded_conf[dm_cst.DEPTH_MAP][depth_map_key][cst.INDEX_DEPTH_MAP_Y],
            overloaded_conf[dm_cst.DEPTH_MAP][depth_map_key][cst.INDEX_DEPTH_MAP_Z],
            overloaded_conf[dm_cst.DEPTH_MAP][depth_map_key][
                cst.INDEX_DEPTH_MAP_COLOR
            ],
            overloaded_conf[dm_cst.DEPTH_MAP][depth_map_key][
                cst.INDEX_DEPTH_MAP_MASK
            ],
            overloaded_conf[dm_cst.DEPTH_MAP][depth_map_key][
                cst.INDEX_DEPTH_MAP_CLASSIFICATION
            ],
            overloaded_conf[dm_cst.DEPTH_MAP][depth_map_key][
                cst.INDEX_DEPTH_MAP_PERFORMANCE_MAP
            ],
            overloaded_conf[dm_cst.DEPTH_MAP][depth_map_key][
                cst.INDEX_DEPTH_MAP_AMBIGUITY
            ],
            overloaded_conf[dm_cst.DEPTH_MAP][depth_map_key][
                cst.INDEX_DEPTH_MAP_FILLING
            ],
        )

    return overloaded_conf

def check_input_size(
    x_path, y_path, z_path, color, mask, classif, performance_map, ambiguity, filling
):
    """
    TODO
    """

    for path in [x_path, y_path, z_path]:
        if inputs.rasterio_get_nb_bands(path) != 1:
            raise RuntimeError("{} is not mono-band image".format(path))

    for path in [color, mask, classif, performance_map, ambiguity, filling]:
        if path is not None:
            if inputs.rasterio_get_size(x_path) != inputs.rasterio_get_size(
                path
            ):
                raise RuntimeError(
                    "The image {} and {} "
                    "do not have the same size".format(x_path, path)
                )
    
def modify_to_absolute_path(config_dir, overloaded_conf):
    """
    TODO
    """
    
    for depth_map_key in overloaded_conf[dm_cst.DEPTH_MAP]:
        depth_map = overloaded_conf[dm_cst.DEPTH_MAP][depth_map_key]
        for tag in [
            cst.INDEX_DEPTH_MAP_X,
            cst.INDEX_DEPTH_MAP_Y,
            cst.INDEX_DEPTH_MAP_Z,
            cst.INDEX_DEPTH_MAP_COLOR,
            cst.INDEX_DEPTH_MAP_MASK,
            cst.INDEX_DEPTH_MAP_CLASSIFICATION,
            cst.INDEX_DEPTH_MAP_PERFORMANCE_MAP,
            cst.INDEX_DEPTH_MAP_AMBIGUITY,
            cst.INDEX_DEPTH_MAP_FILLING,
        ]:
            if depth_map[tag] is not None:
                depth_map[tag] = make_relative_path_absolute(
                    depth_map[tag], config_dir
                )

    if overloaded_conf[cst.ROI] is not None:
        if isinstance(overloaded_conf[cst.ROI], str):
            overloaded_conf[cst.ROI] = make_relative_path_absolute(
                overloaded_conf[cst.ROI], config_dir
            )
