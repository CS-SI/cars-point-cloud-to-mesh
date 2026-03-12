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
Cars point cloud to meshes pipeline
"""

# Standard imports
from __future__ import print_function
import os

import logging

import cars_point_cloud_to_mesh.pipelines.parameters.depth_map_inputs_constants as dm_cst
from cars_point_cloud_to_mesh.applications.point_cloud_fusion import pc_tif_tools

# CARS imports
from cars.applications.application import Application
from cars.core.utils import safe_makedirs
from cars.core import constants as cst
from cars.orchestrator import orchestrator
from cars.pipelines.parameters import output_constants as out_cst
from cars.pipelines.pipeline import Pipeline
from cars.pipelines.pipeline_constants import (
    APPLICATIONS,
    INPUT,
    ORCHESTRATOR,
    OUTPUT,
    # PIPELINE,
)
from cars.pipelines.pipeline_template import PipelineTemplate
import cars_point_cloud_to_mesh.pipelines.parameters.depth_map_inputs as dm_inputs
from json_checker import Checker, Or, OptionalKey

PIPELINE = "point_cloud_to_mesh"

@Pipeline.register(
    "point_cloud_to_mesh",
)
class PointCloudToMeshPipeline(PipelineTemplate):
    """
    PointCloudToMeshPipeline
    """

    def __init__(self, conf, config_dir=None):
        """
        Creates pipeline

        :param pipeline_name: name of the pipeline.
        :type pipeline_name: str
        :param cfg: configuration
        :type cfg: dictionary
        :param config_dir: path to dir containing json
        :type config_dir: str
        """

        # Used conf
        self.used_conf = {}

        if config_dir is not None:
            config_dir = os.path.abspath(config_dir)
        self.config_dir = config_dir

        self.check_global_schema(conf)
        if PIPELINE in conf:
            self.check_pipeline_conf(conf)

        # Orchestrator
        self.used_conf[ORCHESTRATOR] = self.check_orchestrator(
            conf.get(ORCHESTRATOR, None)
        )

        # Inputs 
        self.inputs = self.check_inputs(conf, config_dir=config_dir)
        self.used_conf[INPUT] = self.inputs

        # Output 
        output = self.check_output(conf.get(OUTPUT, None))
        self.used_conf[OUTPUT] = output
        self.out_dir = output[out_cst.OUT_DIRECTORY]

        # Pipeline 
        pipeline_conf = conf.get(PIPELINE, {})
        self.used_conf[PIPELINE] = {}

        # Applications
        application_conf = self.check_applications(
            pipeline_conf.get(APPLICATIONS, {})
        )

        self.used_conf[PIPELINE][APPLICATIONS] = application_conf

    def check_pipeline_conf(self, conf):
        pipeline_schema = {
            OptionalKey(APPLICATIONS): dict,
        }

        checker_inputs = Checker(pipeline_schema)
        checker_inputs.validate(conf[PIPELINE])

    def check_inputs(self, conf, config_dir=None):

        # remove unexpected tags
        if "classification_buildings_description" not in conf[INPUT]:
            raise RuntimeError(
                "No classification_buildings_description provided"
            )
        if "dsm_color" not in conf[INPUT]:
            raise RuntimeError("no dsm color provideds")

        copied_classification_buildings_description = conf[INPUT][
            "classification_buildings_description"
        ]
        copied_dsm_color = conf[INPUT]["dsm_color"]
        del conf[INPUT]["classification_buildings_description"]
        del conf[INPUT]["dsm_color"]

        overloaded_conf = dm_inputs.check_depth_map_inputs(
            conf[INPUT], config_dir=config_dir
        )

        # add deleted data in input
        overloaded_conf["classification_buildings_description"] = (
            copied_classification_buildings_description
        )
        overloaded_conf["dsm_color"] = copied_dsm_color
        dm_schema = {
            cst.INDEX_DEPTH_MAP_X: str,
            cst.INDEX_DEPTH_MAP_Y: str,
            cst.INDEX_DEPTH_MAP_Z: str,
            cst.INDEX_DEPTH_MAP_COLOR: str,
            cst.INDEX_DEPTH_MAP_MASK: Or(str, None),
            cst.INDEX_DEPTH_MAP_FILLING: Or(str, None),
            # edges data
            cst.INDEX_DEPTH_MAP_EDGES_MASK: Or(str, None),
            cst.INDEX_DEPTH_MAP_EDGES_NORMALS: Or(str, None),
            cst.INDEX_DEPTH_MAP_EDGES_DEPTH_MAP: Or(str, None),
            cst.INDEX_DEPTH_MAP_EDGES_TILE_ID: Or(str, None),
            # also, config.json has classif named classification because
            # pc_inputs requires it :)
            cst.INDEX_DEPTH_MAP_CLASSIFICATION: str,  # require classif
            cst.INDEX_DEPTH_MAP_PERFORMANCE_MAP: Or(str, None),
            cst.INDEX_DEPTH_MAP_AMBIGUITY: Or(str, None),
            cst.INDEX_DEPTH_MAP_FILLING: Or(str, None),
            cst.INDEX_DEPTH_MAP_EPSG: Or(str, int, None),
        }
        checker_dm = Checker(dm_schema)

        for depth_map_key in overloaded_conf[dm_cst.DEPTH_MAP]:
            checker_dm.validate(
                overloaded_conf[dm_cst.DEPTH_MAP][depth_map_key]
            )

        return overloaded_conf

    def check_output(self, conf):
        """
        Check the output given

        :param conf: configuration of output
        :type conf: dict

        :return overloader output
        :rtype : dict
        """

        overloaded_conf = conf.copy()
        out_dir = conf[out_cst.OUT_DIRECTORY]
        out_dir = os.path.abspath(out_dir)
        # Ensure that output directory and its subdirectories exist
        safe_makedirs(out_dir)

        # Overload some parameters
        overloaded_conf[out_cst.OUT_DIRECTORY] = out_dir
        overloaded_conf[cst.EPSG] = conf.get(cst.EPSG, 4978)

        # Check schema
        output_schema = {
            out_cst.OUT_DIRECTORY: str,
            cst.EPSG: Or(int, str),
        }
        checker_output = Checker(output_schema)
        checker_output.validate(overloaded_conf)

        return overloaded_conf

    def check_applications(self, conf):
        """
        Check the given configuration for applications
        and instantiate default mesh pipeline applications.

        :param conf: configuration of applications
        :type conf: dict
        :return: normalized configuration used by applications
        :rtype: dict
        """

        # Safety: allow None
        if conf is None:
            conf = {}

        # Expected applications in mesh pipeline
        needed_applications = [
            "create_dtm_mesh",
            "point_cloud_to_polygons",
            "group_close_polygons",
            "point_clouds_and_polygons_to_mesh",
        ]

        # Validate that no unexpected application is provided
        for app_key in conf.keys():
            if app_key not in needed_applications:
                msg = (
                    f"No {app_key} application used in the "
                    "mesh default pipeline"
                )
                logging.error(msg)
                raise NameError(msg)

        # Normalize configuration
        used_conf = {}

        for app_key in needed_applications:
            used_conf[app_key] = conf.get(app_key, {}) or {}

        # Instantiate applications (always loaded)
        self.dtm_creation = Application(
            "create_dtm_mesh",
            cfg=used_conf["create_dtm_mesh"],
        )
        used_conf["create_dtm_mesh"] = self.dtm_creation.get_conf()

        self.pcd_to_grouping_polys = Application(
            "point_cloud_to_polygons",
            cfg=used_conf["point_cloud_to_polygons"],
        )
        used_conf["point_cloud_to_polygons"] = (
            self.pcd_to_grouping_polys.get_conf()
        )

        self.group_grouping_polys = Application(
            "group_close_polygons",
            cfg=used_conf["group_close_polygons"],
        )
        used_conf["group_close_polygons"] = (
            self.group_grouping_polys.get_conf()
        )

        self.grouping_polys_to_mesh = Application(
            "point_clouds_and_polygons_to_mesh",
            cfg=used_conf["point_clouds_and_polygons_to_mesh"],
        )
        used_conf["point_clouds_and_polygons_to_mesh"] = (
            self.grouping_polys_to_mesh.get_conf()
        )

        return used_conf

    def run(self, log_dir=None):
        """
        Run pipeline
        """
        if log_dir is None:
            log_dir = os.path.join(self.out_dir, "logs")

        # start cars orchestrator
        with orchestrator.Orchestrator(
            orchestrator_conf=self.used_conf[ORCHESTRATOR],
            out_dir=self.out_dir,
        ) as cars_orchestrator:

            # generate tiling for the point cloud
            list_depth_map = pc_tif_tools.generate_point_clouds(
                self.inputs[dm_cst.DEPTH_MAP], cars_orchestrator, tile_size=1000
            )

            # generate tiling for the point cloud
            # to be used in the dtm generation
            list_depth_map_dtm_gen = pc_tif_tools.generate_point_clouds(
                self.inputs[dm_cst.DEPTH_MAP],
                cars_orchestrator,
                tile_size=self.dtm_creation.used_config["dtm_precision"],
            )

            dtm_mesh = self.dtm_creation.run(
                list_depth_map_dtm_gen,
                self.out_dir,
                out_epsg=self.used_conf[OUTPUT][cst.EPSG],
                classification_buildings_description=self.inputs[
                    "classification_buildings_description"
                ],
                dsm_color=self.inputs["dsm_color"],
                orchestrator=cars_orchestrator,
            )

            # Create point cloud groups in the form of polygons
            grouping_polygons = self.pcd_to_grouping_polys.run(
                # point clouds paths
                list_depth_map,
                classification_buildings_description=self.inputs[
                    "classification_buildings_description"
                ],
                orchestrator=cars_orchestrator,
            )

            # Actually compute grouping_polygons
            cars_orchestrator.breakpoint()

            # Use the polygon grouping algorithm
            # pylint: disable=unused-variable
            tiles, groups = self.group_grouping_polys.run(grouping_polygons)

            # Create meshes from the groups of polygons
            self.grouping_polys_to_mesh.run(
                # polygons and files associated
                list_depth_map,
                tiles,
                groups,
                dtm_mesh,
                self.out_dir,
                out_epsg=self.used_conf[OUTPUT][cst.EPSG],
                classification_buildings_description=self.inputs[
                    "classification_buildings_description"
                ],
                orchestrator=cars_orchestrator,
            )
