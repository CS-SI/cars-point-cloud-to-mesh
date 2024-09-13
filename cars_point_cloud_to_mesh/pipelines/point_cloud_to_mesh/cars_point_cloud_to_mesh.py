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

import cars.pipelines.point_clouds_to_dsm.pc_constants as pc_cst

# CARS imports
from cars.applications.application import Application
from cars.applications.point_cloud_fusion import pc_tif_tools
from cars.core import constants as cst
from cars.orchestrator import orchestrator
from cars.pipelines.pipeline import Pipeline
from cars.pipelines.pipeline_constants import (
    INPUTS,
    ORCHESTRATOR,
    OUTPUT,
    PIPELINE,
)
from cars.pipelines.pipeline_template import PipelineTemplate
from cars.pipelines.point_clouds_to_dsm import pc_inputs
from json_checker import Checker, Or


@Pipeline.register(
    "point_cloud_to_mesh",
)
class PointCloudToMeshPipeline(PipelineTemplate):
    """
    PointCloudToMeshPipeline
    """

    def __init__(self, conf, config_json_dir=None):
        """
        Creates pipeline

        :param pipeline_name: name of the pipeline.
        :type pipeline_name: str
        :param cfg: configuration
        :type cfg: dictionary
        :param config_json_dir: path to dir containing json
        :type config_json_dir: str
        """

        # Used conf
        self.used_conf = {}

        # Pipeline
        self.used_conf[PIPELINE] = conf.get(PIPELINE, "point_cloud_to_mesh")

        # Default point clouds check
        self.inputs = self.check_inputs(conf, config_json_dir=config_json_dir)
        self.used_conf[INPUTS] = self.inputs

        # Check output
        self.output = self.check_output(conf.get(OUTPUT, None))

        self.orchestrator_conf = self.check_orchestrator(
            conf.get(ORCHESTRATOR, None)
        )
        self.used_conf[ORCHESTRATOR] = self.orchestrator_conf

        self.check_applications(conf["applications"])

    def check_inputs(self, conf, config_json_dir=None):

        # remove unexpected tags
        if "classification_buildings_description" not in conf[INPUTS]:
            raise RuntimeError(
                "No classification_buildings_description provided"
            )
        if "dsm_color" not in conf[INPUTS]:
            raise RuntimeError("no dsm color provideds")

        copied_classification_buildings_description = conf[INPUTS][
            "classification_buildings_description"
        ]
        copied_dsm_color = conf[INPUTS]["dsm_color"]
        del conf[INPUTS]["classification_buildings_description"]
        del conf[INPUTS]["dsm_color"]

        overloaded_conf = pc_inputs.check_point_clouds_inputs(
            conf[INPUTS], config_json_dir=config_json_dir
        )

        # add deleted data in input
        overloaded_conf["classification_buildings_description"] = (
            copied_classification_buildings_description
        )
        overloaded_conf["dsm_color"] = copied_dsm_color

        pc_schema = {
            cst.X: str,
            cst.Y: str,
            cst.Z: str,
            # also, config.json has classif named classification because
            # pc_inputs requires it :)
            cst.POINTS_CLOUD_CLASSIF_KEY_ROOT: str,  # require classif
            cst.POINTS_CLOUD_MSK: Or(str, None),
            cst.POINTS_CLOUD_CONFIDENCE_KEY_ROOT: Or(dict, None),
            cst.POINTS_CLOUD_CLR_KEY_ROOT: str,
            cst.POINTS_CLOUD_FILLING_KEY_ROOT: Or(str, None),
            cst.PC_EPSG: Or(str, int, None),
        }
        checker_pc = Checker(pc_schema)

        for point_cloud_key in overloaded_conf[pc_cst.POINT_CLOUDS]:
            checker_pc.validate(
                overloaded_conf[pc_cst.POINT_CLOUDS][point_cloud_key]
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

        overloaded_conf = conf

        overloaded_conf["out_dir"] = conf.get("out_dir", None)
        overloaded_conf["out_epsg"] = conf.get("out_epsg", 4978)
        output_schema = {"out_dir": str, "out_epsg": Or(int, str)}
        checker_output = Checker(output_schema)
        checker_output.validate(overloaded_conf)

        return overloaded_conf

    def check_applications(self, conf):
        """
        Check the given configuration for applications

        :param conf: configuration of applications
        :type conf: dict
        """

        self.dtm_creation = Application(
            "create_dtm_mesh",
            cfg=conf.get("create_dtm_mesh", {}),
        )

        self.pcd_to_grouping_polys = Application(
            "point_cloud_to_polygons",
            cfg=conf.get("point_cloud_to_polygons", {}),
        )

        self.group_grouping_polys = Application(
            "group_close_polygons", cfg=conf.get("group_close_polygons", {})
        )

        self.grouping_polys_to_mesh = Application(
            "point_clouds_and_polygons_to_mesh",
            cfg=conf.get("point_clouds_and_polygons_to_mesh", {}),
        )

        return conf

    def run(self):
        """
        Run pipeline
        """

        # start cars orchestrator
        with orchestrator.Orchestrator(
            orchestrator_conf=self.orchestrator_conf,
        ) as cars_orchestrator:

            # generate tiling for the point cloud
            list_point_clouds = pc_tif_tools.generate_point_clouds(
                self.inputs["point_clouds"], cars_orchestrator, tile_size=1000
            )

            # generate tiling for the point cloud
            # to be used in the dtm generation
            list_point_clouds_dtm_gen = pc_tif_tools.generate_point_clouds(
                self.inputs["point_clouds"],
                cars_orchestrator,
                tile_size=self.dtm_creation.used_config["dtm_precision"],
            )

            dtm_mesh = self.dtm_creation.run(
                list_point_clouds_dtm_gen,
                self.output["out_dir"],
                out_epsg=self.output["out_epsg"],
                classification_buildings_description=self.inputs[
                    "classification_buildings_description"
                ],
                dsm_color=self.inputs["dsm_color"],
                orchestrator=cars_orchestrator,
            )

            # Create point cloud groups in the form of polygons
            grouping_polygons = self.pcd_to_grouping_polys.run(
                # point clouds paths
                list_point_clouds,
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
                list_point_clouds,
                tiles,
                groups,
                dtm_mesh,
                self.output["out_dir"],
                out_epsg=self.output["out_epsg"],
                classification_buildings_description=self.inputs[
                    "classification_buildings_description"
                ],
                orchestrator=cars_orchestrator,
            )
