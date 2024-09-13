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
"""Tests for `cars_point_cloud_to_mesh` package."""

import os
import tempfile
from glob import glob

# cars_point_cloud_to_mesh imports
import cars_point_cloud_to_mesh

# pylint: disable=line-too-long
from cars_point_cloud_to_mesh.pipelines.point_cloud_to_mesh.cars_point_cloud_to_mesh import (  # noqa: B950,E501
    PointCloudToMeshPipeline,
)

from .helpers import absolute_path, assert_obj_mtl_link, temporary_dir


def test_cars_point_cloud_to_mesh():
    """Sample pytest cars_point_cloud_to_mesh module test function"""
    assert cars_point_cloud_to_mesh.__author__ == "CS Group"
    assert cars_point_cloud_to_mesh.__email__ == "yoann.steux@cs-soprasteria"


def test_end2end_gizeh():
    """
    End to end processing

    Test pipeline with a small part of Toulouse
    """

    with tempfile.TemporaryDirectory(dir=temporary_dir()) as directory:

        i_p = "input/gizeh/"
        input_json = {
            "pipeline": "point_cloud_to_mesh",
            "inputs": {
                "classification_buildings_description": ["bat"],
                "dsm_color": absolute_path(f"{i_p}clr.tif"),
                "point_clouds": {
                    "pc1": {
                        "x": absolute_path(f"{i_p}epi_pc_X.tif"),
                        "y": absolute_path(f"{i_p}epi_pc_Y.tif"),
                        "z": absolute_path(f"{i_p}epi_pc_Z.tif"),
                        "classification": absolute_path(
                            f"{i_p}epi_pc_classification.tif"
                        ),
                        "color": absolute_path(f"{i_p}epi_pc_color.tif"),
                    }
                },
            },
            "applications": {
                "create_dtm_mesh": {
                    "method": "percentile_of_unclassified_points",
                    "percentile": 5,
                },
                "point_cloud_to_polygons": {"radius": 1.5, "simplify": True},
                "group_close_polygons": {"radius": 1.5},
                "point_clouds_and_polygons_to_mesh": {
                    "method": "alpha_shape_delaunay_dtm_projection",
                    "alpha_shape_radius": 1.3,
                    "simplification_method": "douglas then visvalingam",
                    "douglas_threshold": 0.5,
                    "douglas_search_offset": 5,
                    "visvalingam_area": 15,
                },
            },
            "output": {"out_dir": f"{directory}", "out_epsg": 4978},
        }

        meshing_pipeline = PointCloudToMeshPipeline(input_json)
        meshing_pipeline.run()

        list_files_to_check = [
            f"{directory}/dtm_mesh.obj",
            f"{directory}/dtm_mesh.mtl",
            f"{directory}/color.png",
        ]
        for file in list_files_to_check:
            assert os.path.isfile(file)
            print(f"File {file} found")

        objs_path_glob = glob(f"{directory}/buildings/*.obj")

        assert len(list(objs_path_glob)) > 0
        print(
            "Number of created building objs" f" is {len(list(objs_path_glob))}"
        )

        for pobj in objs_path_glob:
            pmtl = pobj[:-3] + "mtl"

            assert os.path.isfile(pmtl)
            print(f"File {pobj} does have a corresponding mtl {pmtl}")
            assert_obj_mtl_link(pobj, pmtl)
