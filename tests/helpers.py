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
Helpers shared testing generic module:
contains global shared generic functions for tests/*.py
"""

import logging

# Standard imports
import os
import tempfile

# Specific values
# 0 = valid pixels
# 255 = value used as no data during the resampling in the epipolar geometry
PROTECTED_VALUES = [255]


def cars_path():
    """
    Return root of cars source directory
    One level down from tests
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def absolute_path(data_path):
    """
    Return a full absolute path to test data
    environment variable.
    """
    data_folder = os.path.join(os.path.dirname(__file__), "data")
    return os.path.join(data_folder, data_path)


def temporary_dir():
    """
    Returns path to temporary dir from CARS_TEST_TEMPORARY_DIR environment
    variable. Defaults to default temporary directory
    (/tmp or TMPDIR environment variable)
    """
    if "CARS_TEST_TEMPORARY_DIR" not in os.environ:
        # return default tmp dir
        logging.info(
            "CARS_TEST_TEMPORARY_DIR is not set, "
            "cars will use default temporary directory instead"
        )
        return tempfile.gettempdir()
    # return env defined tmp dir
    return os.environ["CARS_TEST_TEMPORARY_DIR"]


def assert_obj_mtl_link(obj_path, mtl_path):
    """
    Guarantees that the obj links to the mtl, and that
    all the materials used in the obj are present within
    the mlt.
    """

    objmtls = []
    with open(obj_path, encoding="utf8") as f:

        line = f.readline()
        while line and not line.startswith("mtllib"):
            line = f.readline()

        assert os.path.basename(mtl_path) in line
        print(f"Link to {mtl_path} found in {obj_path}")

        f.seek(0)

        line = f.readline()
        while line:
            while line and not line.startswith("usemtl"):
                line = f.readline()

            mtlname = line[7:-1].strip()
            objmtls.append(mtlname)
            line = f.readline()

        assert len(objmtls) > 0
        print(f"Obj {obj_path} uses at least one material")

    mtlmtls = []
    with open(mtl_path, encoding="utf8") as f:

        line = f.readline()
        while line:
            while line and not line.startswith("newmtl"):
                line = f.readline()

            mtlname = line[7:-1].strip()
            mtlmtls.append(mtlname)
            line = f.readline()

        assert len(mtlmtls) > 0
        print(f"Mtl {mtl_path} contains at least one material")

    for om in objmtls:
        assert om in mtlmtls
    print(f"All materials used in {obj_path} are present in {mtl_path}")
