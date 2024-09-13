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
Top-level package for Cars Point cloud to Mesh.
"""
# pylint: disable=line-too-long

from importlib.metadata import version

import cars_point_cloud_to_mesh.applications
from cars_point_cloud_to_mesh.pipelines.point_cloud_to_mesh import (  # noqa: F401,E501
    cars_point_cloud_to_mesh,
)

# version through setuptools_scm when python3 > 3.8
try:
    __version__ = version("cars_point_cloud_to_mesh")
except Exception:  # pylint: disable=broad-except
    __version__ = "unknown"

__author__ = "CS Group"
__email__ = "yoann.steux@cs-soprasteria.com"
