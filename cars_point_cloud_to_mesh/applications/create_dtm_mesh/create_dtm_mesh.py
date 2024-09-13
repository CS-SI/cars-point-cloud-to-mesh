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
this module contains the abstract dtm mesh creation application class.
"""

import logging
from abc import ABCMeta, abstractmethod
from typing import Dict

from cars.applications.application import Application
from cars.applications.application_template import ApplicationTemplate


@Application.register("create_dtm_mesh")
class CreateDtmMesh(ApplicationTemplate, metaclass=ABCMeta):
    """
    The abstract class for the dtm mesh creation algorithms
    """

    available_applications: Dict = {}
    default_application = "percentile_of_unclassified_points"

    def __new__(cls, conf=None):  # pylint: disable=W0613
        """
        Return the required application
        :raises:
         - KeyError when the required application is not registered

        :param conf: configuration
        :return: a application_to_use object
        """

        polygon_method = cls.default_application
        if bool(conf) is False:
            logging.info("Dtm mesh method not specified, default is used")
        else:
            polygon_method = conf.get("method", cls.default_application)

        if polygon_method not in cls.available_applications:
            log_message = (
                f"No dtm mesh creation application named {polygon_method} "
                "registered"
            )
            logging.error(log_message)
            raise KeyError(log_message)

        log_message = (
            f"The CreateDtmMesh({polygon_method}) application will be used"
        )
        logging.info(log_message)

        return super(CreateDtmMesh, cls).__new__(
            cls.available_applications[polygon_method]
        )

    def __init_subclass__(cls, short_name, **kwargs):  # pylint: disable=E0302
        super().__init_subclass__(**kwargs)
        cls.available_applications[short_name] = cls

    def __init__(self, conf=None):
        """
        Init function

        :param conf: configuration
        :return: an application_to_use object
        """

        super().__init__(conf=conf)

    @abstractmethod
    def run(self, point_clouds, **kwargs):
        """
        Run the dtm mesh creation algorithm
        """
