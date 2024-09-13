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

# pylint: disable=C0302
# pylint: disable=line-too-long
# pylint: disable=import-error

import logging
import os

import numpy as np
import rasterio as rio
from cars.applications.holes_detection.holes_detection_tools import (
    classif_to_stacked_array,
)
from rasterio.plot import reshape_as_raster


def compute_ndvi(color):
    """
    Computes the ndvi color index for the given colors (RGBNir)
    with color being a (4, ...) array
    """
    top = color[3] - color[0]
    bot = color[3] + color[0]
    mask = bot != 0

    ndvi = np.zeros(top.shape)
    ndvi[mask] = top[mask] / bot[mask]
    return ndvi


def compute_ndwi(color):
    """
    Computes the ndwi color index for the given colors (RGBNir)
    with color being a (4, ...) array
    """
    top = color[3] - color[1]
    bot = color[3] + color[1]
    mask = bot != 0

    ndwi = np.zeros(top.shape)
    ndwi[mask] = top[mask] / bot[mask]
    return ndwi


def evaluate_z_around(point_cloud, mask, point, percentile, offset=75):
    """
    Looks at the points around p and computes
    the percentile of the z component, to get
    a sampled height around p
    """

    min_x = max(point[0] - offset, 0)
    max_x = min(point[0] + offset, point_cloud["z"].values.shape[0] - 1)
    min_y = max(point[1] - offset, 0)
    max_y = min(point[1] + offset, point_cloud["z"].values.shape[1] - 1)

    zdata = point_cloud["z"].values[min_x : max_x + 1, min_y : max_y + 1]
    zdata = zdata[mask[min_x : max_x + 1, min_y : max_y + 1]]

    height = np.nanpercentile(zdata, percentile)

    return height


def get_closest_point_id(valid_pts, eval_pt):
    """
    Returns the id of the closest
    point to eval_pt in valid_pts
    """

    dist_x = valid_pts[:, 0] - eval_pt[0]
    dist_y = valid_pts[:, 1] - eval_pt[1]
    valid_pts_dist = dist_x * dist_x + dist_y * dist_y

    return np.argmin(valid_pts_dist)


def get_unclassified_selector(point_cloud, building_index=None):
    """
    Returns the selector of a point cloud based
    on the classifications given and computed
    """
    mask_selector = np.logical_not(np.isnan(point_cloud["x"].values))

    ndvi = np.zeros(mask_selector.shape)
    ndwi = np.zeros(mask_selector.shape)

    ndvi[mask_selector] = compute_ndvi(
        point_cloud["color"].values[:, mask_selector]
    )
    ndwi[mask_selector] = compute_ndwi(
        point_cloud["color"].values[:, mask_selector]
    )

    classif = classif_to_stacked_array(point_cloud, building_index)
    classif_selector = classif > 0
    vegetation_selector = ndvi > 0.2  # arbitrary value
    water_selector = ndwi < -0.5  # arbitrary value

    classified_selector = np.logical_or(
        classif_selector, np.logical_or(vegetation_selector, water_selector)
    )

    unclassified_selector = np.logical_and(
        mask_selector, np.logical_not(classified_selector)
    )
    return unclassified_selector


def export_mtl_file(
    path, filename, texture_path_with_filename_and_ext, id_thread, classif_data
):
    """
    Exports a mtl file with an optional texture image and
    a way to manually name materials if needed
    """
    # save mtl
    with open(path + "/" + filename + ".mtl", "w", encoding="ascii") as mtl:
        if classif_data is None:
            mtl.write(f"newmtl Material.{id_thread:03}\r\n")
            mtl.write("Ka 1.000 1.000 1.000\r\n")
            mtl.write("Kd 1.000 1.000 1.000\r\n")
            mtl.write("Ks 0.000 0.000 0.000\r\n")
            mtl.write("d 1.0\r\n")
            mtl.write("illum 2\r\n")
            if texture_path_with_filename_and_ext is not None:
                mtl.write(f"map_Kd {texture_path_with_filename_and_ext}\r\n")
        else:
            for name, color, tex in zip(
                classif_data["classes"],
                classif_data["classes_color"],
                classif_data["classes_texture"],
                strict=True,
            ):
                mtl.write(f"newmtl Mtl.{name}\r\n")
                mtl.write(f"Ka {color[0]} {color[1]} {color[2]}\r\n")
                mtl.write(f"Kd {color[0]} {color[1]} {color[2]}\r\n")
                mtl.write("Ks 0.000 0.000 0.000\r\n")
                mtl.write("d 1.0\r\n")
                mtl.write("illum 2\r\n")
                if tex is not None:
                    mtl.write(f"map_Kd {tex}\r\n")

    return filename + ".mtl"


def export_obj(
    path,
    filename,
    texture_path_with_filename_and_ext,
    vertices,
    uv_maps,
    triangles,
    mtl_file_path=None,
    classif_data=None,
    id_thread=-1,
):
    """
    Exports a mesh to an obj in plain text format,
    with a material pointing to an image file

    uvs can be None, in which case no uv will be saved

    mtl_file_path is str or None
    If str, it will be used as the reference to the material file
    If None, a new material file will be created
    """
    if id_thread == -1:
        id_thread = 1
    os.makedirs(os.path.dirname(path + "/" + filename + ".tmp"), exist_ok=True)

    if mtl_file_path is None:
        # save mtl
        mtl_file_path = export_mtl_file(
            path,
            filename,
            texture_path_with_filename_and_ext,
            id_thread,
            classif_data,
        )

    # save obj
    with open(path + "/" + filename + ".obj", "w+", encoding="ascii") as obj:

        obj.write(f"mtllib {mtl_file_path}\r\n")

        if classif_data is None:
            obj.write(f"usemtl Material.{id_thread:03}\r\n")

        # save vertices
        for point in vertices:
            obj.write(f"v {point[0]} {point[1]} {point[2]}\r\n")

        # save uvs
        for uv_map in uv_maps:
            obj.write(f"vt {uv_map[0]} {uv_map[1]}\r\n")

        # save triangles
        if classif_data is None:
            for tri in triangles:
                obj.write(
                    f"f {tri[0]+1}/{tri[0]+1} {tri[1]+1}/{tri[1]+1}"
                    f" {tri[2]+1}/{tri[2]+1}\r\n"
                )
        else:
            for icl, current_class in enumerate(classif_data["classes"]):
                obj.write(f"usemtl Mtl.{current_class}\r\n")
                for itri, tri in enumerate(triangles):
                    if classif_data["triangles_classification"][itri] == icl:
                        obj.write(
                            f"f {tri[0]+1}/{tri[0]+1} {tri[1]+1}/{tri[1]+1}"
                            f" {tri[2]+1}/{tri[2]+1}\r\n"
                        )


def tif_to_png(in_file, out_file, scale_factor=1):
    """
    Takes a path to a tif file in_file, and turns
    its data interpreted as a color into a png,
    saved in out_file
    """

    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    with rio.open(in_file) as in_data:

        msk = in_data.read_masks(1)

        rgb = np.concatenate(
            (
                in_data.read(1)[:, :, np.newaxis],
                in_data.read(2)[:, :, np.newaxis],
                in_data.read(3)[:, :, np.newaxis],
            ),
            axis=-1,
        )

        low_bound = np.percentile(rgb[msk > 0], 5, axis=0)
        high_bound = np.percentile(rgb[msk > 0], 95, axis=0)

        rgb = (rgb - low_bound) / (high_bound - low_bound)
        rgb = np.clip(rgb, 0, 1) * 255
        rgb[msk == 0] = 0

        rgb = rgb.astype(np.uint8)

        with rio.open(
            out_file,
            "w",
            driver="PNG",
            height=rgb.shape[0] * scale_factor,
            width=rgb.shape[1] * scale_factor,
            count=3,
            dtype="uint8",
            nodata=None,
            resampling=rio.enums.Resampling.bilinear,
        ) as out_data:
            out_data.write(reshape_as_raster(rgb))


def get_relevant_info(file):
    """
    Helper function returning the information needed in
    the run function of the application. In this case,
    the inverse projection matrix and the file's epsg
    """
    with rio.open(file) as clr_file:

        if not clr_file.crs.is_epsg_code:
            message = (
                f"The CRS of the color file {file} "
                "is not supported, as it is not linked to an EPSG code."
                f"CRS found : {clr_file.crs}"
            )
            logging.error(message)
            raise RuntimeError(message)

        trans = clr_file.transform
        mat_tr = np.array(
            [
                [trans[0], trans[1], trans[2]],
                [trans[3], trans[4], trans[5]],
                [0, 0, 1],
            ]
        )
        inv_mat_tr = np.linalg.inv(mat_tr)

        inv_mat_tr[0] /= clr_file.width
        inv_mat_tr[1] /= clr_file.height

    return inv_mat_tr, clr_file.crs.to_epsg()


def update_lrud(lrud, bbpts):
    """
    Helper function, keeps track of the bounds of a point cloud
    when going through each point one by one in a loop.
    """

    if bbpts["left"] is None:
        return lrud

    if lrud["l"] is None or lrud["l"][0] > bbpts["left"][0]:
        lrud["l"] = bbpts["left"]

    if lrud["r"] is None or lrud["r"][0] < bbpts["right"][0]:
        lrud["r"] = bbpts["right"]

    if lrud["u"] is None or lrud["u"][1] < bbpts["up"][1]:
        lrud["u"] = bbpts["up"]

    if lrud["d"] is None or lrud["d"][1] > bbpts["down"][1]:
        lrud["d"] = bbpts["down"]

    return lrud


def filter_triangles_of_mesh(dtm_mesh, ratio):
    """
    Filters a mesh by removing triangles
    whose edge lengths differ significantly
    from the norm
    """

    dists = []
    for point1, point2, point3 in dtm_mesh["triangles"]:
        dists.append(
            np.linalg.norm(
                dtm_mesh["vertices"][point1] - dtm_mesh["vertices"][point2]
            )
        )
        dists.append(
            np.linalg.norm(
                dtm_mesh["vertices"][point1] - dtm_mesh["vertices"][point3]
            )
        )
        dists.append(
            np.linalg.norm(
                dtm_mesh["vertices"][point2] - dtm_mesh["vertices"][point3]
            )
        )
    med_dist = np.median(np.array(dists))

    new_tris = []
    for point1, point2, point3 in dtm_mesh["triangles"]:
        d12 = np.linalg.norm(
            dtm_mesh["vertices"][point1] - dtm_mesh["vertices"][point2]
        )
        d13 = np.linalg.norm(
            dtm_mesh["vertices"][point1] - dtm_mesh["vertices"][point3]
        )
        d23 = np.linalg.norm(
            dtm_mesh["vertices"][point2] - dtm_mesh["vertices"][point3]
        )

        if (
            d12 > ratio * med_dist
            or d23 > ratio * med_dist
            or d13 > ratio * med_dist
        ):
            continue

        new_tris.append([point1, point2, point3])

    return new_tris
