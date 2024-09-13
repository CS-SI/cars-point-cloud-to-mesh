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
All the functions directly used by the point_to_point_based_grouping file
"""

import numpy as np
from numba import njit


@njit
def aabb_intersect(
    aabb1_min: np.ndarray,
    aabb1_max: np.ndarray,
    aabb2_min: np.ndarray,
    aabb2_max: np.ndarray,
) -> bool:
    """
    Returns whether or not the two Axis-Aligned Bounding Boxes overlap
    """
    intersection = np.logical_and(
        aabb1_min <= aabb2_max, aabb1_max >= aabb2_min
    )
    return np.all(intersection)


@njit
def in_distance(poly1: np.ndarray, poly2: np.ndarray, dist_max: float) -> bool:
    """
    Returns whether or not the two polygons are at a
    distance inferior to dist_max.
    Uses a distance from point to point, because our polygons are very dense.
    """
    poly_shape = poly1.shape[0]

    for i in range(poly_shape):
        pts = poly2 - poly1[i, :]
        # it seems like njit has no implementation for norm :(
        # dists = np.linalg.norm( pts, axis=1 )
        dists2 = pts[:, 0] * pts[:, 0] + pts[:, 1] * pts[:, 1]

        if np.any(dists2 <= dist_max * dist_max):
            return True

    return False


def compute_intersecting_tiles(polygons, radius):
    """
    Computes the set of tiles that overlap
    using their polygons' bounding boxes
    """
    tmin = np.zeros(polygons.shape + (2,))
    tmax = np.zeros(polygons.shape + (2,))

    for row in range(polygons.shape[0]):
        for col in range(polygons.shape[1]):
            if len(polygons[row, col].data["contours"]) <= 0:
                continue

            polygons[row, col].data["aabbs"] = np.array(
                polygons[row, col].data["aabbs"]
            )

            tmin[row, col] = np.min(
                polygons[row, col].data["aabbs"][:, 0], axis=0
            )
            tmax[row, col] = np.max(
                polygons[row, col].data["aabbs"][:, 1], axis=0
            )

    intersecting_tiles = []

    for row in range(polygons.shape[0] - 1):
        for col in range(polygons.shape[1] - 1):
            # inflate aabb by radius
            rcmin = tmin[row, col] - radius
            rcmax = tmax[row, col] + radius

            if aabb_intersect(
                rcmin, rcmax, tmin[row + 1, col], tmax[row + 1, col]
            ):
                intersecting_tiles.append([(row, col), (row + 1, col)])

            if aabb_intersect(
                rcmin, rcmax, tmin[row, col + 1], tmax[row, col + 1]
            ):
                intersecting_tiles.append([(row, col), (row, col + 1)])

            if aabb_intersect(
                rcmin, rcmax, tmin[row + 1, col + 1], tmax[row + 1, col + 1]
            ):
                intersecting_tiles.append([(row, col), (row + 1, col + 1)])

    return intersecting_tiles, (tmin, tmax)


def compute_intersecting_polygons(polygons, radius):
    """
    Computes the sets of polygons that are
    within a distance <= radius off one another
    """
    intersecting_tiles, (tmin, tmax) = compute_intersecting_tiles(
        polygons, radius
    )

    intersecting_polygons = []
    for tile1, tile2 in intersecting_tiles:

        t1min = tmin[tile1] - radius
        t1max = tmax[tile1] + radius

        t2min = tmin[tile2] - radius
        t2max = tmax[tile2] + radius

        potential_t1 = []
        potential_t2 = []

        for paabb_id, paabb in enumerate(polygons[tile1].data["aabbs"]):
            if aabb_intersect(t2min, t2max, paabb[0], paabb[1]):
                potential_t1.append(paabb_id)

        for paabb_id, paabb in enumerate(polygons[tile2].data["aabbs"]):
            if aabb_intersect(t1min, t1max, paabb[0], paabb[1]):
                potential_t2.append(paabb_id)

        # find intersecting polygons
        for i in potential_t1:

            imin = polygons[tile1].data["aabbs"][i][0] - radius
            imax = polygons[tile1].data["aabbs"][i][1] + radius

            for j in potential_t2:

                if not aabb_intersect(
                    imin,
                    imax,
                    polygons[tile2].data["aabbs"][j][0],
                    polygons[tile2].data["aabbs"][j][1],
                ):
                    continue

                if in_distance(
                    polygons[tile1].data["contours"][i],
                    polygons[tile2].data["contours"][j],
                    radius,
                ):
                    intersecting_polygons.append((tile1 + (i,), tile2 + (j,)))
    return intersecting_polygons


def connected_components_grouping(links):
    """
    Simple graph algorithm computing the
    connected components and grouping them

    Links is a list of n lists of links between the nth element
    and the other ones

    links = [ [1, 2], [2, 0], [0, 1, 3], [2], [] ] has three elements forming a
    triangle and a fourth only linked to one vertex, so all four are in the
    same group.
    The last element is alone, he is thus in his own group
    The expected output is either [0, 0, 0, 0, 1] or [1, 1, 1, 1, 0]
    """
    visited = [False for _ in links]
    group = [-1 for _ in range(len(links))]

    group_id = 0
    for i in range(len(links)):
        if visited[i]:
            continue
        visited[i] = True

        to_visit = [i]
        group[i] = group_id
        while len(to_visit) > 0:
            nxt = to_visit.pop(0)
            for neighbor in links[nxt]:
                if visited[neighbor]:
                    continue
                group[neighbor] = group_id
                visited[neighbor] = True
                to_visit.append(neighbor)

        group_id += 1

    return group


def compute_new_tiles(polygons, intersecting_polygons):
    """
    Uses the polygons and the set of intersecting
    polygons to make tiles based on the point cloud files
    that need to be opened.
    """
    new_tiles = {}
    groups = {}

    # fill new tiles with standalone polygons first
    for row in range(polygons.shape[0]):
        for col in range(polygons.shape[1]):
            tile_polys = [
                polygons[row, col].data["contours"][i]
                for i in range(len(polygons[row, col].data["contours"]))
                if i
                not in [
                    x
                    for (r, c, x), (r2, c2, x2) in intersecting_polygons
                    if r == row and c == col
                ]
                and i
                not in [
                    x2
                    for (r, c, x), (r2, c2, x2) in intersecting_polygons
                    if r2 == row and c2 == col
                ]
            ]

            if len(tile_polys) > 0:
                new_tiles[(row, col)] = tile_polys
                groups[(row, col)] = list(range(len(new_tiles[(row, col)])))

    list_inters_polys = []
    links = []
    for p1, p2 in intersecting_polygons:

        if p1 not in list_inters_polys:
            list_inters_polys.append(p1)
            links.append([])
            ip1 = len(list_inters_polys) - 1
        else:
            ip1 = list_inters_polys.index(p1)

        if p2 not in list_inters_polys:
            list_inters_polys.append(p2)
            links.append([])
            ip2 = len(list_inters_polys) - 1
        else:
            ip2 = list_inters_polys.index(p2)

        links[ip2].append(ip1)
        links[ip1].append(ip2)

    group = connected_components_grouping(links)
    if len(group) == 0:
        return new_tiles, groups

    polys_in_group = [[] for _ in range(max(group) + 1)]
    for poly_num, group_num in enumerate(group):
        polys_in_group[group_num].append(poly_num)

    for p_in_m in polys_in_group:
        keys = []
        for r, c, x in [list_inters_polys[i] for i in p_in_m]:
            if (r, c) not in keys:
                keys.append((r, c))

            # sort so that keys are always ordered the same
            keys.sort(key=lambda x: x[1])
            keys.sort(key=lambda x: x[0])
        key = keys[0]
        for k in keys[1:]:
            key += k

        if key in new_tiles:
            new_tiles[key] += [  # noqa: B909
                polygons[r, c].data["contours"][x]
                for r, c, x in [list_inters_polys[i] for i in p_in_m]
            ]
        else:
            new_tiles[key] = [  # noqa: B909
                polygons[r, c].data["contours"][x]
                for r, c, x in [list_inters_polys[i] for i in p_in_m]
            ]
        if key not in groups:
            groups[key] = []  # noqa: B909
            ngind = 0
        else:
            ngind = max(groups[key]) + 1

        groups[key] += [ngind for pi in p_in_m]
    return new_tiles, groups
