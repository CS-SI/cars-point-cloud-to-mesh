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
All the functions directly used by the external_alpha_shape file
"""

import numpy as np
from numba import njit


class Grid:
    """
    An acceleration structure. Essentially a regular grid.
    Insertion is O(1)
    Radius search is O( ceil(r/grid_factor+2)² * mean_nb_nodes_per_leaf )
    which reduces to O(9*mean_nb_nodes_per_leaf) for r <= grid_factor
    """

    def __init__(self, pts_full, grid_factor=None):
        if grid_factor is None:
            grid_factor = [2, 2]
        pts = pts_full[:, :2]
        self.points = pts_full
        self.grid_base = pts.min(axis=0)
        self.span = pts.max(axis=0) - self.grid_base
        self.grid_factor = np.array(grid_factor)
        self.grid_size = np.ceil(self.span / self.grid_factor + 1).astype(int)

        self.grid = np.empty(self.grid_size, dtype=object)

        for i in np.ndindex(self.grid.shape):
            self.grid[i] = []

        grid_indices = ((pts - self.grid_base) / self.grid_factor).astype(int)
        for i, index in enumerate(grid_indices):
            self.grid[index[0], index[1]].append(i)

    def grid_pos(self, point):
        return self.grid_pos_float(point).astype(int)

    def grid_pos_float(self, point):
        return (point[:2] - self.grid_base) / self.grid_factor

    def world_pos(self, point):
        return point[:2] * self.grid_factor + self.grid_base

    def mean_leaves_nb(self):
        sum_leaves_size = np.sum([len(cell) for cell in self.grid.flat])
        nb = np.sum([1 if len(cell) > 0 else 0 for cell in self.grid.flat])
        return sum_leaves_size / nb if nb != 0 else 0

    def get_search_d(self, radius):
        return np.ceil(radius / self.grid_factor).astype(int)

    def get_pts_by_cell_around(self, pt, r):
        """
        returns an iterator over cells that are intersecting
        with a circle of radius r
        """
        dxy = self.get_search_d(r)
        x, y = self.grid_pos(pt)

        neighbors_iterator = self.grid[
            max(0, x - dxy[0]) : x + dxy[0] + 1,
            max(0, y - dxy[1]) : y + dxy[1] + 1,
        ].flat

        return neighbors_iterator

    def leaves_nb_grid(self):
        """
        returns the number of elements per leaf as a 2D array
        for visualization/debugging
        """
        grid_leaves = np.zeros(self.grid.shape)
        for pos in np.ndindex(grid_leaves.shape):
            grid_leaves[pos] = len(self.grid[pos])

        return grid_leaves

    def keep_only_masked(self, mask):
        """
        Removes all the points that are not selected by the mask.
        Basically resets and recomputes the grid with a subset of
        the points previously present.
        """

        pts_full = self.points[mask != 0]
        grid_factor = self.grid_factor

        self.points = pts_full

        pts = pts_full[:, :2]
        self.grid_base = pts.min(axis=0)
        self.span = pts.max(axis=0) - self.grid_base

        self.grid_factor = np.array(grid_factor)
        self.grid_size = np.ceil(self.span / self.grid_factor + 1).astype(int)

        self.grid = np.empty(self.grid_size, dtype=object)

        for i in np.ndindex(self.grid.shape):
            self.grid[i] = []

        grid_indices = ((pts - self.grid_base) / self.grid_factor).astype(int)
        for i, index in enumerate(grid_indices):
            self.grid[index[0], index[1]].append(i)


@njit
def angle_between(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Returns the signed angle between two vectors
    """
    x: float = np.dot(vec1, vec2)
    y: float = vec1[0] * vec2[1] - vec1[1] * vec2[0]

    theta: float = np.arctan2(y, x)

    theta = ((theta % (2 * np.pi)) + 2 * np.pi) % (2 * np.pi)

    return theta


@njit
def circle_angle(a: np.ndarray, b: np.ndarray, r: float, u: np.ndarray):
    """
    Given two points A and B, returns the angle
    between vectors U and V.

    U is an arbitrary vector
    V is the vector going from A to X, the center of
    the circle of radius R going through A and B,
    obtained by making the circle roll around A clockwise
    """
    ab = b - a
    m = (a + b) * 0.5
    d = np.linalg.norm(ab)

    n = np.array([-ab[1], ab[0]], dtype=np.float32)
    dn = np.linalg.norm(n)
    n = n / dn

    a2 = (d / 2) * (d / 2)
    c2 = r * r

    b = np.sqrt(c2 - a2)

    # always keep only the first option to simulate rolling
    x1 = m + b * n
    a1 = angle_between((x1 - a).astype(np.float32), u.astype(np.float32))
    # X2: np.ndarray = M - b * N
    # A2: np.float32 = angle_between( X2 - A, U )

    return (a1, x1)


@njit
def distance(pt1: np.ndarray, pt2: np.ndarray) -> np.float32:
    return np.linalg.norm(pt1 - pt2)


@njit
def distances2(pts: np.ndarray, pt2: np.ndarray) -> np.ndarray:
    return np.sum(np.square(pts - pt2), axis=1)


@njit
def point_in_poly(pt: np.ndarray, poly: np.ndarray) -> bool:
    """
    Shoots an horizontal ray towards the right
    If an odd number of edges are met, the point is in the polygon
    """
    # pt.shape : (2,)
    # poly.shape : (n, 2)

    intersections = 0

    for i in range(len(poly) - 1):
        segment_start = poly[i]
        segment_end = poly[i + 1]

        # Vérifie si le segment intersecte le rayon horizontal
        if (segment_start[1] > pt[1]) != (segment_end[1] > pt[1]) and (
            pt[0]
            < (segment_end[0] - segment_start[0])
            * (pt[1] - segment_start[1])
            / (segment_end[1] - segment_start[1])
            + segment_start[0]
        ):
            intersections += 1

    return intersections % 2 == 1


def trace_contour(
    pts: np.ndarray,
    mask: np.ndarray,
    id_start_pt: int,
    start_vector: np.ndarray,
    alpha_shape_diameter: float,
    grid: Grid = None,
):
    """
    Traces the contour of the point cloud pts, starting from
    id_start_pt with orientation start_vector using a disk
    of diameter alpha_shape_diameter

    Mask makes it so fewer calculation
    will be performed in some cases
    """

    if grid is None:
        g_grid = Grid(pts, [alpha_shape_diameter, alpha_shape_diameter])
    else:
        g_grid = grid

    start_point = id_start_pt

    polygon = [start_point]

    curr_point = start_point
    curr_pos = pts[curr_point, :]
    curr_vec = start_vector

    # find the primary contour
    while True:

        neighbors_iterator = g_grid.get_pts_by_cell_around(
            curr_pos, alpha_shape_diameter
        )

        least_angle = 7
        least_neighbor = -1
        least_circle_center = -1

        for nblist in neighbors_iterator:
            for neighbor in nblist:

                if mask[neighbor] != 0:
                    continue
                if neighbor == curr_point:
                    continue

                dist = distance(curr_pos, pts[neighbor])
                if dist >= alpha_shape_diameter:
                    continue

                angle, circle_center = circle_angle(
                    curr_pos, pts[neighbor], alpha_shape_diameter / 2, curr_vec
                )
                if 0 < angle < least_angle:
                    least_angle = angle
                    least_neighbor = neighbor
                    least_circle_center = circle_center

        if least_neighbor == -1:
            if len(polygon) < 2:
                return None
            least_neighbor = polygon[-2]

        polygon.append(least_neighbor)

        curr_point = least_neighbor  # neigh becomes curr
        curr_pos = pts[least_neighbor]  # update pos
        curr_vec = (
            least_circle_center - curr_pos
        )  # curr_vec = neigh to center found

        # if next point is also the start, break
        if curr_point == start_point:
            break

    return polygon


@njit
def simplify_criterion(v: np.ndarray, w: np.ndarray) -> bool:
    """
    The criterion for simplifying a polygon edge.
    In its own function to be accelerated via njit
    """
    if np.all(v + w == 0):
        return False
    theta = np.arctan2(w[1] * v[0] - w[0] * v[1], w[0] * v[0] + w[1] * v[1])
    return theta < 0  # <0 is notch, >0 is ear


def simplify(contours):
    """
    Simplifies the contours in-place by deleting "notches",
    to consume less memory while mostly keeping the polygon shape
    """
    for contour in contours:

        i = 1
        while i < len(contour) - 1:
            v = contour[i - 1] - contour[i]
            w = contour[i + 1] - contour[i]

            if simplify_criterion(v, w):
                contour = np.delete(contour, i, axis=0)
            else:
                i += 1

    return contours
