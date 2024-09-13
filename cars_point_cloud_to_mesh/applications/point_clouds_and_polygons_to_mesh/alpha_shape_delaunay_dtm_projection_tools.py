# pylint: disable=too-many-lines
# !/usr/bin/env python
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
All the functions used by the alpha_shape_delaunay_dtm_projection file
"""

import numpy as np
import triangle as trlib
from cars.applications.holes_detection.holes_detection_tools import (
    classif_to_stacked_array,
)
from cars.core import projection
from scipy.signal import find_peaks

# https://github.com/pylint-dev/pylint/issues/3273
from scipy.spatial import ConvexHull  # pylint: disable=no-name-in-module

from ..point_cloud_to_polygons import external_alpha_shape_tools as east


def get_grid(point_clouds, grid_size, building_index):
    """
    Creates a grid object with all
    the points in of point_clouds
    """

    pts = []
    for pcd in point_clouds:
        classif = classif_to_stacked_array(pcd, building_index)
        classified_selector = np.logical_and(
            classif.flatten() > 0,
            np.logical_not(np.isnan(pcd["x"].values.flatten())),
        )
        t_pts = np.column_stack(
            (
                pcd["x"].values.flatten()[classified_selector],
                pcd["y"].values.flatten()[classified_selector],
                pcd["z"].values.flatten()[classified_selector],
            )
        )
        t_pts = projection.points_cloud_conversion(
            t_pts, pcd.attrs["epsg"], 32631
        )
        pts.append(t_pts)

    pts = np.row_stack(pts)
    if len(pts) == 0:
        return None
    return east.Grid(pts, [grid_size, grid_size])


def group_by_external_poly(grid, polys, groups):
    """
    Returns groups of points according to
    the polygon they're inside of
    """

    pts_groups = [[] for _ in range(max(groups) + 1)]
    mask = np.zeros(grid.points.shape[0])

    for poly, poly_group in zip(polys, groups):  # noqa: B905

        wmin = poly[:, :2].min(axis=0)
        wmax = poly[:, :2].max(axis=0)
        gmin = grid.grid_pos(wmin)
        gmax = grid.grid_pos(wmax)

        for ptlist in grid.grid[
            gmin[0] : gmax[0] + 1, gmin[1] : gmax[1] + 1
        ].flat:
            for pt in ptlist:
                if mask[pt] != 0:
                    continue
                if not east.point_in_poly(grid.points[pt], poly):
                    continue
                mask[pt] = 1  # noqa: B909
                pts_groups[poly_group].append(pt)  # noqa: B909
    return pts_groups


def group_by_propagation(grid, mask, ids_list, alpha_shape_radius):
    """
    Performs a propagation over the points in grid (masked by mask) and
    with a radius of 2*alpha_shape_radius (alpha shape diameter).
    Returns groups of points.
    """

    visited = np.zeros(grid.points.shape[0])
    groups = []
    for pt in ids_list:
        if visited[pt] != 0:
            continue

        visited[pt] = 1  # noqa: B909

        groups.append([pt])
        tv = [pt]
        while len(tv) > 0:
            nxt = tv.pop(-1)

            gpos = grid.grid_pos(grid.points[nxt])

            nlist = []
            for ptlist in grid.grid[
                gpos[0] - 1 : gpos[0] + 2, gpos[1] - 1 : gpos[1] + 2
            ].flat:
                nlist += ptlist

            nlist = np.array(nlist, dtype=int)
            nlist = nlist[mask[nlist] > 0]
            nlist = nlist[visited[nlist] == 0]
            dists = east.distances2(
                grid.points[nlist][:, :2], grid.points[nxt][:2]
            )
            nlist = nlist[dists <= 2 * alpha_shape_radius]

            visited[nlist] = 1  # noqa: B909
            nlist_list = nlist.tolist()
            groups[-1] += nlist_list  # noqa: B909
            tv += nlist_list

    return groups


def group_points(grid, polys, groups, alpha_shape_radius):
    """
    Returns points in groups by using all
    the polygons of said group

    The returned groups are guaranteed to
    each represent strictly one alpha shape
    of alpha=alpha_shape_radius
    """

    pts_groups = group_by_external_poly(grid, polys, groups)

    # pts_group is a rough estimation but a full propagation is
    # needed to ensure a group will result in strictly one alpha shape
    # (basically the only case where this is needed is when we
    # have concentric buildings, with "islands" inside of bigger buildings)
    pts_groups_per_point = np.zeros(grid.points.shape[0], dtype=int) - 1

    i = 0
    for pts_ids in pts_groups:
        if len(pts_ids) > 0:
            pts_groups_per_point[pts_ids] = i
            i += 1

    grid.keep_only_masked(pts_groups_per_point >= 0)

    pts_groups_per_point = pts_groups_per_point[pts_groups_per_point >= 0]
    pts_groups = [[] for _ in np.unique(pts_groups_per_point)]
    for i, grp in enumerate(pts_groups_per_point):
        pts_groups[grp].append(i)

    new_pts_groups = []

    # propagate for each group
    for i, grp in enumerate(pts_groups):
        subgroups = group_by_propagation(
            grid, pts_groups_per_point == i, grp, alpha_shape_radius
        )
        new_pts_groups += subgroups

    return new_pts_groups


def get_building_contours(
    grid, pts_groups, alpha_shape_radius, groups_per_point
):
    """
    Returns the contour (including inner) of
    each building as lists of ids of points,
    with a key corresponding to the group id,
    because not all groups may lead to a
    contour (not enough points, etc)
    """

    buildings = {}
    buildingsids = {}
    for i, pts_group in enumerate(pts_groups):

        pts_mask = np.ones(grid.points.shape[0])
        pts_mask[pts_group] = 0  # mask off points that are not in this group
        id_start_pt = pts_group[np.argmin(grid.points[pts_group][:, 0])]

        outer = east.trace_contour(
            grid.points[:, :2],
            pts_mask,
            id_start_pt,
            np.array([-1, 0]),
            alpha_shape_radius * 2,  # diameter
            grid,
        )

        if outer is None:
            # do not do this: buildingsids[i] = []
            buildings[i] = []
            continue

        buildings[i] = [grid.points[outer]]
        buildingsids[i] = [outer]

        wmin = buildings[i][0].min(axis=0)
        wmax = buildings[i][0].max(axis=0)
        gmin = grid.grid_pos(wmin)
        gmax = grid.grid_pos(wmax)
        for g_ind in np.ndindex(tuple((gmax - gmin).astype(int))):

            index = gmin + g_ind

            # all points belonging to the current group
            pts_in_grp = [
                x
                for x in grid.grid[index[0], index[1]]
                if groups_per_point[x] == i
            ]
            if len(pts_in_grp) > 0:
                continue

            # check that no part of the grid is outside of the polygon
            g_wpos = grid.world_pos(np.array(index))
            if (not east.point_in_poly(g_wpos, buildings[i][0])) or np.any(
                [
                    east.point_in_poly(g_wpos, innercontour)
                    for innercontour in buildings[i][1:]
                ]
            ):
                continue
            if (
                not east.point_in_poly(
                    g_wpos + (grid.grid_factor[0], 0), buildings[i][0]
                )
            ) or np.any(
                [
                    east.point_in_poly(
                        g_wpos + (grid.grid_factor[0], 0), innercontour
                    )
                    for innercontour in buildings[i][1:]
                ]
            ):
                continue
            if (
                not east.point_in_poly(
                    g_wpos + (0, grid.grid_factor[1]), buildings[i][0]
                )
            ) or np.any(
                [
                    east.point_in_poly(
                        g_wpos + (0, grid.grid_factor[1]), innercontour
                    )
                    for innercontour in buildings[i][1:]
                ]
            ):
                continue
            if (
                not east.point_in_poly(
                    g_wpos + grid.grid_factor, buildings[i][0]
                )
            ) or np.any(
                [
                    east.point_in_poly(g_wpos + grid.grid_factor, innercontour)
                    for innercontour in buildings[i][1:]
                ]
            ):
                continue

            # ndindex is "for x: for y:"
            # -> there will always be a cell with points to
            # the left (and also up)
            leftids = [
                x
                for x in grid.grid[index[0] - 1, index[1]]
                if groups_per_point[x] == i
            ]
            if len(leftids) > 0:
                circle_center = g_wpos + alpha_shape_radius

                lpts = grid.points[leftids]
                dx = -lpts[:, 0] + circle_center[0]
                dy = -lpts[:, 1] + circle_center[1]

                sintheta = dy / alpha_shape_radius
                costheta = np.sqrt(1 - sintheta * sintheta)

                circle_dist = dx - costheta * alpha_shape_radius

                ptid_chosen = np.argmin(circle_dist)
                pt_chosen = leftids[ptid_chosen]

                inner = east.trace_contour(
                    grid.points[:, :2],
                    pts_mask,
                    pt_chosen,
                    np.array([costheta[ptid_chosen], sintheta[ptid_chosen]]),
                    alpha_shape_radius * 2,  # diameter
                    grid,
                )

                innerpts = grid.points[inner][:, :2]

                buildings[i].append(innerpts)
                buildingsids[i].append(inner)

    return buildingsids


def get_segments(contour, closed=True):
    """
    Returns the segments (tuples of ids representing a segment)
    representing the polygon, whether it is closed or not
    """

    if closed:
        return [(i, i + 1) for i in range(len(contour) - 2)] + [
            (len(contour) - 2, 0)
        ]
    return [(i, i + 1) for i in range(len(contour) - 1)] + [
        (len(contour) - 1, 0)
    ]


def get_point_in_contour(contour):
    """
    Returns the coordinates of a point that is
    inside the (counter-clockwise) contour
    """

    for i in range(len(contour) - 2):

        v = contour[i] - contour[i + 1]
        w = contour[i + 2] - contour[i + 1]
        if np.linalg.norm(v + w) == 0:
            continue
        direction = (v + w) / np.linalg.norm(v + w)

        # [-PI;PI]
        theta = np.arctan2(w[1] * v[0] - w[0] * v[1], w[0] * v[0] + w[1] * v[1])
        if theta < -0.01:  # negative is a notch
            return contour[i + 1] + direction * 0.1
        if theta > 0.01:  # positive is an ear
            return contour[i + 1] - direction * 0.1

    return contour[0] - contour[0]  # (0, 0)


def delaunay_triangulate(buildingids, grid):
    """
    Returns the delaunay triangulation of a
    building's contours, holes included, by
    using the triangle library

    If for any reason no triangulation could
    be found, return None
    """

    pts = []
    sgs = []
    hls = []

    nb_pts = 0

    for i, contour in enumerate(buildingids):

        # contours are open by default
        pts.append(grid.points[contour][:, :2])

        # offset ids by the number of points
        sgs.append(
            np.array(get_segments(contour, closed=False), dtype=int) + nb_pts
        )

        if i > 0:
            hls.append(get_point_in_contour(pts[-1]))
        nb_pts += len(pts[-1])  # we removed a point :)

    pts = np.vstack(pts)
    if len(pts) <= 2:
        return None

    sgs = np.vstack(sgs)

    # When using Delaunay on planar as is the case here,
    # inserting twice the same point leads to a crash.
    # Fix : Merge points (will need to update segment indices)
    new_indices = np.arange(len(pts))
    tbr = np.zeros(len(pts))
    for i, pt in enumerate(pts):
        same_pts_indices = np.where(np.all(pts == pt, axis=1))[0]
        for j in same_pts_indices:
            if i != j:
                new_indices[max(i, j)] = min(i, j)  # noqa: B909
                tbr[max(i, j)] = 1  # noqa: B909

    tbr = np.argwhere(tbr == 1)
    for index in tbr:
        pts = np.delete(pts, index, axis=0)
        new_indices[new_indices > index] -= 1
        tbr[tbr > index] -= 1

    for seg in sgs:
        seg[0] = new_indices[seg[0]]  # noqa: B909
        seg[1] = new_indices[seg[1]]  # noqa: B909

    delaunay_input = {"vertices": pts, "segments": sgs}

    if len(hls) > 0:  # there may not be any hole in the building
        delaunay_input["holes"] = np.array(hls)

    delaunay_output = trlib.triangulate(delaunay_input, "p")

    dks = delaunay_output.keys()

    # this may happen if all points are colinear
    if "segments" not in dks or "triangles" not in dks:
        return None
    if len(delaunay_output["triangles"]) <= 0:
        return None

    if len(sgs) == len(delaunay_output["segments"]):
        delaunay_output["segments"] = sgs

    return delaunay_output


def dist_to_line(p, ll, lr):
    """
    Returns the distance of point p to the line ll<->lr
    """

    top = (lr[0] - ll[0]) * (ll[1] - p[1]) - (ll[0] - p[0]) * (lr[1] - ll[1])
    bot = np.sqrt(
        (lr[0] - ll[0]) * (lr[0] - ll[0]) + (lr[1] - ll[1]) * (lr[1] - ll[1])
    )
    return abs(top) / max(bot, 0.00001)  # division by 0 safeguard


def is_valid_line(left, right, ctr, threshold):
    """
    Returns whether the line ctr[left]->ctr[right] is
    representative of the shape created by points in ctr[left:right+1]
    according to the parameter threshold
    """

    pt_l = ctr[left]
    pt_r = ctr[right]

    for i in range(left + 1, right - 1, 1):
        if np.all(pt_l == pt_r):  #
            continue
        if dist_to_line(ctr[i], pt_l, pt_r) > threshold:
            return False

    return True


def line_error(left, right, ctr):
    """
    Returns the total error generated by a
    line from left to right (sum of distances
    to the line).
    """

    pt_l = ctr[left]
    pt_r = ctr[right]
    err = 0
    nb = 0
    for i in range(left + 1, right - 1, 1):
        if np.all(pt_l == pt_r):  #
            continue
        nb += 1
        err += dist_to_line(ctr[i], pt_l, pt_r)

    return err / max(nb, 1)


def simplify_douglaspeucker(ctrpts, ctr, threshold, search_radius):
    """
    Returns the contour point ids representing
    a simplified version of the contour, as per
    douglas-peucker's definition
    """

    simplified_ids = []

    id_left = 0
    id_right = len(ctr) - 1

    simplified_ids.append(id_left)

    while True:
        while not is_valid_line(id_left, id_right, ctrpts, threshold):
            id_right = np.floor(id_left + (id_right - id_left) / 1.3).astype(
                int
            )

        # search for least distance line in a small neighborhood
        errors = []
        ids_nh = list(
            range(
                max(id_left + 1, id_right - search_radius),
                min(len(ctr) - 1, id_right + search_radius + 1),
            )
        )

        if len(ids_nh) > 0:
            for id_nh in ids_nh:
                errors.append(line_error(id_left, id_nh, ctrpts))
            id_right = ids_nh[np.argmin(errors)]

        simplified_ids.append(id_right)

        if id_right == len(ctr) - 1:
            break

        id_left = id_right
        id_right = len(ctr) - 1

    if len(simplified_ids) > 2:
        return np.array(ctr, dtype=int)[simplified_ids]

    # if the simplified contour is too small, just keep the original
    return np.array(ctr, dtype=int)


def area_errors(points):
    """
    Returns the area of the triangle created
    by each point's presence
    """

    errs = []
    for i, pi in enumerate(points):
        pim = points[(i - 1) % len(points)]
        pip = points[(i + 1) % len(points)]
        errs.append(
            0.5
            * abs(
                pim[0] * pi[1]
                + pi[0] * pip[1]
                + pip[0] * pim[1]
                - pim[0] * pip[1]
                - pi[0] * pim[1]
                - pip[0] * pi[1]
            )
        )

    return np.array(errs)


def simplify_visvalingam(ctrpts, ctr, threshold):
    """
    Returns the contour point ids representing
    a simplified version of the contour, as per
    visvalingam's definition
    """

    points = ctrpts.copy()
    ids = ctr.copy()
    errs = area_errors(points)

    while np.min(errs) < threshold and len(points) > 4:
        id_worst = np.argmin(errs)
        points = np.delete(points, id_worst, axis=0)
        ids = np.delete(ids, id_worst, axis=0)
        errs = area_errors(points)

    if len(ids) > 2:
        return ids
    # if the simplified contour is too small, just keep the original
    return ctr


def simplify(
    contour, grid, simplification_method, doug_trsh, doug_win, visval_area
):
    """
    Simplifie le contour, en utilisant une des méthodes supportées
    """

    # enlever le point qui ferme
    contour = contour[:-1]
    if simplification_method == "none":
        return contour

    if simplification_method == "douglas":
        ctrpts = grid.points[contour][:, :2]
        contour = simplify_douglaspeucker(ctrpts, contour, doug_trsh, doug_win)
        return contour

    if simplification_method == "visvalingam":
        ctrpts = grid.points[contour][:, :2]
        contour = simplify_visvalingam(ctrpts, contour, visval_area)
        return contour

    if simplification_method == "douglas then visvalingam":
        ctrpts = grid.points[contour][:, :2]
        contour = simplify_douglaspeucker(ctrpts, contour, doug_trsh, doug_win)
        ctrpts = grid.points[contour]
        contour = simplify_visvalingam(ctrpts, contour, visval_area)
        return contour

    # should never happen, but just in case do the same as "none"
    return contour


def project_on_dtm(points, dtm_mesh):
    """
    Projets the 2D points onto the dtm_mesh's triangles vertically (direction z)
    Returns the points with a third axis z
    """

    # points is a (n, 2) ndarray of floats

    # triangle_ids is a (m, 3) ndarray of ints
    triangle_ids = dtm_mesh["triangles"]
    # vertices is a (n, 3) ndarray of floats
    vertices = dtm_mesh["vertices"]

    min_height_achievable = np.min(dtm_mesh["vertices"][:, 2])

    nb_pts = len(points)

    # project every point vertically onto the mesh
    t_r = np.zeros(nb_pts) + min_height_achievable - 0.1
    mask = np.ones(nb_pts, dtype=bool)

    for tri in triangle_ids:

        # array of ids, used to access the overarching points
        # so as to not compute things for every vertex everytime
        over_ids = np.arange(nb_pts).astype(int)
        over_ids = over_ids[mask]

        p1 = vertices[tri[0]][:3]
        p2 = vertices[tri[1]][:3]
        p3 = vertices[tri[2]][:3]

        # vectors
        c = p3[:2] - p1[:2]
        b = p2[:2] - p1[:2]
        p = points[mask] - p1[:2]

        # dot products
        cc = np.dot(c, c)
        bc = np.dot(b, c)
        pc = np.dot(p, c)
        bb = np.dot(b, b)
        pb = np.dot(p, b)

        # barycentric coordinates
        denom = cc * bb - bc * bc
        u = (bb * pc - bc * pb) / denom
        v = (cc * pb - bc * pc) / denom

        ids_in = np.argwhere(
            np.logical_and(np.logical_and(u + v <= 1, u >= 0), v >= 0)
        )

        if len(ids_in) == 0:  # no points corresponding to this triangle
            continue

        ids_in = ids_in.reshape(-1)

        heights = (
            (p3[2] - p1[2]) * u[ids_in] + (p2[2] - p1[2]) * v[ids_in] + p1[2]
        )

        t_r[over_ids[ids_in]] = heights  # noqa: B909
        mask[over_ids[ids_in]] = False  # noqa: B909

        if mask.sum() == 0:  # all points were assigned a height! :D
            break

    # at least one point has a height and at least one doesn't
    if mask.sum() > 0 and mask.sum() < nb_pts - 1:

        over_ids = np.arange(nb_pts).astype(int)
        over_ids_with_height = over_ids[np.logical_not(mask)]
        over_ids_without_height = over_ids[mask]

        # heights not found -> min height of the building
        t_r[over_ids_without_height] = np.min(t_r[over_ids_with_height], axis=0)

    return np.hstack((points, t_r[:, np.newaxis]))


def get_hist_data(arr):
    """
    returns an histogram of the array's data with x and y components
    """

    hd = np.histogram(arr, bins="auto")
    data_points = np.array(
        [
            [a, b]
            for a, b in zip(
                # ( x_i + x_{i+1} ) / 2
                [
                    (x + xp) / 2
                    for x, xp in zip(hd[1][:-1], hd[1][1:], strict=True)
                ],
                hd[0],
                strict=True,
            )
        ]
    )
    return data_points


def get_peaks(
    data_points, pad_zeros, smooth_curve, min_dist_meters, prominence_percent
):
    """
    Get peaks found in the data points given
    """

    tot_h = data_points[:, 0].max() - data_points[:, 0].min()
    onetick = tot_h / (len(data_points) - 1)

    if pad_zeros:
        data_points = np.concatenate(
            (
                [[data_points[0, 0] - onetick, 0]],
                data_points,
                [[data_points[-1, 0] + onetick, 0]],
            ),
            axis=0,
        )

    sc_mult_neigh = smooth_curve / 2
    sc_mult_self = 1 - smooth_curve
    for i in range(1, len(data_points) - 1):
        data_points[i, 1] = (
            data_points[i - 1, 1] + data_points[i + 1, 1]
        ) * sc_mult_neigh + data_points[i, 1] * sc_mult_self

    prom = data_points[:, 1].max() * prominence_percent

    results = find_peaks(
        data_points[:, 1],
        distance=max(min_dist_meters / onetick, 1),
        prominence=prom,
    )

    return data_points[results[0]]


def filter_clusters(
    pts, full_to_compressed, grid, clusters, rad_neighbors, dims=3
):
    """
    attempts to remove noise and to overall improve
    the quality of the clusters given
    """

    if len(np.unique(clusters)) < 2:
        return clusters

    new_clusters = np.zeros(clusters.shape, dtype=int)

    for i, pt in enumerate(pts):

        neighs = []
        for neigh_l in grid.get_pts_by_cell_around(pt, rad_neighbors):
            neighs += neigh_l
        neighs = np.array(neighs, dtype=int)
        neighs = full_to_compressed[neighs[full_to_compressed[neighs] >= 0]]
        dists = np.sum(np.square(pts[neighs][:, :dims] - pt[:dims]), axis=1)
        neighs = neighs[dists <= rad_neighbors * rad_neighbors]
        votes = np.bincount(clusters[neighs])

        if votes.sum() > 0:
            new_clusters[i] = np.argmax(votes)
        else:
            # point is alone in his radius :(
            new_clusters[i] = clusters[i]

    return new_clusters


def peaks_clustering(
    pts,
    pad_zeros=True,
    smooth_curve=0.33,
    min_dist_meters=1.5,
    prominence_percent=0.2,
):
    """
    creates groups of points based on the density of heights
    """

    dps = get_hist_data(pts[:, 2])

    peaks = get_peaks(
        dps, pad_zeros, smooth_curve, min_dist_meters, prominence_percent
    )

    clusters = np.zeros(pts.shape[0], dtype=int)
    for i, pt in enumerate(pts):
        dm, jm = 1e10, -1
        for j, peak in enumerate(peaks):
            d = abs(pt[2] - peak[0])
            if d < dm:
                dm = d
                jm = j
        clusters[i] = jm

    return clusters


def reassign_clusters_with_area(
    grid,
    clusters,
    full_to_compressed,
    compressed_to_full,
    min_area,
    search_radius,
):
    """
    Takes clusters as inputs, with possibly
    an area that's too small to be kept.

    Sort clusters by points

    Go through them small to large
        Compute footprint as the area of the convex hull
        If they have too small of a footprint:
        - Count the points of clusters close
        - to each point of this one
        - Reassign the cluster to the other cluster
        - that's the most present in close proximity
    """

    clusters_nb_pts = np.bincount(clusters)
    clusters_ids = np.argsort(clusters_nb_pts)
    clusters_nb_pts = clusters_nb_pts[clusters_ids]

    for cid in clusters_ids[:-1]:

        points = grid.points[compressed_to_full[clusters == cid]][:, :2]
        if len(points) < 1:
            continue

        # 3 pts needed for the convex hull to be computed
        # area is 0 when 2 or less points anyway
        if len(points) >= 3:
            # footprint
            hull = ConvexHull(points)
            area = hull.volume

            if area >= min_area:
                continue

        # vote for a cluster
        votes = np.zeros(len(clusters_ids), dtype=int)

        # votes come from cluster points
        can_vote = np.ones(len(clusters), dtype=bool)
        can_vote[clusters == cid] = False

        for pt in points:

            neighs_iter = grid.get_pts_by_cell_around(pt, search_radius)
            neighs = []
            for nl in neighs_iter:
                neighs += nl
            neighs = np.array(neighs, dtype=int)

            nids = full_to_compressed[neighs]
            nids = nids[nids >= 0]
            nids = nids[can_vote[nids]]

            npts = grid.points[compressed_to_full[nids]]
            ndists = east.distances2(npts[:, :2], pt[:2])
            nids = nids[ndists <= search_radius * search_radius]

            can_vote[nids] = False

            ptvotes = np.bincount(clusters[nids])
            ptvotes = np.pad(ptvotes, (0, len(clusters_ids) - len(ptvotes)))

            votes += ptvotes

        new_cluster = np.argmax(votes)

        clusters[clusters == cid] = new_cluster

    return clusters


def refine_buildings_by_height(
    buildings,
    grid,
    alpha_shape_radius,
    height_detection_min_building_points=20,
    subgroups_filtering_radius=2.5,
    smooth_curve=0.33,
    min_dist_meters=1.5,
    prominence_percent=0.2,
    min_cluster_area=50,
    cluster_reassign_search_radius=2.5,
):
    """
    attempts to create a classification of zones with
    different heights in the global building shape
    """

    # buildings -> for each grid point : its corresponding building

    # buildings -> all possible buildings + none (-1)
    b_ids = np.unique(buildings)

    new_groups = []
    new_groups_per_point = np.zeros(grid.points.shape[0]) - 1
    id_next_group = 0

    for b_id in b_ids:

        if b_id < 0:
            # remove -1
            continue

        # all points in building of id b_id
        building = grid.points[buildings == b_id][:, :3]

        # small buildings won't have clear zones anyways
        if len(building) <= height_detection_min_building_points:
            continue

        # for all grid points -> the corresponding id in building (-1 if not in)
        full_to_compressed = np.zeros(grid.points.shape[0], dtype=int) - 1
        full_to_compressed[buildings == b_id] = np.arange(len(building))

        # for all the building's points : the corresponding ids in grid points
        compressed_to_full = np.argwhere(buildings == b_id).reshape(-1)

        # clusters -> for each building point : its corresponding cluster id 0-n
        clusters = peaks_clustering(
            building, smooth_curve, min_dist_meters, prominence_percent
        )
        clusters = filter_clusters(
            building,
            full_to_compressed,
            grid,
            clusters,
            subgroups_filtering_radius,
            dims=2,
        )
        clusters = filter_clusters(
            building,
            full_to_compressed,
            grid,
            clusters,
            subgroups_filtering_radius,
            dims=2,
        )
        clusters = filter_clusters(
            building,
            full_to_compressed,
            grid,
            clusters,
            subgroups_filtering_radius,
            dims=3,
        )

        points_groups = []
        if len(np.unique(clusters)) > 1:
            # A split in 2 or more groups happened, so we
            # must check coherence for the subsequent alphashape retrieval step
            # propagate in all clusters to create "sub buildings"

            new_clusters = np.zeros(len(clusters), dtype=int) - 1
            number_additions = np.zeros(len(clusters), dtype=int)
            subg_offset = 0
            for i in np.unique(clusters):

                # compressed_to_full but only for the cluster's points
                ptsids_full = compressed_to_full[clusters == i]
                # for a grid point: is the point in the cluster out-0 or in-1
                mask = np.zeros(grid.points.shape[0], dtype=int)
                mask[ptsids_full] = 1

                # list of lists of points
                subgroups = group_by_propagation(
                    grid, mask, ptsids_full, alpha_shape_radius
                )

                for i, subgroup in enumerate(subgroups):
                    new_clusters[full_to_compressed[subgroup]] = subg_offset + i
                    number_additions[full_to_compressed[subgroup]] += 1
                subg_offset += len(subgroups)

            new_clusters = reassign_clusters_with_area(
                grid,
                new_clusters,
                full_to_compressed,
                compressed_to_full,
                min_cluster_area,
                cluster_reassign_search_radius,
            )

            for i in np.unique(new_clusters):
                points_groups.append(
                    compressed_to_full[(new_clusters == i).nonzero()]
                )

        else:
            points_groups.append(compressed_to_full.tolist())

        for height_group in points_groups:
            new_groups.append(height_group)
            new_groups_per_point[height_group] = id_next_group
            id_next_group += 1

    return new_groups, new_groups_per_point
