import os

import numpy as np
import pandas as pd
from scipy.spatial import Voronoi
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


# based on https://stackoverflow.com/a/20678647/2443944
def voronoi_finite_polygons_2d(vor, radius=None):
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")
    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all([v >= 0 for v in vertices]):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def save_2d_area(dense_name='hubert', vocab_size=100):
    """
    Saves the Voronoi diagram for the given dense model and vocabulary size.

    Parameters:
    - dense_name (str): name of the dense model.
    - vocab_size (int): size of the vocabulary.
    """
    # read the 2D coordinates of the centers from a CSV file
    input_file = f"assets/dimension_reduction/{dense_name}_T-SNE_2D.csv"
    assert os.path.exists(input_file)
    centers_2d = pd.read_csv(input_file)

    # select the centers with the given vocabulary size
    centers_2d = centers_2d[centers_2d['src_vocab'].astype(int) == int(vocab_size)]

    # compute the Voronoi diagram
    vor = Voronoi(centers_2d[['X', 'Y']].values)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    # create a bounding polygon
    bounds = [
        [vor.min_bound[0], vor.min_bound[1]],
        [vor.min_bound[0], vor.max_bound[1]],
        [vor.max_bound[0], vor.max_bound[1]],
        [vor.max_bound[0], vor.min_bound[1]],
    ]
    ploy_all = Polygon(bounds)

    # create an empty data frame to store the Voronoi regions for each unit
    res = pd.DataFrame()

    # iterate over the Voronoi regions
    for r in regions:
        # retrieve the vertices of the region
        v = vertices[r]

        # create a polygon object for the region
        polygon_onj = Polygon(v)

        # clip the polygon to the bounding polygon
        polygon_clip = polygon_onj.intersection(ploy_all)

        # retrieve the x and y coordinates of the vertices
        x, y = polygon_clip.exterior.coords.xy

        # find the unit corresponding to the region
        unit = np.argmax([polygon_onj.contains(Point(x)) for x in centers_2d[['X', 'Y']].values])

        # store the x and y coordinates of the vertices in the data frame
        res.loc[unit, 'X'] = ",".join([str(x_) for x_ in x])
        res.loc[unit, 'Y'] = ",".join([str(y_) for y_ in y])

    # sort the data frame by the unit index
    res = res.sort_index()

    # save the results
    os.makedirs("assets/2d_area", exist_ok=True)
    res.to_csv(f"assets/2d_area/{dense_name}_{vocab_size}.csv")
