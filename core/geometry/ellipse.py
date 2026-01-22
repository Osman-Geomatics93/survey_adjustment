"""Ellipse polygon generation (QGIS-free).

This module provides functions to generate polygon vertices for error ellipses,
suitable for visualization in GIS applications.
"""

from __future__ import annotations

import math
from typing import List, Tuple


def ellipse_polygon_points(
    center_e: float,
    center_n: float,
    semi_major: float,
    semi_minor: float,
    orientation: float,
    num_vertices: int = 64,
) -> List[Tuple[float, float]]:
    """
    Generate polygon vertices approximating an ellipse.

    The ellipse is defined by its center, semi-axes, and orientation.
    The orientation follows surveying convention: measured from north (Y-axis),
    clockwise positive.

    Args:
        center_e: Easting coordinate of center
        center_n: Northing coordinate of center
        semi_major: Semi-major axis length (meters)
        semi_minor: Semi-minor axis length (meters)
        orientation: Orientation of semi-major axis in radians
                    (from north, clockwise positive)
        num_vertices: Number of vertices to generate (36-72 recommended)

    Returns:
        List of (easting, northing) tuples forming a closed polygon.
        First and last points are identical to close the ring.
    """
    if num_vertices < 4:
        num_vertices = 4

    # Convert surveying orientation (from north, clockwise) to math angle
    # Math angle is from east (X), counter-clockwise
    # surveying: theta from north (Y), clockwise
    # math: alpha from east (X), counter-clockwise
    # alpha = 90 - theta (in degrees), or pi/2 - theta (in radians)
    math_angle = math.pi / 2 - orientation

    cos_theta = math.cos(math_angle)
    sin_theta = math.sin(math_angle)

    points = []
    for i in range(num_vertices):
        # Parametric angle around ellipse
        t = 2 * math.pi * i / num_vertices

        # Point on unrotated ellipse centered at origin
        x = semi_major * math.cos(t)
        y = semi_minor * math.sin(t)

        # Rotate by orientation
        x_rot = x * cos_theta - y * sin_theta
        y_rot = x * sin_theta + y * cos_theta

        # Translate to center
        e = center_e + x_rot
        n = center_n + y_rot

        points.append((e, n))

    # Close the polygon
    points.append(points[0])

    return points
