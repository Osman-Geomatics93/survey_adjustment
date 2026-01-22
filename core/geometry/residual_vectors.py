"""Residual vector geometry generation (QGIS-free).

This module provides functions to generate line geometries representing
observation residuals for visualization in GIS applications.
"""

from __future__ import annotations

import math
from typing import Tuple, Optional


def distance_residual_vector(
    from_e: float,
    from_n: float,
    to_e: float,
    to_n: float,
    residual: float,
    scale: float = 1000.0,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Generate a line segment representing a distance residual.

    The vector is placed along the observation line (from -> to),
    starting at the midpoint. A positive residual (observed > computed)
    means the line should appear longer, so we draw the vector pointing
    outward from the midpoint.

    Args:
        from_e: Easting of from-point
        from_n: Northing of from-point
        to_e: Easting of to-point
        to_n: Northing of to-point
        residual: Distance residual in meters (observed - computed)
        scale: Scale factor for visualization (residual * scale = vector length)

    Returns:
        Tuple of ((start_e, start_n), (end_e, end_n)) defining the line
    """
    # Midpoint of the observation line
    mid_e = (from_e + to_e) / 2
    mid_n = (from_n + to_n) / 2

    # Direction from from-point to to-point
    dx = to_e - from_e
    dy = to_n - from_n
    length = math.sqrt(dx * dx + dy * dy)

    if length < 1e-10:
        # Degenerate case: points are coincident
        return ((mid_e, mid_n), (mid_e, mid_n))

    # Unit vector along observation direction
    ux = dx / length
    uy = dy / length

    # Vector length based on residual and scale
    vec_len = residual * scale

    # Start at midpoint, end at midpoint + scaled residual
    start = (mid_e, mid_n)
    end = (mid_e + ux * vec_len, mid_n + uy * vec_len)

    return (start, end)


def direction_residual_vector(
    station_e: float,
    station_n: float,
    target_e: float,
    target_n: float,
    residual: float,
    scale: float = 1000.0,
    arrow_length: Optional[float] = None,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Generate a line segment representing a direction/bearing residual.

    The vector is placed at the station point, perpendicular to the
    direction to the target. The vector shows the angular offset
    (positive = clockwise residual).

    Args:
        station_e: Easting of station (observation point)
        station_n: Northing of station
        target_e: Easting of target point
        target_n: Northing of target point
        residual: Direction residual in radians (observed - computed)
        scale: Scale factor converting radians to meters
               (e.g., 1000 means 1 radian = 1000m vector length)
        arrow_length: Optional fixed length for the base arrow.
                     If None, uses 10% of distance to target or 10m minimum.

    Returns:
        Tuple of ((start_e, start_n), (end_e, end_n)) defining the line
    """
    # Direction from station to target
    dx = target_e - station_e
    dy = target_n - station_n
    dist = math.sqrt(dx * dx + dy * dy)

    if dist < 1e-10:
        # Degenerate case
        return ((station_e, station_n), (station_e, station_n))

    # Determine arrow base length
    if arrow_length is None:
        arrow_length = max(dist * 0.1, 10.0)

    # Unit vector toward target
    ux = dx / dist
    uy = dy / dist

    # Base point: along the direction line at arrow_length distance
    base_e = station_e + ux * arrow_length
    base_n = station_n + uy * arrow_length

    # Perpendicular vector (90 degrees clockwise from direction)
    # Clockwise rotation: (ux, uy) -> (uy, -ux)
    perp_x = uy
    perp_y = -ux

    # Vector length based on residual and scale
    # Positive residual (observed > computed) means clockwise offset
    vec_len = residual * scale

    # End point: base + perpendicular offset
    end_e = base_e + perp_x * vec_len
    end_n = base_n + perp_y * vec_len

    return ((base_e, base_n), (end_e, end_n))


def angle_residual_vector(
    at_e: float,
    at_n: float,
    from_e: float,
    from_n: float,
    to_e: float,
    to_n: float,
    residual: float,
    scale: float = 1000.0,
    arrow_length: Optional[float] = None,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Generate a line segment representing an angle residual.

    An angle is measured at a station (at_point) from a backsight (from_point)
    to a foresight (to_point). The residual vector shows the angular offset
    of the foresight direction.

    Args:
        at_e: Easting of angle vertex (station)
        at_n: Northing of angle vertex
        from_e: Easting of backsight point
        from_n: Northing of backsight point
        to_e: Easting of foresight point
        to_n: Northing of foresight point
        residual: Angle residual in radians (observed - computed)
        scale: Scale factor converting radians to meters
        arrow_length: Optional fixed length for the base arrow.
                     If None, uses 10% of distance to foresight or 10m minimum.

    Returns:
        Tuple of ((start_e, start_n), (end_e, end_n)) defining the line
    """
    # For angle residuals, we show the effect on the foresight direction
    # This is similar to direction residual, placed along the foresight ray
    return direction_residual_vector(
        station_e=at_e,
        station_n=at_n,
        target_e=to_e,
        target_n=to_n,
        residual=residual,
        scale=scale,
        arrow_length=arrow_length,
    )
