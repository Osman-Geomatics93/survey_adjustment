"""survey_adjustment.core.solver.geometry

Geometry helpers for 2D least-squares adjustment.

Conventions:
  - Coordinates: Easting = X, Northing = Y
  - Azimuth: North = 0, clockwise positive
  - Angles: radians internally

Implementation detail:
  - Azimuth is computed using ``atan2(dE, dN)``.
"""

from __future__ import annotations

import math
from typing import Tuple


TAU = 2.0 * math.pi


def wrap_pi(angle: float) -> float:
    """Normalize angle to (-π, π]."""
    # Python's modulo for floats works well here.
    a = (angle + math.pi) % TAU - math.pi
    # Force +π instead of -π for deterministic behavior.
    if a <= -math.pi:
        a += TAU
    return a


def wrap_2pi(angle: float) -> float:
    """Normalize angle to [0, 2π)."""
    a = angle % TAU
    if a < 0:
        a += TAU
    return a


def azimuth(e1: float, n1: float, e2: float, n2: float) -> float:
    """Compute azimuth from (e1,n1) to (e2,n2) in radians.

    Returns an angle in [0, 2π), with North=0 and clockwise positive.
    """
    return wrap_2pi(math.atan2(e2 - e1, n2 - n1))


def distance_2d(e1: float, n1: float, e2: float, n2: float) -> float:
    """Compute 2D horizontal distance."""
    de = e2 - e1
    dn = n2 - n1
    return math.hypot(de, dn)


def distance_partials(e1: float, n1: float, e2: float, n2: float) -> Tuple[float, float, float, float]:
    """Partials of distance d w.r.t (E1, N1, E2, N2)."""
    de = e2 - e1
    dn = n2 - n1
    d = math.hypot(de, dn)
    if d == 0.0:
        raise ValueError("Cannot compute distance partials for coincident points")
    return (-de / d, -dn / d, de / d, dn / d)


def azimuth_partials(e1: float, n1: float, e2: float, n2: float) -> Tuple[float, float, float, float]:
    """Partials of azimuth α w.r.t (E1, N1, E2, N2).

    α = atan2(dE, dN)  (east as y, north as x)
    """
    de = e2 - e1
    dn = n2 - n1
    r2 = de * de + dn * dn
    if r2 == 0.0:
        raise ValueError("Cannot compute azimuth partials for coincident points")

    # Using derivative of atan2(y, x): dθ/dx = -y/(x^2+y^2), dθ/dy = x/(x^2+y^2)
    # Here y=de, x=dn.
    da_dde = dn / r2
    da_ddn = -de / r2

    # Map to coordinates (E1,N1,E2,N2)
    dE1 = -da_dde
    dN1 = -da_ddn
    dE2 = da_dde
    dN2 = da_ddn
    return (dE1, dN1, dE2, dN2)


def angle_at_point(
    e_at: float,
    n_at: float,
    e_from: float,
    n_from: float,
    e_to: float,
    n_to: float,
) -> float:
    """Angle at (e_at,n_at) from "from" ray to "to" ray, clockwise positive.

    Result is in [0, 2π).
    """
    az_from = azimuth(e_at, n_at, e_from, n_from)
    az_to = azimuth(e_at, n_at, e_to, n_to)
    return wrap_2pi(az_to - az_from)


def angle_partials(
    e_at: float,
    n_at: float,
    e_from: float,
    n_from: float,
    e_to: float,
    n_to: float,
) -> Tuple[float, float, float, float, float, float]:
    """Partials of angle w.r.t (E_from, N_from, E_at, N_at, E_to, N_to)."""

    # az_to depends on (at, to)
    dE_at_to, dN_at_to, dE_to, dN_to = azimuth_partials(e_at, n_at, e_to, n_to)
    # az_from depends on (at, from)
    dE_at_from, dN_at_from, dE_from, dN_from = azimuth_partials(e_at, n_at, e_from, n_from)

    # angle = az_to - az_from
    dE_at = dE_at_to - dE_at_from
    dN_at = dN_at_to - dN_at_from

    return (
        -dE_from,  # because angle = az_to - az_from
        -dN_from,
        dE_at,
        dN_at,
        dE_to,
        dN_to,
    )
