"""QGIS-free observation parsing utilities.

These helpers are used by QGIS Processing algorithms but are intentionally free
of any QGIS imports so they can be unit-tested in a plain Python environment.

Supported formats:
- points CSV
- distances CSV
- directions CSV
- angles CSV
- traverse (record-based) file with lines starting with POINT/DIST/DIR/ANGLE

Angle conventions:
- Input angles/directions may be in degrees; values are stored internally in radians.
- Sigma for angular obs may be in radians or arcseconds.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ...core.models.point import Point
from ...core.models.observation import (
    DistanceObservation,
    DirectionObservation,
    AngleObservation,
    HeightDifferenceObservation,
    GnssBaselineObservation,
)
from ...core.models.network import Network


def _parse_bool(value: str | bool | int | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    return default


def _get(row: Dict[str, str], keys: Sequence[str], default: str = "") -> str:
    for k in keys:
        if k in row and row[k] != "":
            return row[k]
    return default


def _angle_to_rad(value: float, unit: str) -> float:
    unit_l = unit.strip().lower()
    if unit_l in {"rad", "radian", "radians"}:
        return value
    if unit_l in {"deg", "degree", "degrees"}:
        return math.radians(value)
    raise ValueError(f"Unknown angle unit: {unit}")


def _sigma_to_rad(value: float, unit: str) -> float:
    unit_l = unit.strip().lower()
    if unit_l in {"rad", "radian", "radians"}:
        return value
    if unit_l in {"arcsec", "arcsecond", "arcseconds"}:
        return value * math.pi / (180.0 * 3600.0)
    if unit_l in {"deg", "degree", "degrees"}:
        return math.radians(value)
    raise ValueError(f"Unknown sigma unit: {unit}")


def parse_points_csv(path: str | Path) -> Dict[str, Point]:
    """Parse points from a CSV.

    Expected columns (flexible):
      - id/point_id/name
      - easting/x
      - northing/y
      - fixed_e, fixed_easting
      - fixed_n, fixed_northing
      - sigma_e, sigma_easting
      - sigma_n, sigma_northing
    """

    path = Path(path)
    points: Dict[str, Point] = {}

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = _get(row, ["point_id", "id", "pid"]).strip()
            if not pid:
                raise ValueError(f"Missing point_id in {path}")

            name = _get(row, ["name", "point_name"], default=pid).strip()
            e = float(_get(row, ["easting", "x", "E", "east"], default="0"))
            n = float(_get(row, ["northing", "y", "N", "north"], default="0"))

            fixed_e = _parse_bool(_get(row, ["fixed_e", "fixed_easting", "fixed_x"], default=""), default=False)
            fixed_n = _parse_bool(_get(row, ["fixed_n", "fixed_northing", "fixed_y"], default=""), default=False)

            sigma_e = float(_get(row, ["sigma_e", "sigma_easting", "se"], default="0.0"))
            sigma_n = float(_get(row, ["sigma_n", "sigma_northing", "sn"], default="0.0"))

            points[pid] = Point(
                id=pid,
                name=name,
                easting=e,
                northing=n,
                fixed_easting=fixed_e,
                fixed_northing=fixed_n,
                sigma_easting=sigma_e,
                sigma_northing=sigma_n,
            )

    return points


def parse_distances_csv(path: str | Path, sigma_default: float = 0.003) -> List[DistanceObservation]:
    """Parse distance observations from CSV."""

    path = Path(path)
    out: List[DistanceObservation] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            obs_id = _get(row, ["obs_id", "id"], default="").strip()
            from_p = _get(row, ["from_point", "from", "from_point_id"], default="").strip()
            to_p = _get(row, ["to_point", "to", "to_point_id"], default="").strip()
            dist = float(_get(row, ["distance", "value"], default="0"))
            sigma = float(_get(row, ["sigma_distance", "sigma"], default=str(sigma_default)))

            out.append(
                DistanceObservation(
                    id=obs_id,
                    obs_type=None,  # overwritten in __post_init__
                    value=dist,
                    sigma=sigma,
                    from_point_id=from_p,
                    to_point_id=to_p,
                )
            )
    return out


def parse_directions_csv(
    path: str | Path,
    direction_unit: str = "degrees",
    sigma_unit: str = "radians",
    sigma_default: float = 0.00015,
) -> List[DirectionObservation]:
    """Parse direction observations from CSV."""

    path = Path(path)
    out: List[DirectionObservation] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            obs_id = _get(row, ["obs_id", "id"], default="").strip()
            from_p = _get(row, ["from_point", "from", "from_point_id"], default="").strip()
            to_p = _get(row, ["to_point", "to", "to_point_id"], default="").strip()
            direction = float(_get(row, ["direction", "value"], default="0"))
            sigma = float(_get(row, ["sigma_direction", "sigma"], default=str(sigma_default)))
            set_id = _get(row, ["set_id", "set"], default="").strip()

            direction_rad = _angle_to_rad(direction, direction_unit)
            sigma_rad = _sigma_to_rad(sigma, sigma_unit)

            out.append(
                DirectionObservation(
                    id=obs_id,
                    obs_type=None,  # overwritten in __post_init__
                    value=direction_rad,
                    sigma=sigma_rad,
                    from_point_id=from_p,
                    to_point_id=to_p,
                    set_id=set_id,
                )
            )
    return out


def parse_angles_csv(
    path: str | Path,
    angle_unit: str = "degrees",
    sigma_unit: str = "radians",
    sigma_default: float = 0.00015,
) -> List[AngleObservation]:
    """Parse angle observations from CSV."""

    path = Path(path)
    out: List[AngleObservation] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            obs_id = _get(row, ["obs_id", "id"], default="").strip()
            at_p = _get(row, ["at_point", "at", "at_point_id"], default="").strip()
            from_p = _get(row, ["from_point", "from", "from_point_id"], default="").strip()
            to_p = _get(row, ["to_point", "to", "to_point_id"], default="").strip()
            angle = float(_get(row, ["angle", "value"], default="0"))
            sigma = float(_get(row, ["sigma_angle", "sigma"], default=str(sigma_default)))

            angle_rad = _angle_to_rad(angle, angle_unit)
            sigma_rad = _sigma_to_rad(sigma, sigma_unit)

            out.append(
                AngleObservation(
                    id=obs_id,
                    obs_type=None,  # overwritten in __post_init__
                    value=angle_rad,
                    sigma=sigma_rad,
                    at_point_id=at_p,
                    from_point_id=from_p,
                    to_point_id=to_p,
                )
            )

    return out


def parse_leveling_csv(
    path: str | Path,
    sigma_unit: str = "m",
    sigma_default: float = 0.001,
) -> List[HeightDifferenceObservation]:
    """Parse height difference (leveling) observations from CSV.

    Expected columns:
      - obs_id/id (optional)
      - from_point/from/from_id/from_point_id
      - to_point/to/to_id/to_point_id
      - dh/value/height_diff - height difference in meters
      - sigma/sigma_dh - standard deviation

    Args:
        path: Path to CSV file
        sigma_unit: Unit for sigma values ("m" for meters, "mm" for millimeters)
        sigma_default: Default sigma if not specified in CSV

    Returns:
        List of HeightDifferenceObservation objects
    """
    path = Path(path)
    out: List[HeightDifferenceObservation] = []

    # Sigma conversion factor
    sigma_scale = 0.001 if sigma_unit.lower() == "mm" else 1.0

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            obs_id = _get(row, ["obs_id", "id"], default="").strip()
            from_p = _get(row, ["from_point", "from", "from_id", "from_point_id"], default="").strip()
            to_p = _get(row, ["to_point", "to", "to_id", "to_point_id"], default="").strip()
            dh = float(_get(row, ["dh", "value", "height_diff", "delta_h"], default="0"))
            sigma_raw = float(_get(row, ["sigma", "sigma_dh"], default=str(sigma_default / sigma_scale)))
            sigma = sigma_raw * sigma_scale

            out.append(
                HeightDifferenceObservation(
                    id=obs_id,
                    obs_type=None,  # overwritten in __post_init__
                    value=dh,
                    sigma=sigma,
                    from_point_id=from_p,
                    to_point_id=to_p,
                )
            )
    return out


def parse_leveling_points_csv(path: str | Path) -> Dict[str, Point]:
    """Parse points with heights from CSV for leveling networks.

    Expected columns:
      - id/point_id
      - height/h/H
      - fixed_height/fixed_h/fixed (optional, default False)
      - name (optional)
      - easting/x, northing/y (optional, for visualization)

    Returns:
        Dictionary mapping point_id to Point objects
    """
    path = Path(path)
    points: Dict[str, Point] = {}

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = _get(row, ["point_id", "id", "pid"]).strip()
            if not pid:
                raise ValueError(f"Missing point_id in {path}")

            name = _get(row, ["name", "point_name"], default=pid).strip()
            height = float(_get(row, ["height", "h", "H", "elevation"], default="0"))

            # Coordinates are optional for leveling-only networks
            e = float(_get(row, ["easting", "x", "E", "east"], default="0"))
            n = float(_get(row, ["northing", "y", "N", "north"], default="0"))

            fixed_h = _parse_bool(_get(row, ["fixed_height", "fixed_h", "fixed"], default=""), default=False)

            points[pid] = Point(
                id=pid,
                name=name,
                easting=e,
                northing=n,
                height=height,
                fixed_height=fixed_h,
            )

    return points


def parse_traverse_file(
    path: str | Path,
    angle_unit: str = "degrees",
    direction_unit: str = "degrees",
    sigma_angle_unit: str = "radians",
    sigma_direction_unit: str = "radians",
) -> Network:
    """Parse a record-based traverse file.

    Line formats (comma-separated):
      POINT,point_id,name,easting,northing,fixed_e,fixed_n,sigma_e,sigma_n[,height,fixed_h]
      DIST,obs_id,from_point,to_point,distance,sigma_distance
      DIR,obs_id,from_point,to_point,direction,sigma_direction,set_id
      ANGLE,obs_id,at_point,from_point,to_point,angle,sigma_angle
      HDIFF,obs_id,from_point,to_point,dh,sigma_dh

    Lines starting with '#' or empty are ignored.
    """

    path = Path(path)

    points: Dict[str, Point] = {}
    observations: List[object] = []

    with path.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            rec = parts[0].upper()

            if rec == "POINT":
                if len(parts) < 5:
                    raise ValueError(f"Invalid POINT record: {line}")
                pid = parts[1]
                name = parts[2] if len(parts) > 2 else pid
                e = float(parts[3])
                n = float(parts[4])
                fixed_e = _parse_bool(parts[5] if len(parts) > 5 else False)
                fixed_n = _parse_bool(parts[6] if len(parts) > 6 else False)
                sigma_e = float(parts[7]) if len(parts) > 7 and parts[7] != "" else 0.0
                sigma_n = float(parts[8]) if len(parts) > 8 and parts[8] != "" else 0.0
                # Optional height fields
                height = float(parts[9]) if len(parts) > 9 and parts[9] != "" else None
                fixed_h = _parse_bool(parts[10] if len(parts) > 10 else False)

                points[pid] = Point(
                    id=pid,
                    name=name,
                    easting=e,
                    northing=n,
                    height=height,
                    fixed_easting=fixed_e,
                    fixed_northing=fixed_n,
                    fixed_height=fixed_h,
                    sigma_easting=sigma_e,
                    sigma_northing=sigma_n,
                )

            elif rec == "DIST":
                if len(parts) < 6:
                    raise ValueError(f"Invalid DIST record: {line}")
                obs_id, frm, to, dist, sigma = parts[1], parts[2], parts[3], float(parts[4]), float(parts[5])
                observations.append(
                    DistanceObservation(
                        id=obs_id,
                        obs_type=None,
                        value=dist,
                        sigma=sigma,
                        from_point_id=frm,
                        to_point_id=to,
                    )
                )

            elif rec == "DIR":
                if len(parts) < 6:
                    raise ValueError(f"Invalid DIR record: {line}")
                obs_id, frm, to = parts[1], parts[2], parts[3]
                direction = _angle_to_rad(float(parts[4]), direction_unit)
                sigma = _sigma_to_rad(float(parts[5]), sigma_direction_unit)
                set_id = parts[6] if len(parts) > 6 else ""
                observations.append(
                    DirectionObservation(
                        id=obs_id,
                        obs_type=None,
                        value=direction,
                        sigma=sigma,
                        from_point_id=frm,
                        to_point_id=to,
                        set_id=set_id,
                    )
                )

            elif rec == "ANGLE":
                if len(parts) < 7:
                    raise ValueError(f"Invalid ANGLE record: {line}")
                obs_id, at_p, frm, to = parts[1], parts[2], parts[3], parts[4]
                angle = _angle_to_rad(float(parts[5]), angle_unit)
                sigma = _sigma_to_rad(float(parts[6]), sigma_angle_unit)
                observations.append(
                    AngleObservation(
                        id=obs_id,
                        obs_type=None,
                        value=angle,
                        sigma=sigma,
                        at_point_id=at_p,
                        from_point_id=frm,
                        to_point_id=to,
                    )
                )

            elif rec == "HDIFF":
                # Height difference observation for leveling
                if len(parts) < 6:
                    raise ValueError(f"Invalid HDIFF record: {line}")
                obs_id, frm, to = parts[1], parts[2], parts[3]
                dh = float(parts[4])
                sigma = float(parts[5])
                observations.append(
                    HeightDifferenceObservation(
                        id=obs_id,
                        obs_type=None,
                        value=dh,
                        sigma=sigma,
                        from_point_id=frm,
                        to_point_id=to,
                    )
                )

            else:
                raise ValueError(f"Unknown record type '{rec}' in line: {line}")

    net = Network(points=points)
    for obs in observations:
        net.add_observation(obs)
    return net


def parse_gnss_baselines_csv(
    path: str | Path,
    covariance_format: str = "full",
) -> List[GnssBaselineObservation]:
    """Parse GNSS baseline observations from CSV.

    Expected columns for 'full' covariance format:
      - obs_id/id (optional)
      - from_point/from/from_id/from_point_id
      - to_point/to/to_id/to_point_id
      - dE/delta_E - Easting component (meters)
      - dN/delta_N - Northing component (meters)
      - dH/delta_H - Height component (meters)
      - cov_EE, cov_EN, cov_EH, cov_NN, cov_NH, cov_HH - covariance terms (m²)

    Expected columns for 'sigmas_corr' format:
      - obs_id/id (optional)
      - from_point/from/from_id/from_point_id
      - to_point/to/to_id/to_point_id
      - dE, dN, dH - baseline components (meters)
      - sigma_E, sigma_N, sigma_H - standard deviations (meters)
      - rho_EN, rho_EH, rho_NH - correlation coefficients (optional, default 0)

    Args:
        path: Path to CSV file
        covariance_format: "full" for covariance terms, "sigmas_corr" for sigmas + correlations

    Returns:
        List of GnssBaselineObservation objects
    """
    path = Path(path)
    out: List[GnssBaselineObservation] = []

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            obs_id = _get(row, ["obs_id", "id"], default="").strip()
            from_p = _get(row, ["from_point", "from", "from_id", "from_point_id"], default="").strip()
            to_p = _get(row, ["to_point", "to", "to_id", "to_point_id"], default="").strip()

            dE = float(_get(row, ["dE", "delta_E", "de"], default="0"))
            dN = float(_get(row, ["dN", "delta_N", "dn"], default="0"))
            dH = float(_get(row, ["dH", "delta_H", "dh"], default="0"))

            if covariance_format == "sigmas_corr":
                # Parse sigmas and correlations
                sigma_E = float(_get(row, ["sigma_E", "sE", "se"], default="0.01"))
                sigma_N = float(_get(row, ["sigma_N", "sN", "sn"], default="0.01"))
                sigma_H = float(_get(row, ["sigma_H", "sH", "sh"], default="0.01"))
                rho_EN = float(_get(row, ["rho_EN", "corr_EN"], default="0"))
                rho_EH = float(_get(row, ["rho_EH", "corr_EH"], default="0"))
                rho_NH = float(_get(row, ["rho_NH", "corr_NH"], default="0"))

                # Convert to covariance
                cov_EE = sigma_E ** 2
                cov_NN = sigma_N ** 2
                cov_HH = sigma_H ** 2
                cov_EN = rho_EN * sigma_E * sigma_N
                cov_EH = rho_EH * sigma_E * sigma_H
                cov_NH = rho_NH * sigma_N * sigma_H
            else:
                # Full covariance format
                cov_EE = float(_get(row, ["cov_EE", "var_E"], default="0.0001"))
                cov_EN = float(_get(row, ["cov_EN"], default="0"))
                cov_EH = float(_get(row, ["cov_EH"], default="0"))
                cov_NN = float(_get(row, ["cov_NN", "var_N"], default="0.0001"))
                cov_NH = float(_get(row, ["cov_NH"], default="0"))
                cov_HH = float(_get(row, ["cov_HH", "var_H"], default="0.0001"))

            out.append(
                GnssBaselineObservation(
                    id=obs_id,
                    obs_type=None,
                    value=0.0,  # Placeholder
                    sigma=1.0,  # Placeholder
                    from_point_id=from_p,
                    to_point_id=to_p,
                    dE=dE,
                    dN=dN,
                    dH=dH,
                    cov_EE=cov_EE,
                    cov_EN=cov_EN,
                    cov_EH=cov_EH,
                    cov_NN=cov_NN,
                    cov_NH=cov_NH,
                    cov_HH=cov_HH,
                )
            )

    return out


def parse_gnss_points_csv(path: str | Path) -> Dict[str, Point]:
    """Parse points with 3D coordinates from CSV for GNSS networks.

    Expected columns:
      - id/point_id
      - easting/x/E
      - northing/y/N
      - height/h/H
      - fixed_e/fixed_easting (optional)
      - fixed_n/fixed_northing (optional)
      - fixed_h/fixed_height (optional)
      - name (optional)

    Returns:
        Dictionary mapping point_id to Point objects
    """
    path = Path(path)
    points: Dict[str, Point] = {}

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = _get(row, ["point_id", "id", "pid"]).strip()
            if not pid:
                raise ValueError(f"Missing point_id in {path}")

            name = _get(row, ["name", "point_name"], default=pid).strip()
            e = float(_get(row, ["easting", "x", "E", "east"], default="0"))
            n = float(_get(row, ["northing", "y", "N", "north"], default="0"))
            h = float(_get(row, ["height", "h", "H", "elevation"], default="0"))

            fixed_e = _parse_bool(_get(row, ["fixed_e", "fixed_easting", "fixed_x"], default=""), default=False)
            fixed_n = _parse_bool(_get(row, ["fixed_n", "fixed_northing", "fixed_y"], default=""), default=False)
            fixed_h = _parse_bool(_get(row, ["fixed_h", "fixed_height", "fixed"], default=""), default=False)

            points[pid] = Point(
                id=pid,
                name=name,
                easting=e,
                northing=n,
                height=h,
                fixed_easting=fixed_e,
                fixed_northing=fixed_n,
                fixed_height=fixed_h,
            )

    return points
