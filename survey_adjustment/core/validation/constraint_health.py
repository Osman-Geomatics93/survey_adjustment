"""Constraint health analysis and auto-datum for survey networks.

This module provides structured analysis of network constraints (datum definition)
and optional auto-application of minimal constraints for common adjustment types.

No QGIS imports - pure Python for testability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.network import Network
    from ..models.point import Point


class ConstraintStatus(Enum):
    """Status of a constraint category."""
    OK = "ok"           # Properly constrained
    WARNING = "warning" # Marginal or suboptimal
    ERROR = "error"     # Missing or insufficient


@dataclass
class AppliedConstraint:
    """Record of an automatically applied constraint."""
    point_id: str
    constraint_type: str  # "fixed_easting", "fixed_northing", "fixed_height", "fixed_orientation"
    value: float
    reason: str

    def to_dict(self) -> Dict:
        return {
            "point_id": self.point_id,
            "constraint_type": self.constraint_type,
            "value": self.value,
            "reason": self.reason,
        }


@dataclass
class ConstraintHealth:
    """Structured summary of network constraint status.

    Provides a clear checklist of datum requirements and their status.
    """
    # Overall status
    is_solvable: bool = False

    # Horizontal datum (E, N)
    horizontal_status: ConstraintStatus = ConstraintStatus.ERROR
    horizontal_message: str = ""
    fixed_easting_points: List[str] = field(default_factory=list)
    fixed_northing_points: List[str] = field(default_factory=list)

    # Rotation/orientation constraint (for 2D with directions)
    orientation_status: ConstraintStatus = ConstraintStatus.OK
    orientation_message: str = ""
    direction_sets: List[str] = field(default_factory=list)

    # Height datum (for 1D, 3D, mixed with heights)
    height_status: ConstraintStatus = ConstraintStatus.OK
    height_message: str = ""
    fixed_height_points: List[str] = field(default_factory=list)
    height_required: bool = False

    # Network connectivity
    connectivity_status: ConstraintStatus = ConstraintStatus.OK
    connectivity_message: str = ""
    disconnected_points: List[str] = field(default_factory=list)

    # Degrees of freedom
    dof: int = 0
    dof_status: ConstraintStatus = ConstraintStatus.OK
    dof_message: str = ""
    num_observations: int = 0
    num_unknowns: int = 0

    # Actionable messages (what to fix)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Applied auto-constraints (if any)
    applied_constraints: List[AppliedConstraint] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Serialize to dictionary for JSON output."""
        return {
            "is_solvable": self.is_solvable,
            "horizontal": {
                "status": self.horizontal_status.value,
                "message": self.horizontal_message,
                "fixed_easting_points": self.fixed_easting_points,
                "fixed_northing_points": self.fixed_northing_points,
            },
            "orientation": {
                "status": self.orientation_status.value,
                "message": self.orientation_message,
                "direction_sets": self.direction_sets,
            },
            "height": {
                "status": self.height_status.value,
                "message": self.height_message,
                "fixed_height_points": self.fixed_height_points,
                "required": self.height_required,
            },
            "connectivity": {
                "status": self.connectivity_status.value,
                "message": self.connectivity_message,
                "disconnected_points": self.disconnected_points,
            },
            "degrees_of_freedom": {
                "value": self.dof,
                "status": self.dof_status.value,
                "message": self.dof_message,
                "num_observations": self.num_observations,
                "num_unknowns": self.num_unknowns,
            },
            "errors": self.errors,
            "warnings": self.warnings,
            "applied_constraints": [c.to_dict() for c in self.applied_constraints],
        }


def _get_observation_points(network: "Network") -> Tuple[Set[str], Set[str], Set[str]]:
    """Get sets of points involved in different observation types.

    Returns:
        Tuple of (classical_points, gnss_points, leveling_points)
    """
    from ..models.observation import (
        DistanceObservation,
        DirectionObservation,
        AngleObservation,
        HeightDifferenceObservation,
        GnssBaselineObservation,
    )

    classical = set()
    gnss = set()
    leveling = set()

    for obs in network.observations:
        if not obs.enabled:
            continue
        if isinstance(obs, (DistanceObservation, DirectionObservation)):
            classical.add(obs.from_point_id)
            classical.add(obs.to_point_id)
        elif isinstance(obs, AngleObservation):
            classical.add(obs.at_point_id)
            classical.add(obs.from_point_id)
            classical.add(obs.to_point_id)
        elif isinstance(obs, GnssBaselineObservation):
            gnss.add(obs.from_point_id)
            gnss.add(obs.to_point_id)
        elif isinstance(obs, HeightDifferenceObservation):
            leveling.add(obs.from_point_id)
            leveling.add(obs.to_point_id)

    return classical, gnss, leveling


def _check_connectivity(network: "Network", point_ids: Set[str], obs_filter=None) -> List[str]:
    """Check if all points in point_ids are connected via observations.

    Args:
        network: The network to check
        point_ids: Set of point IDs that should be connected
        obs_filter: Optional function to filter observations

    Returns:
        List of disconnected point IDs (empty if fully connected)
    """
    if len(point_ids) <= 1:
        return []

    # Build adjacency list
    adj: Dict[str, Set[str]] = {pid: set() for pid in point_ids}

    from ..models.observation import (
        DistanceObservation,
        DirectionObservation,
        AngleObservation,
        HeightDifferenceObservation,
        GnssBaselineObservation,
    )

    for obs in network.observations:
        if not obs.enabled:
            continue
        if obs_filter and not obs_filter(obs):
            continue

        if isinstance(obs, (DistanceObservation, DirectionObservation, GnssBaselineObservation, HeightDifferenceObservation)):
            if obs.from_point_id in adj and obs.to_point_id in adj:
                adj[obs.from_point_id].add(obs.to_point_id)
                adj[obs.to_point_id].add(obs.from_point_id)
        elif isinstance(obs, AngleObservation):
            # Angle connects at_point to both from_point and to_point
            if obs.at_point_id in adj:
                if obs.from_point_id in adj:
                    adj[obs.at_point_id].add(obs.from_point_id)
                    adj[obs.from_point_id].add(obs.at_point_id)
                if obs.to_point_id in adj:
                    adj[obs.at_point_id].add(obs.to_point_id)
                    adj[obs.to_point_id].add(obs.at_point_id)

    # BFS from first point
    start = next(iter(point_ids))
    visited = {start}
    queue = [start]

    while queue:
        current = queue.pop(0)
        for neighbor in adj.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return sorted(point_ids - visited)


def _count_direction_sets(network: "Network") -> List[str]:
    """Get list of unique direction set IDs."""
    from ..models.observation import DirectionObservation

    sets = set()
    for obs in network.observations:
        if obs.enabled and isinstance(obs, DirectionObservation):
            sets.add(obs.set_id)
    return sorted(sets)


def analyze_constraint_health(
    network: "Network",
    adjustment_type: str = "auto",
) -> ConstraintHealth:
    """Analyze network constraints and return structured health summary.

    Args:
        network: Network to analyze
        adjustment_type: One of "2d", "1d", "3d", "mixed", or "auto" (detect from observations)

    Returns:
        ConstraintHealth with detailed status and actionable messages
    """
    from ..models.observation import (
        DistanceObservation,
        DirectionObservation,
        AngleObservation,
        HeightDifferenceObservation,
        GnssBaselineObservation,
    )

    health = ConstraintHealth()

    # Detect observation types
    classical_points, gnss_points, leveling_points = _get_observation_points(network)
    has_classical = len(classical_points) > 0
    has_gnss = len(gnss_points) > 0
    has_leveling = len(leveling_points) > 0

    # Auto-detect adjustment type
    if adjustment_type == "auto":
        if has_gnss or (has_classical and has_leveling):
            adjustment_type = "mixed"
        elif has_leveling and not has_classical and not has_gnss:
            adjustment_type = "1d"
        elif has_gnss and not has_classical and not has_leveling:
            adjustment_type = "3d"
        else:
            adjustment_type = "2d"

    # Get fixed points
    fixed_e = [p.id for p in network.points.values() if p.fixed_easting]
    fixed_n = [p.id for p in network.points.values() if p.fixed_northing]
    fixed_h = [p.id for p in network.points.values() if p.fixed_height]

    health.fixed_easting_points = sorted(fixed_e)
    health.fixed_northing_points = sorted(fixed_n)
    health.fixed_height_points = sorted(fixed_h)

    # Direction sets
    health.direction_sets = _count_direction_sets(network)

    # Analyze based on adjustment type
    if adjustment_type == "2d":
        _analyze_2d_constraints(network, health, classical_points)
    elif adjustment_type == "1d":
        _analyze_1d_constraints(network, health, leveling_points)
    elif adjustment_type == "3d":
        _analyze_3d_constraints(network, health, gnss_points)
    else:  # mixed
        _analyze_mixed_constraints(network, health, classical_points, gnss_points, leveling_points)

    # Determine overall solvability
    health.is_solvable = (
        health.horizontal_status != ConstraintStatus.ERROR and
        health.height_status != ConstraintStatus.ERROR and
        health.orientation_status != ConstraintStatus.ERROR and
        health.connectivity_status != ConstraintStatus.ERROR and
        health.dof_status != ConstraintStatus.ERROR
    )

    return health


def _analyze_2d_constraints(
    network: "Network",
    health: ConstraintHealth,
    classical_points: Set[str],
) -> None:
    """Analyze constraints for 2D classical adjustment."""
    from ..models.observation import DirectionObservation

    # Horizontal datum
    if not health.fixed_easting_points:
        health.horizontal_status = ConstraintStatus.ERROR
        health.horizontal_message = "No fixed Easting (E) coordinate"
        health.errors.append(
            "No horizontal datum: fix E and N coordinates of at least one point, "
            "or fix E of one point and N of another"
        )
    elif not health.fixed_northing_points:
        health.horizontal_status = ConstraintStatus.ERROR
        health.horizontal_message = "No fixed Northing (N) coordinate"
        health.errors.append(
            "Incomplete horizontal datum: fix N coordinate of at least one point"
        )
    else:
        health.horizontal_status = ConstraintStatus.OK
        health.horizontal_message = f"Fixed E: {health.fixed_easting_points}, Fixed N: {health.fixed_northing_points}"

    # Orientation constraint (only needed if directions present)
    has_directions = any(
        isinstance(obs, DirectionObservation) and obs.enabled
        for obs in network.observations
    )

    if has_directions:
        if len(health.direction_sets) > 0:
            # Directions present - orientation will be solved as unknowns
            # Check if rotation is constrained (need 2+ fixed points or angles)
            num_fully_fixed = len(set(health.fixed_easting_points) & set(health.fixed_northing_points))
            has_angles = any(
                hasattr(obs, 'at_point_id') and obs.enabled
                for obs in network.observations
            )

            if num_fully_fixed >= 2 or has_angles:
                health.orientation_status = ConstraintStatus.OK
                health.orientation_message = f"{len(health.direction_sets)} direction set(s), rotation constrained"
            else:
                health.orientation_status = ConstraintStatus.WARNING
                health.orientation_message = "Rotation may be weakly constrained"
                health.warnings.append(
                    "Rotation constraint is weak: consider fixing E/N of a second point, "
                    "or adding angle observations"
                )
        else:
            health.orientation_status = ConstraintStatus.OK
            health.orientation_message = "No direction observations"
    else:
        health.orientation_status = ConstraintStatus.OK
        health.orientation_message = "No direction observations"

    # Height not required for 2D
    health.height_required = False
    health.height_status = ConstraintStatus.OK
    health.height_message = "Not required for 2D adjustment"

    # Connectivity
    disconnected = _check_connectivity(network, classical_points)
    if disconnected:
        health.connectivity_status = ConstraintStatus.ERROR
        health.connectivity_message = f"Disconnected points: {disconnected}"
        health.disconnected_points = disconnected
        health.errors.append(
            f"Network is disconnected: points {disconnected} are not connected to the main network"
        )
    else:
        health.connectivity_status = ConstraintStatus.OK
        health.connectivity_message = "All points connected"

    # DOF
    _calculate_2d_dof(network, health, classical_points)


def _analyze_1d_constraints(
    network: "Network",
    health: ConstraintHealth,
    leveling_points: Set[str],
) -> None:
    """Analyze constraints for 1D leveling adjustment."""
    from ..models.observation import HeightDifferenceObservation

    # Horizontal not required for 1D
    health.horizontal_status = ConstraintStatus.OK
    health.horizontal_message = "Not required for 1D leveling"

    health.orientation_status = ConstraintStatus.OK
    health.orientation_message = "Not applicable for 1D leveling"

    # Height datum is required
    health.height_required = True

    # Check for fixed heights among leveling points
    fixed_h_in_leveling = [p for p in health.fixed_height_points if p in leveling_points]

    if not fixed_h_in_leveling:
        health.height_status = ConstraintStatus.ERROR
        health.height_message = "No fixed height benchmark"
        health.errors.append(
            "No height datum: set fixed_height=True for at least one benchmark point"
        )
    else:
        health.height_status = ConstraintStatus.OK
        health.height_message = f"Fixed heights: {fixed_h_in_leveling}"

    # Connectivity (only leveling observations)
    def leveling_filter(obs):
        return isinstance(obs, HeightDifferenceObservation)

    disconnected = _check_connectivity(network, leveling_points, leveling_filter)
    if disconnected:
        health.connectivity_status = ConstraintStatus.ERROR
        health.connectivity_message = f"Disconnected points: {disconnected}"
        health.disconnected_points = disconnected
        health.errors.append(
            f"Leveling network is disconnected: points {disconnected} have no height difference path to fixed benchmarks"
        )
    else:
        health.connectivity_status = ConstraintStatus.OK
        health.connectivity_message = "All points connected"

    # DOF
    _calculate_1d_dof(network, health, leveling_points)


def _analyze_3d_constraints(
    network: "Network",
    health: ConstraintHealth,
    gnss_points: Set[str],
) -> None:
    """Analyze constraints for 3D GNSS baseline adjustment."""
    from ..models.observation import GnssBaselineObservation

    # For 3D GNSS, need E, N, H each constrained
    fixed_e_in_gnss = [p for p in health.fixed_easting_points if p in gnss_points]
    fixed_n_in_gnss = [p for p in health.fixed_northing_points if p in gnss_points]
    fixed_h_in_gnss = [p for p in health.fixed_height_points if p in gnss_points]

    # Horizontal datum
    errors_h = []
    if not fixed_e_in_gnss:
        errors_h.append("No fixed Easting")
    if not fixed_n_in_gnss:
        errors_h.append("No fixed Northing")

    if errors_h:
        health.horizontal_status = ConstraintStatus.ERROR
        health.horizontal_message = "; ".join(errors_h)
        health.errors.append(
            f"Incomplete 3D datum: {', '.join(errors_h).lower()}. "
            "Fix E, N, and H coordinates of at least one reference station."
        )
    else:
        health.horizontal_status = ConstraintStatus.OK
        health.horizontal_message = f"Fixed E: {fixed_e_in_gnss}, Fixed N: {fixed_n_in_gnss}"

    # Orientation not applicable for GNSS (no directions)
    health.orientation_status = ConstraintStatus.OK
    health.orientation_message = "Not applicable for GNSS baselines"

    # Height datum
    health.height_required = True
    if not fixed_h_in_gnss:
        health.height_status = ConstraintStatus.ERROR
        health.height_message = "No fixed Height"
        if "Incomplete 3D datum" not in str(health.errors):
            health.errors.append(
                "No height datum: fix H coordinate of at least one reference station"
            )
    else:
        health.height_status = ConstraintStatus.OK
        health.height_message = f"Fixed heights: {fixed_h_in_gnss}"

    # Connectivity
    def gnss_filter(obs):
        return isinstance(obs, GnssBaselineObservation)

    disconnected = _check_connectivity(network, gnss_points, gnss_filter)
    if disconnected:
        health.connectivity_status = ConstraintStatus.ERROR
        health.connectivity_message = f"Disconnected points: {disconnected}"
        health.disconnected_points = disconnected
        health.errors.append(
            f"GNSS network is disconnected: points {disconnected} have no baseline path to fixed station"
        )
    else:
        health.connectivity_status = ConstraintStatus.OK
        health.connectivity_message = "All points connected"

    # DOF
    _calculate_3d_dof(network, health, gnss_points)


def _analyze_mixed_constraints(
    network: "Network",
    health: ConstraintHealth,
    classical_points: Set[str],
    gnss_points: Set[str],
    leveling_points: Set[str],
) -> None:
    """Analyze constraints for mixed adjustment."""
    from ..models.observation import DirectionObservation

    all_points = classical_points | gnss_points | leveling_points
    has_horizontal = bool(classical_points | gnss_points)
    has_height = bool(gnss_points | leveling_points)

    # Horizontal datum (needed if classical or GNSS)
    if has_horizontal:
        if not health.fixed_easting_points:
            health.horizontal_status = ConstraintStatus.ERROR
            health.horizontal_message = "No fixed Easting"
            health.errors.append(
                "No horizontal datum: fix E and N of at least one point"
            )
        elif not health.fixed_northing_points:
            health.horizontal_status = ConstraintStatus.ERROR
            health.horizontal_message = "No fixed Northing"
            health.errors.append(
                "Incomplete horizontal datum: fix N of at least one point"
            )
        else:
            health.horizontal_status = ConstraintStatus.OK
            health.horizontal_message = f"Fixed E: {health.fixed_easting_points}, Fixed N: {health.fixed_northing_points}"
    else:
        health.horizontal_status = ConstraintStatus.OK
        health.horizontal_message = "Not required (leveling only)"

    # Orientation (only for classical with directions)
    has_directions = any(
        isinstance(obs, DirectionObservation) and obs.enabled
        for obs in network.observations
    )
    if has_directions:
        num_fully_fixed = len(set(health.fixed_easting_points) & set(health.fixed_northing_points))
        if num_fully_fixed >= 2:
            health.orientation_status = ConstraintStatus.OK
            health.orientation_message = "Constrained by multiple fixed points"
        else:
            health.orientation_status = ConstraintStatus.WARNING
            health.orientation_message = "May be weakly constrained"
            health.warnings.append(
                "Direction orientations may be poorly constrained with only one fixed point"
            )
    else:
        health.orientation_status = ConstraintStatus.OK
        health.orientation_message = "No direction observations"

    # Height datum (needed if GNSS or leveling)
    health.height_required = has_height
    if has_height:
        height_points = gnss_points | leveling_points
        fixed_h_relevant = [p for p in health.fixed_height_points if p in height_points]

        if not fixed_h_relevant:
            health.height_status = ConstraintStatus.ERROR
            health.height_message = "No fixed height for GNSS/leveling"
            health.errors.append(
                "No height datum: fix H of at least one point involved in GNSS or leveling"
            )
        else:
            health.height_status = ConstraintStatus.OK
            health.height_message = f"Fixed heights: {fixed_h_relevant}"
    else:
        health.height_status = ConstraintStatus.OK
        health.height_message = "Not required (2D classical only)"

    # Connectivity (across all observation types)
    disconnected = _check_connectivity(network, all_points)
    if disconnected:
        health.connectivity_status = ConstraintStatus.ERROR
        health.connectivity_message = f"Disconnected points: {disconnected}"
        health.disconnected_points = disconnected
        health.errors.append(
            f"Network is disconnected: points {disconnected} are isolated"
        )
    else:
        health.connectivity_status = ConstraintStatus.OK
        health.connectivity_message = "All points connected"

    # DOF
    _calculate_mixed_dof(network, health, classical_points, gnss_points, leveling_points)


def _calculate_2d_dof(network: "Network", health: ConstraintHealth, points: Set[str]) -> None:
    """Calculate degrees of freedom for 2D adjustment."""
    from ..models.observation import DistanceObservation, DirectionObservation, AngleObservation

    # Count observations
    num_obs = 0
    for obs in network.observations:
        if obs.enabled and isinstance(obs, (DistanceObservation, DirectionObservation, AngleObservation)):
            num_obs += 1

    # Count unknowns: 2 coords per free point + 1 orientation per direction set
    num_unknowns = 0
    for pid in points:
        p = network.points.get(pid)
        if p:
            if not p.fixed_easting:
                num_unknowns += 1
            if not p.fixed_northing:
                num_unknowns += 1
    num_unknowns += len(health.direction_sets)

    health.num_observations = num_obs
    health.num_unknowns = num_unknowns
    health.dof = num_obs - num_unknowns

    if health.dof < 0:
        health.dof_status = ConstraintStatus.ERROR
        health.dof_message = f"Underdetermined: {num_obs} obs, {num_unknowns} unknowns"
        health.errors.append(
            f"Underdetermined system: need at least {num_unknowns - num_obs} more observations"
        )
    elif health.dof == 0:
        health.dof_status = ConstraintStatus.WARNING
        health.dof_message = "Just-determined (DOF=0)"
        health.warnings.append(
            "Network is just-determined with no redundancy for outlier detection"
        )
    else:
        health.dof_status = ConstraintStatus.OK
        health.dof_message = f"Redundant: DOF = {health.dof}"


def _calculate_1d_dof(network: "Network", health: ConstraintHealth, points: Set[str]) -> None:
    """Calculate degrees of freedom for 1D leveling adjustment."""
    from ..models.observation import HeightDifferenceObservation

    num_obs = sum(1 for obs in network.observations
                  if obs.enabled and isinstance(obs, HeightDifferenceObservation))

    num_unknowns = sum(1 for pid in points
                       if not network.points[pid].fixed_height)

    health.num_observations = num_obs
    health.num_unknowns = num_unknowns
    health.dof = num_obs - num_unknowns

    if health.dof < 0:
        health.dof_status = ConstraintStatus.ERROR
        health.dof_message = f"Underdetermined: {num_obs} obs, {num_unknowns} unknowns"
        health.errors.append(
            f"Underdetermined system: need at least {num_unknowns - num_obs} more height differences"
        )
    elif health.dof == 0:
        health.dof_status = ConstraintStatus.WARNING
        health.dof_message = "Just-determined (DOF=0)"
        health.warnings.append(
            "Leveling network is just-determined with no redundancy"
        )
    else:
        health.dof_status = ConstraintStatus.OK
        health.dof_message = f"Redundant: DOF = {health.dof}"


def _calculate_3d_dof(network: "Network", health: ConstraintHealth, points: Set[str]) -> None:
    """Calculate degrees of freedom for 3D GNSS adjustment."""
    from ..models.observation import GnssBaselineObservation

    num_baselines = sum(1 for obs in network.observations
                        if obs.enabled and isinstance(obs, GnssBaselineObservation))
    num_obs = 3 * num_baselines  # Each baseline has 3 components

    num_unknowns = 0
    for pid in points:
        p = network.points.get(pid)
        if p:
            if not p.fixed_easting:
                num_unknowns += 1
            if not p.fixed_northing:
                num_unknowns += 1
            if not p.fixed_height:
                num_unknowns += 1

    health.num_observations = num_obs
    health.num_unknowns = num_unknowns
    health.dof = num_obs - num_unknowns

    if health.dof < 0:
        health.dof_status = ConstraintStatus.ERROR
        health.dof_message = f"Underdetermined: {num_obs} obs components, {num_unknowns} unknowns"
        health.errors.append(
            f"Underdetermined system: need at least {(num_unknowns - num_obs + 2) // 3} more baselines"
        )
    elif health.dof == 0:
        health.dof_status = ConstraintStatus.WARNING
        health.dof_message = "Just-determined (DOF=0)"
        health.warnings.append(
            "GNSS network is just-determined with no redundancy"
        )
    else:
        health.dof_status = ConstraintStatus.OK
        health.dof_message = f"Redundant: DOF = {health.dof}"


def _calculate_mixed_dof(
    network: "Network",
    health: ConstraintHealth,
    classical_points: Set[str],
    gnss_points: Set[str],
    leveling_points: Set[str],
) -> None:
    """Calculate degrees of freedom for mixed adjustment."""
    from ..models.observation import (
        DistanceObservation, DirectionObservation, AngleObservation,
        GnssBaselineObservation, HeightDifferenceObservation,
    )

    # Count observations
    num_classical = sum(1 for obs in network.observations
                        if obs.enabled and isinstance(obs, (DistanceObservation, DirectionObservation, AngleObservation)))
    num_gnss = sum(1 for obs in network.observations
                   if obs.enabled and isinstance(obs, GnssBaselineObservation))
    num_leveling = sum(1 for obs in network.observations
                       if obs.enabled and isinstance(obs, HeightDifferenceObservation))

    num_obs = num_classical + 3 * num_gnss + num_leveling

    # Count unknowns
    all_points = classical_points | gnss_points | leveling_points
    horizontal_points = classical_points | gnss_points
    height_points = gnss_points | leveling_points

    num_unknowns = 0
    for pid in all_points:
        p = network.points.get(pid)
        if p:
            if pid in horizontal_points:
                if not p.fixed_easting:
                    num_unknowns += 1
                if not p.fixed_northing:
                    num_unknowns += 1
            if pid in height_points:
                if not p.fixed_height:
                    num_unknowns += 1

    num_unknowns += len(health.direction_sets)

    health.num_observations = num_obs
    health.num_unknowns = num_unknowns
    health.dof = num_obs - num_unknowns

    if health.dof < 0:
        health.dof_status = ConstraintStatus.ERROR
        health.dof_message = f"Underdetermined: {num_obs} obs, {num_unknowns} unknowns"
        health.errors.append(
            f"Underdetermined system: need at least {num_unknowns - num_obs} more observations"
        )
    elif health.dof == 0:
        health.dof_status = ConstraintStatus.WARNING
        health.dof_message = "Just-determined (DOF=0)"
        health.warnings.append(
            "Network is just-determined with no redundancy"
        )
    else:
        health.dof_status = ConstraintStatus.OK
        health.dof_message = f"Redundant: DOF = {health.dof}"


def apply_minimal_constraints(
    network: "Network",
    adjustment_type: str = "auto",
    reference_point: Optional[str] = None,
) -> List[AppliedConstraint]:
    """Apply minimal datum constraints to make network solvable.

    This function modifies the network in-place and returns a list of
    applied constraints for the audit trail.

    Args:
        network: Network to modify (modified in-place)
        adjustment_type: One of "2d", "1d", "3d", "mixed", or "auto"
        reference_point: Specific point ID to use as datum origin.
                        If None, uses first point alphabetically.

    Returns:
        List of AppliedConstraint records describing what was applied
    """
    from ..models.observation import (
        DistanceObservation, DirectionObservation, AngleObservation,
        GnssBaselineObservation, HeightDifferenceObservation,
    )

    applied: List[AppliedConstraint] = []

    # Detect observation types
    classical_points, gnss_points, leveling_points = _get_observation_points(network)
    has_classical = len(classical_points) > 0
    has_gnss = len(gnss_points) > 0
    has_leveling = len(leveling_points) > 0

    # Auto-detect adjustment type
    if adjustment_type == "auto":
        if has_gnss or (has_classical and has_leveling):
            adjustment_type = "mixed"
        elif has_leveling and not has_classical and not has_gnss:
            adjustment_type = "1d"
        elif has_gnss and not has_classical and not has_leveling:
            adjustment_type = "3d"
        else:
            adjustment_type = "2d"

    # Determine reference point
    if reference_point is None:
        if adjustment_type == "2d":
            candidates = classical_points
        elif adjustment_type == "1d":
            candidates = leveling_points
        elif adjustment_type == "3d":
            candidates = gnss_points
        else:  # mixed
            candidates = classical_points | gnss_points | leveling_points

        if candidates:
            reference_point = sorted(candidates)[0]

    if reference_point is None or reference_point not in network.points:
        return applied  # Cannot apply constraints

    ref_point = network.points[reference_point]

    # Apply constraints based on adjustment type
    if adjustment_type == "2d":
        applied.extend(_apply_2d_constraints(network, ref_point))
    elif adjustment_type == "1d":
        applied.extend(_apply_1d_constraints(network, ref_point))
    elif adjustment_type == "3d":
        applied.extend(_apply_3d_constraints(network, ref_point))
    else:  # mixed
        applied.extend(_apply_mixed_constraints(network, ref_point, has_classical, has_gnss, has_leveling))

    return applied


def _apply_2d_constraints(network: "Network", ref_point: "Point") -> List[AppliedConstraint]:
    """Apply minimal 2D constraints.

    For 2D adjustments with distances only (no directions), we need 3 constraints:
    - 2 for translation (E, N of one point)
    - 1 for rotation (E or N of a second point)

    For 2D adjustments with directions, the orientation unknowns handle rotation,
    so only 2 constraints are needed.
    """
    from ..models.observation import DirectionObservation

    applied = []

    # Fix E and N of reference point for translation
    if not ref_point.fixed_easting:
        ref_point.fixed_easting = True
        applied.append(AppliedConstraint(
            point_id=ref_point.id,
            constraint_type="fixed_easting",
            value=ref_point.easting,
            reason="Auto-datum: translation constraint (E)"
        ))

    if not ref_point.fixed_northing:
        ref_point.fixed_northing = True
        applied.append(AppliedConstraint(
            point_id=ref_point.id,
            constraint_type="fixed_northing",
            value=ref_point.northing,
            reason="Auto-datum: translation constraint (N)"
        ))

    # Check if we have direction observations (which handle rotation via orientation unknowns)
    has_directions = any(
        isinstance(obs, DirectionObservation) and obs.enabled
        for obs in network.observations
    )

    # If no directions, we need an additional constraint for rotation
    if not has_directions:
        # Find a second point to constrain
        classical_points, _, _ = _get_observation_points(network)
        other_points = sorted(p for p in classical_points if p != ref_point.id)

        if other_points:
            second_point_id = other_points[0]
            second_point = network.points[second_point_id]

            # Fix E of second point for rotation constraint
            if not second_point.fixed_easting:
                second_point.fixed_easting = True
                applied.append(AppliedConstraint(
                    point_id=second_point.id,
                    constraint_type="fixed_easting",
                    value=second_point.easting,
                    reason="Auto-datum: rotation constraint (E)"
                ))

    return applied


def _apply_1d_constraints(network: "Network", ref_point: "Point") -> List[AppliedConstraint]:
    """Apply minimal 1D constraints."""
    applied = []

    if not ref_point.fixed_height:
        # Ensure point has a height value
        if ref_point.height is None:
            ref_point.height = 0.0
        ref_point.fixed_height = True
        applied.append(AppliedConstraint(
            point_id=ref_point.id,
            constraint_type="fixed_height",
            value=ref_point.height,
            reason="Auto-datum: benchmark height constraint"
        ))

    return applied


def _apply_3d_constraints(network: "Network", ref_point: "Point") -> List[AppliedConstraint]:
    """Apply minimal 3D GNSS constraints."""
    applied = []

    if not ref_point.fixed_easting:
        ref_point.fixed_easting = True
        applied.append(AppliedConstraint(
            point_id=ref_point.id,
            constraint_type="fixed_easting",
            value=ref_point.easting,
            reason="Auto-datum: 3D reference (E)"
        ))

    if not ref_point.fixed_northing:
        ref_point.fixed_northing = True
        applied.append(AppliedConstraint(
            point_id=ref_point.id,
            constraint_type="fixed_northing",
            value=ref_point.northing,
            reason="Auto-datum: 3D reference (N)"
        ))

    if not ref_point.fixed_height:
        if ref_point.height is None:
            ref_point.height = 0.0
        ref_point.fixed_height = True
        applied.append(AppliedConstraint(
            point_id=ref_point.id,
            constraint_type="fixed_height",
            value=ref_point.height,
            reason="Auto-datum: 3D reference (H)"
        ))

    return applied


def _apply_mixed_constraints(
    network: "Network",
    ref_point: "Point",
    has_classical: bool,
    has_gnss: bool,
    has_leveling: bool,
) -> List[AppliedConstraint]:
    """Apply minimal mixed constraints."""
    applied = []

    needs_horizontal = has_classical or has_gnss
    needs_height = has_gnss or has_leveling

    if needs_horizontal:
        if not ref_point.fixed_easting:
            ref_point.fixed_easting = True
            applied.append(AppliedConstraint(
                point_id=ref_point.id,
                constraint_type="fixed_easting",
                value=ref_point.easting,
                reason="Auto-datum: horizontal reference (E)"
            ))

        if not ref_point.fixed_northing:
            ref_point.fixed_northing = True
            applied.append(AppliedConstraint(
                point_id=ref_point.id,
                constraint_type="fixed_northing",
                value=ref_point.northing,
                reason="Auto-datum: horizontal reference (N)"
            ))

    if needs_height:
        if not ref_point.fixed_height:
            if ref_point.height is None:
                ref_point.height = 0.0
            ref_point.fixed_height = True
            applied.append(AppliedConstraint(
                point_id=ref_point.id,
                constraint_type="fixed_height",
                value=ref_point.height,
                reason="Auto-datum: height reference"
            ))

    return applied


def format_validation_message(health: ConstraintHealth) -> str:
    """Format constraint health as a human-readable validation message.

    Returns multi-line string suitable for error display.
    """
    lines = []

    if health.errors:
        lines.append("Errors:")
        for err in health.errors:
            lines.append(f"  - {err}")

    if health.warnings:
        if lines:
            lines.append("")
        lines.append("Warnings:")
        for warn in health.warnings:
            lines.append(f"  - {warn}")

    if health.applied_constraints:
        if lines:
            lines.append("")
        lines.append("Auto-applied constraints:")
        for c in health.applied_constraints:
            lines.append(f"  - {c.point_id}: {c.constraint_type} = {c.value} ({c.reason})")

    return "\n".join(lines) if lines else "Constraints OK"
