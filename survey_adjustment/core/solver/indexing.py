"""survey_adjustment.core.solver.indexing

Parameter indexing helpers.

The least-squares solver works with a single parameter vector. This module
builds a stable mapping from logical unknowns to vector indices.

Unknown types:
  - Point coordinate components (Easting, Northing) for non-fixed components
  - Direction-set orientations (one per DirectionObservation.set_id)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..models.network import Network
from ..models.observation import DirectionObservation


@dataclass(frozen=True)
class ParameterIndex:
    """Mapping of unknown parameters to indices in the solver vector."""

    coord_index: Dict[Tuple[str, str], int]          # (point_id, 'E'|'N') -> idx
    orientation_index: Dict[str, int]               # direction set_id -> idx
    num_params: int
    coord_order: List[Tuple[str, str]]
    orientation_order: List[str]


def build_parameter_index(network: Network, use_direction_orientations: bool = True) -> ParameterIndex:
    """Build a stable parameter index.

    Ordering (stable / reproducible):
      1) point IDs sorted ascending
      2) components in the order E then N (if that component is not fixed)
      3) direction set IDs sorted ascending (if enabled)
    """

    coord_index: Dict[Tuple[str, str], int] = {}
    orientation_index: Dict[str, int] = {}

    coord_order: List[Tuple[str, str]] = []
    orientation_order: List[str] = []

    idx = 0
    for pid in sorted(network.points.keys()):
        p = network.points[pid]
        if not p.fixed_easting:
            coord_index[(pid, 'E')] = idx
            coord_order.append((pid, 'E'))
            idx += 1
        if not p.fixed_northing:
            coord_index[(pid, 'N')] = idx
            coord_order.append((pid, 'N'))
            idx += 1

    if use_direction_orientations:
        set_ids = set()
        for obs in network.get_enabled_observations():
            if isinstance(obs, DirectionObservation):
                set_ids.add(obs.set_id)
        for set_id in sorted(set_ids):
            orientation_index[set_id] = idx
            orientation_order.append(set_id)
            idx += 1

    return ParameterIndex(
        coord_index=coord_index,
        orientation_index=orientation_index,
        num_params=idx,
        coord_order=coord_order,
        orientation_order=orientation_order,
    )


def count_observations(network: Network) -> int:
    """Count enabled observations."""
    return len(network.get_enabled_observations())


def compute_degrees_of_freedom(network: Network, index: ParameterIndex) -> int:
    """Compute degrees of freedom (redundancy): m - n."""
    return count_observations(network) - index.num_params


def validate_network_for_adjustment(network: Network, index: ParameterIndex) -> List[str]:
    """Pre-adjustment validation specific to the solver.

    Network.validate() already covers basic checks (missing points, connectivity,
    min observations, etc.). Here we add a minimal 2D datum check.

    In 2D, a practical minimal datum definition is:
      - at least one fully-fixed point (translation)
      - at least one additional fixed coordinate component on another point
        (rotation)
    """
    errors = list(network.validate())

    # Additional datum constraint check
    fully_fixed = [p for p in network.points.values() if p.fixed_easting and p.fixed_northing]
    if not fully_fixed:
        # Network.validate already flags, but keep a clear solver-specific message.
        errors.append("Adjustment requires at least one fully fixed point to define translation")

    # Count fixed coordinate components and how many distinct points provide them
    fixed_components = []
    for pid, p in network.points.items():
        if p.fixed_easting:
            fixed_components.append((pid, 'E'))
        if p.fixed_northing:
            fixed_components.append((pid, 'N'))

    distinct_fixed_points = {pid for pid, _ in fixed_components}
    if len(fixed_components) < 3 or len(distinct_fixed_points) < 2:
        errors.append(
            "Insufficient datum constraints for 2D adjustment: need at least 3 fixed coordinate "
            "components across at least 2 points (e.g., one fixed point + one additional fixed component)"
        )

    dof = compute_degrees_of_freedom(network, index)
    if dof < 0:
        errors.append(
            f"Insufficient observations for unknowns: dof={dof} (need m >= n)."
        )

    return errors
