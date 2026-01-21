"""
Network class for survey network adjustment.

The Network class is the main container for survey data, holding:
- Points (control and unknown stations)
- Observations (distances, directions, angles)

It provides methods for:
- Adding/retrieving points and observations
- Grouping direction observations by set
- Validating network geometry and connectivity
- Serialization to/from dictionary/JSON
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict

from .point import Point
from .observation import (
    Observation,
    ObservationType,
    DistanceObservation,
    DirectionObservation,
    AngleObservation
)


@dataclass
class Network:
    """
    Container for a survey adjustment network.

    Attributes:
        name: Human-readable name for the network
        points: Dictionary mapping point IDs to Point objects
        observations: List of all observations in the network
    """

    name: str = "Unnamed Network"
    points: Dict[str, Point] = field(default_factory=dict)
    observations: List[Observation] = field(default_factory=list)

    def get_point(self, point_id: str) -> Point:
        """
        Retrieve a point by ID.

        Args:
            point_id: The unique identifier of the point

        Returns:
            The Point object

        Raises:
            KeyError: If point_id is not found
        """
        if point_id not in self.points:
            raise KeyError(f"Point '{point_id}' not found in network")
        return self.points[point_id]

    def add_point(self, point: Point) -> None:
        """
        Add a point to the network.

        Args:
            point: Point object to add

        Raises:
            ValueError: If a point with the same ID already exists
        """
        if point.id in self.points:
            raise ValueError(f"Point '{point.id}' already exists in network")
        self.points[point.id] = point

    def update_point(self, point: Point) -> None:
        """
        Update an existing point in the network.

        Args:
            point: Point object with updated values

        Raises:
            KeyError: If point does not exist
        """
        if point.id not in self.points:
            raise KeyError(f"Point '{point.id}' not found in network")
        self.points[point.id] = point

    def remove_point(self, point_id: str) -> None:
        """
        Remove a point from the network.

        Args:
            point_id: ID of point to remove

        Raises:
            KeyError: If point does not exist
        """
        if point_id not in self.points:
            raise KeyError(f"Point '{point_id}' not found in network")
        del self.points[point_id]

    def add_observation(self, obs: Observation) -> None:
        """
        Add an observation to the network.

        Args:
            obs: Observation object to add
        """
        self.observations.append(obs)

    def remove_observation(self, obs_id: str) -> None:
        """
        Remove an observation by ID.

        Args:
            obs_id: ID of observation to remove

        Raises:
            KeyError: If observation not found
        """
        for i, obs in enumerate(self.observations):
            if obs.id == obs_id:
                del self.observations[i]
                return
        raise KeyError(f"Observation '{obs_id}' not found in network")

    def get_observation(self, obs_id: str) -> Observation:
        """
        Get an observation by ID.

        Args:
            obs_id: ID of observation to retrieve

        Returns:
            The Observation object

        Raises:
            KeyError: If observation not found
        """
        for obs in self.observations:
            if obs.id == obs_id:
                return obs
        raise KeyError(f"Observation '{obs_id}' not found in network")

    def get_fixed_points(self) -> List[Point]:
        """
        Get all completely fixed points (both coordinates fixed).

        Returns:
            List of fixed Point objects
        """
        return [p for p in self.points.values() if p.is_fixed]

    def get_free_points(self) -> List[Point]:
        """
        Get all completely free points (both coordinates adjustable).

        Returns:
            List of free Point objects
        """
        return [p for p in self.points.values() if p.is_free]

    def get_partially_fixed_points(self) -> List[Point]:
        """
        Get points with mixed fixed/free coordinates.

        Returns:
            List of partially fixed Point objects
        """
        return [p for p in self.points.values() if p.is_partially_fixed]

    def get_observations_by_type(self, obs_type: ObservationType) -> List[Observation]:
        """
        Get all observations of a specific type.

        Args:
            obs_type: The ObservationType to filter by

        Returns:
            List of matching Observation objects
        """
        return [obs for obs in self.observations if obs.obs_type == obs_type]

    def get_enabled_observations(self) -> List[Observation]:
        """
        Get all enabled observations (not disabled for outlier handling).

        Returns:
            List of enabled Observation objects
        """
        return [obs for obs in self.observations if obs.enabled]

    def get_direction_sets(self) -> Dict[str, List[DirectionObservation]]:
        """
        Group direction observations by their set_id.

        Each set shares a common orientation unknown.

        Returns:
            Dictionary mapping set_id to list of DirectionObservations
        """
        sets: Dict[str, List[DirectionObservation]] = defaultdict(list)
        for obs in self.observations:
            if isinstance(obs, DirectionObservation):
                sets[obs.set_id].append(obs)
        return dict(sets)

    def get_point_ids_from_observations(self) -> Set[str]:
        """
        Extract all point IDs referenced in observations.

        Returns:
            Set of point IDs
        """
        point_ids: Set[str] = set()
        for obs in self.observations:
            if isinstance(obs, DistanceObservation):
                point_ids.add(obs.from_point_id)
                point_ids.add(obs.to_point_id)
            elif isinstance(obs, DirectionObservation):
                point_ids.add(obs.from_point_id)
                point_ids.add(obs.to_point_id)
            elif isinstance(obs, AngleObservation):
                point_ids.add(obs.at_point_id)
                point_ids.add(obs.from_point_id)
                point_ids.add(obs.to_point_id)
        return point_ids

    def validate(self) -> List[str]:
        """
        Validate the network for common errors.

        Checks for:
        - Missing points referenced in observations
        - Disconnected points (not in any observation)
        - Minimum requirements for adjustment
        - Network connectivity

        Returns:
            List of error messages (empty if valid)
        """
        errors: List[str] = []

        # Check for empty network
        if not self.points:
            errors.append("Network has no points")
            return errors

        if not self.observations:
            errors.append("Network has no observations")
            return errors

        # Check for missing points in observations
        obs_point_ids = self.get_point_ids_from_observations()
        for point_id in obs_point_ids:
            if point_id not in self.points:
                errors.append(f"Observation references missing point: '{point_id}'")

        # Check for disconnected points (defined but not in any observation)
        defined_point_ids = set(self.points.keys())
        disconnected = defined_point_ids - obs_point_ids
        for point_id in disconnected:
            errors.append(f"Point '{point_id}' is not connected to any observation")

        # Check minimum fixed points
        fixed_points = self.get_fixed_points()
        if not fixed_points:
            errors.append("Network has no fixed points (datum definition required)")

        # Check network connectivity using graph traversal
        if self.points and self.observations:
            connectivity_errors = self._check_connectivity()
            errors.extend(connectivity_errors)

        # Check for sufficient observations
        num_unknowns = self._count_unknowns()
        num_observations = len(self.get_enabled_observations())
        if num_observations < num_unknowns:
            errors.append(
                f"Insufficient observations: {num_observations} observations "
                f"for {num_unknowns} unknowns (need at least {num_unknowns})"
            )

        return errors

    def _check_connectivity(self) -> List[str]:
        """
        Check if all points are connected in the network graph.

        Returns:
            List of connectivity error messages
        """
        errors = []

        # Build adjacency list
        adjacency: Dict[str, Set[str]] = defaultdict(set)
        for obs in self.observations:
            if isinstance(obs, DistanceObservation):
                adjacency[obs.from_point_id].add(obs.to_point_id)
                adjacency[obs.to_point_id].add(obs.from_point_id)
            elif isinstance(obs, DirectionObservation):
                adjacency[obs.from_point_id].add(obs.to_point_id)
                adjacency[obs.to_point_id].add(obs.from_point_id)
            elif isinstance(obs, AngleObservation):
                adjacency[obs.at_point_id].add(obs.from_point_id)
                adjacency[obs.at_point_id].add(obs.to_point_id)
                adjacency[obs.from_point_id].add(obs.at_point_id)
                adjacency[obs.to_point_id].add(obs.at_point_id)

        # Get points that are in observations
        obs_point_ids = self.get_point_ids_from_observations()
        if not obs_point_ids:
            return errors

        # BFS to find connected components
        visited: Set[str] = set()
        start_point = next(iter(obs_point_ids))
        queue = [start_point]
        visited.add(start_point)

        while queue:
            current = queue.pop(0)
            for neighbor in adjacency.get(current, set()):
                if neighbor not in visited and neighbor in obs_point_ids:
                    visited.add(neighbor)
                    queue.append(neighbor)

        # Check if all observation points were visited
        unvisited = obs_point_ids - visited
        if unvisited:
            errors.append(
                f"Network is disconnected. Unreachable points: {', '.join(sorted(unvisited))}"
            )

        return errors

    def _count_unknowns(self) -> int:
        """
        Count the number of unknowns in the adjustment.

        Unknowns include:
        - Coordinate components of free/partially-fixed points
        - Orientation unknowns for direction sets

        Returns:
            Total number of unknowns
        """
        unknowns = 0

        # Count coordinate unknowns
        for point in self.points.values():
            if not point.fixed_easting:
                unknowns += 1
            if not point.fixed_northing:
                unknowns += 1

        # Count orientation unknowns (one per direction set)
        direction_sets = self.get_direction_sets()
        unknowns += len(direction_sets)

        return unknowns

    def get_degrees_of_freedom(self) -> int:
        """
        Calculate degrees of freedom for the adjustment.

        DOF = number of observations - number of unknowns

        Returns:
            Degrees of freedom (redundancy)
        """
        num_obs = len(self.get_enabled_observations())
        num_unknowns = self._count_unknowns()
        return num_obs - num_unknowns

    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of network contents.

        Returns:
            Dictionary with network statistics
        """
        obs_by_type = {}
        for obs_type in ObservationType:
            count = len(self.get_observations_by_type(obs_type))
            if count > 0:
                obs_by_type[obs_type.value] = count

        return {
            "name": self.name,
            "num_points": len(self.points),
            "num_fixed_points": len(self.get_fixed_points()),
            "num_free_points": len(self.get_free_points()),
            "num_observations": len(self.observations),
            "num_enabled_observations": len(self.get_enabled_observations()),
            "observations_by_type": obs_by_type,
            "num_direction_sets": len(self.get_direction_sets()),
            "num_unknowns": self._count_unknowns(),
            "degrees_of_freedom": self.get_degrees_of_freedom()
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize network to dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            "name": self.name,
            "points": {pid: p.to_dict() for pid, p in self.points.items()},
            "observations": [obs.to_dict() for obs in self.observations]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Network':
        """
        Create a Network from a dictionary.

        Args:
            data: Dictionary with network data

        Returns:
            New Network instance
        """
        network = cls(name=data.get("name", "Unnamed Network"))

        # Load points
        points_data = data.get("points", {})
        if isinstance(points_data, dict):
            for point_id, point_data in points_data.items():
                if isinstance(point_data, dict):
                    point_data["id"] = point_id
                    network.add_point(Point.from_dict(point_data))
        elif isinstance(points_data, list):
            for point_data in points_data:
                network.add_point(Point.from_dict(point_data))

        # Load observations
        for obs_data in data.get("observations", []):
            obs = Observation.from_dict(obs_data)
            network.add_observation(obs)

        return network

    def __repr__(self) -> str:
        return (
            f"Network('{self.name}', "
            f"points={len(self.points)}, "
            f"observations={len(self.observations)})"
        )
