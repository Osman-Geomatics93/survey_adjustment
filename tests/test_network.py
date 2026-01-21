"""
Tests for the Network class.
"""

import pytest
import math
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from survey_adjustment.core.models.point import Point
from survey_adjustment.core.models.observation import (
    ObservationType,
    DistanceObservation,
    DirectionObservation,
    AngleObservation
)
from survey_adjustment.core.models.network import Network


class TestNetworkCreation:
    """Tests for Network creation."""

    def test_create_empty_network(self):
        """Test creating an empty network."""
        network = Network(name="Test Network")

        assert network.name == "Test Network"
        assert len(network.points) == 0
        assert len(network.observations) == 0

    def test_add_point(self):
        """Test adding a point to the network."""
        network = Network()
        point = Point(id="A", name="Station A", easting=1000.0, northing=2000.0)

        network.add_point(point)

        assert len(network.points) == 1
        assert "A" in network.points
        assert network.get_point("A") == point

    def test_add_duplicate_point_raises_error(self):
        """Test that adding duplicate point raises error."""
        network = Network()
        point1 = Point(id="A", name="Station A", easting=1000.0, northing=2000.0)
        point2 = Point(id="A", name="Station A Copy", easting=1001.0, northing=2001.0)

        network.add_point(point1)

        with pytest.raises(ValueError, match="already exists"):
            network.add_point(point2)

    def test_get_nonexistent_point_raises_error(self):
        """Test that getting nonexistent point raises error."""
        network = Network()

        with pytest.raises(KeyError, match="not found"):
            network.get_point("X")

    def test_add_observation(self):
        """Test adding an observation to the network."""
        network = Network()
        obs = DistanceObservation(
            id="D001",
            obs_type=ObservationType.DISTANCE,
            value=100.0,
            sigma=0.01,
            from_point_id="A",
            to_point_id="B"
        )

        network.add_observation(obs)

        assert len(network.observations) == 1
        assert network.observations[0] == obs


class TestNetworkQueries:
    """Tests for Network query methods."""

    @pytest.fixture
    def sample_network(self):
        """Create a sample network for testing."""
        network = Network(name="Sample Network")

        # Add points
        network.add_point(Point(
            id="A", name="Control A", easting=1000.0, northing=2000.0,
            fixed_easting=True, fixed_northing=True
        ))
        network.add_point(Point(
            id="B", name="Station B", easting=1500.0, northing=2300.0,
            fixed_easting=False, fixed_northing=False
        ))
        network.add_point(Point(
            id="C", name="Control C", easting=2000.0, northing=2000.0,
            fixed_easting=True, fixed_northing=True
        ))
        network.add_point(Point(
            id="D", name="Station D", easting=1500.0, northing=1700.0,
            fixed_easting=True, fixed_northing=False
        ))

        # Add observations
        network.add_observation(DistanceObservation(
            id="D001", obs_type=ObservationType.DISTANCE,
            value=583.095, sigma=0.003,
            from_point_id="A", to_point_id="B"
        ))
        network.add_observation(DistanceObservation(
            id="D002", obs_type=ObservationType.DISTANCE,
            value=583.095, sigma=0.003,
            from_point_id="B", to_point_id="C"
        ))
        network.add_observation(DirectionObservation(
            id="DIR001", obs_type=ObservationType.DIRECTION,
            value=math.radians(45.0), sigma=0.00015,
            from_point_id="A", to_point_id="B", set_id="SET_A"
        ))
        network.add_observation(DirectionObservation(
            id="DIR002", obs_type=ObservationType.DIRECTION,
            value=math.radians(135.0), sigma=0.00015,
            from_point_id="A", to_point_id="C", set_id="SET_A"
        ))

        return network

    def test_get_fixed_points(self, sample_network):
        """Test getting fixed points."""
        fixed = sample_network.get_fixed_points()

        assert len(fixed) == 2
        fixed_ids = {p.id for p in fixed}
        assert "A" in fixed_ids
        assert "C" in fixed_ids

    def test_get_free_points(self, sample_network):
        """Test getting free points."""
        free = sample_network.get_free_points()

        assert len(free) == 1
        assert free[0].id == "B"

    def test_get_partially_fixed_points(self, sample_network):
        """Test getting partially fixed points."""
        partial = sample_network.get_partially_fixed_points()

        assert len(partial) == 1
        assert partial[0].id == "D"

    def test_get_observations_by_type(self, sample_network):
        """Test getting observations by type."""
        distances = sample_network.get_observations_by_type(ObservationType.DISTANCE)
        directions = sample_network.get_observations_by_type(ObservationType.DIRECTION)
        angles = sample_network.get_observations_by_type(ObservationType.ANGLE)

        assert len(distances) == 2
        assert len(directions) == 2
        assert len(angles) == 0

    def test_get_direction_sets(self, sample_network):
        """Test grouping directions by set."""
        sets = sample_network.get_direction_sets()

        assert len(sets) == 1
        assert "SET_A" in sets
        assert len(sets["SET_A"]) == 2


class TestNetworkValidation:
    """Tests for Network validation."""

    def test_validate_empty_network(self):
        """Test validation of empty network."""
        network = Network()
        errors = network.validate()

        assert len(errors) > 0
        assert any("no points" in e for e in errors)

    def test_validate_no_observations(self):
        """Test validation with no observations."""
        network = Network()
        network.add_point(Point(id="A", name="A", easting=0, northing=0))

        errors = network.validate()

        assert any("no observations" in e for e in errors)

    def test_validate_missing_point(self):
        """Test validation with missing referenced point."""
        network = Network()
        network.add_point(Point(id="A", name="A", easting=0, northing=0, fixed_easting=True, fixed_northing=True))

        # Observation references point "B" which doesn't exist
        network.add_observation(DistanceObservation(
            id="D001", obs_type=ObservationType.DISTANCE,
            value=100.0, sigma=0.01,
            from_point_id="A", to_point_id="B"
        ))

        errors = network.validate()

        assert any("missing point" in e.lower() for e in errors)

    def test_validate_no_fixed_points(self):
        """Test validation with no fixed points (no datum)."""
        network = Network()
        network.add_point(Point(id="A", name="A", easting=0, northing=0))
        network.add_point(Point(id="B", name="B", easting=100, northing=0))
        network.add_observation(DistanceObservation(
            id="D001", obs_type=ObservationType.DISTANCE,
            value=100.0, sigma=0.01,
            from_point_id="A", to_point_id="B"
        ))

        errors = network.validate()

        assert any("no fixed points" in e.lower() for e in errors)

    def test_validate_disconnected_point(self):
        """Test validation with disconnected point."""
        network = Network()
        network.add_point(Point(
            id="A", name="A", easting=0, northing=0,
            fixed_easting=True, fixed_northing=True
        ))
        network.add_point(Point(id="B", name="B", easting=100, northing=0))
        network.add_point(Point(id="C", name="C", easting=200, northing=0))  # Disconnected

        network.add_observation(DistanceObservation(
            id="D001", obs_type=ObservationType.DISTANCE,
            value=100.0, sigma=0.01,
            from_point_id="A", to_point_id="B"
        ))

        errors = network.validate()

        assert any("not connected" in e.lower() for e in errors)

    def test_validate_disconnected_network(self):
        """Test validation with disconnected network components."""
        network = Network()
        # Component 1
        network.add_point(Point(
            id="A", name="A", easting=0, northing=0,
            fixed_easting=True, fixed_northing=True
        ))
        network.add_point(Point(id="B", name="B", easting=100, northing=0))
        # Component 2 (disconnected)
        network.add_point(Point(id="C", name="C", easting=500, northing=500))
        network.add_point(Point(id="D", name="D", easting=600, northing=500))

        network.add_observation(DistanceObservation(
            id="D001", obs_type=ObservationType.DISTANCE,
            value=100.0, sigma=0.01,
            from_point_id="A", to_point_id="B"
        ))
        network.add_observation(DistanceObservation(
            id="D002", obs_type=ObservationType.DISTANCE,
            value=100.0, sigma=0.01,
            from_point_id="C", to_point_id="D"
        ))

        errors = network.validate()

        assert any("disconnected" in e.lower() or "unreachable" in e.lower() for e in errors)

    def test_validate_valid_network(self):
        """Test validation of a valid network."""
        network = Network()
        network.add_point(Point(
            id="A", name="A", easting=0, northing=0,
            fixed_easting=True, fixed_northing=True
        ))
        network.add_point(Point(
            id="B", name="B", easting=0, northing=100,
            fixed_easting=True, fixed_northing=True
        ))
        network.add_point(Point(id="C", name="C", easting=100, northing=50))

        network.add_observation(DistanceObservation(
            id="D001", obs_type=ObservationType.DISTANCE,
            value=111.8, sigma=0.01,
            from_point_id="A", to_point_id="C"
        ))
        network.add_observation(DistanceObservation(
            id="D002", obs_type=ObservationType.DISTANCE,
            value=111.8, sigma=0.01,
            from_point_id="B", to_point_id="C"
        ))
        network.add_observation(DistanceObservation(
            id="D003", obs_type=ObservationType.DISTANCE,
            value=100.0, sigma=0.01,
            from_point_id="A", to_point_id="B"
        ))

        errors = network.validate()

        assert len(errors) == 0


class TestNetworkStatistics:
    """Tests for Network statistics methods."""

    @pytest.fixture
    def valid_network(self):
        """Create a valid network for testing statistics."""
        network = Network(name="Stats Test")

        # 2 fixed points, 1 free point
        network.add_point(Point(
            id="A", name="A", easting=0, northing=0,
            fixed_easting=True, fixed_northing=True
        ))
        network.add_point(Point(
            id="B", name="B", easting=100, northing=0,
            fixed_easting=True, fixed_northing=True
        ))
        network.add_point(Point(id="C", name="C", easting=50, northing=50))

        # 3 distance observations
        network.add_observation(DistanceObservation(
            id="D001", obs_type=ObservationType.DISTANCE,
            value=70.7, sigma=0.01,
            from_point_id="A", to_point_id="C"
        ))
        network.add_observation(DistanceObservation(
            id="D002", obs_type=ObservationType.DISTANCE,
            value=70.7, sigma=0.01,
            from_point_id="B", to_point_id="C"
        ))
        network.add_observation(DistanceObservation(
            id="D003", obs_type=ObservationType.DISTANCE,
            value=100.0, sigma=0.01,
            from_point_id="A", to_point_id="B"
        ))

        return network

    def test_count_unknowns(self, valid_network):
        """Test counting unknowns."""
        # 1 free point = 2 unknowns (E and N)
        unknowns = valid_network._count_unknowns()
        assert unknowns == 2

    def test_degrees_of_freedom(self, valid_network):
        """Test degrees of freedom calculation."""
        # 3 observations - 2 unknowns = 1 DOF
        dof = valid_network.get_degrees_of_freedom()
        assert dof == 1

    def test_summary(self, valid_network):
        """Test network summary."""
        summary = valid_network.summary()

        assert summary["name"] == "Stats Test"
        assert summary["num_points"] == 3
        assert summary["num_fixed_points"] == 2
        assert summary["num_free_points"] == 1
        assert summary["num_observations"] == 3
        assert summary["degrees_of_freedom"] == 1

    def test_summary_with_directions(self):
        """Test summary with direction observations (orientation unknowns)."""
        network = Network()
        network.add_point(Point(
            id="A", name="A", easting=0, northing=0,
            fixed_easting=True, fixed_northing=True
        ))
        network.add_point(Point(id="B", name="B", easting=100, northing=0))
        network.add_point(Point(id="C", name="C", easting=0, northing=100))

        # Add direction set from A
        network.add_observation(DirectionObservation(
            id="DIR001", obs_type=ObservationType.DIRECTION,
            value=math.radians(0), sigma=0.0001,
            from_point_id="A", to_point_id="B", set_id="SET_A"
        ))
        network.add_observation(DirectionObservation(
            id="DIR002", obs_type=ObservationType.DIRECTION,
            value=math.radians(90), sigma=0.0001,
            from_point_id="A", to_point_id="C", set_id="SET_A"
        ))

        summary = network.summary()

        # 2 free point coords (B: 2, C: 2) + 1 orientation unknown = 5 unknowns
        # But only B is free (C has 2 coord unknowns)
        # Actually: B has 2 unknowns, C has 2 unknowns = 4 coord unknowns + 1 orientation = 5
        assert summary["num_direction_sets"] == 1


class TestNetworkSerialization:
    """Tests for Network serialization/deserialization."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        network = Network(name="Test")
        network.add_point(Point(id="A", name="Station A", easting=100, northing=200))
        network.add_observation(DistanceObservation(
            id="D001", obs_type=ObservationType.DISTANCE,
            value=100.0, sigma=0.01,
            from_point_id="A", to_point_id="B"
        ))

        data = network.to_dict()

        assert data["name"] == "Test"
        assert "A" in data["points"]
        assert len(data["observations"]) == 1

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "name": "Loaded Network",
            "points": {
                "A": {"id": "A", "name": "A", "easting": 100, "northing": 200},
                "B": {"id": "B", "name": "B", "easting": 200, "northing": 300}
            },
            "observations": [
                {
                    "obs_type": "distance",
                    "id": "D001",
                    "from_point_id": "A",
                    "to_point_id": "B",
                    "value": 141.42,
                    "sigma": 0.01
                }
            ]
        }

        network = Network.from_dict(data)

        assert network.name == "Loaded Network"
        assert len(network.points) == 2
        assert len(network.observations) == 1
        assert network.get_point("A").easting == 100

    def test_from_dict_list_format(self):
        """Test deserialization from dictionary with list points."""
        data = {
            "name": "List Format",
            "points": [
                {"id": "A", "name": "A", "easting": 100, "northing": 200},
                {"id": "B", "name": "B", "easting": 200, "northing": 300}
            ],
            "observations": []
        }

        network = Network.from_dict(data)

        assert len(network.points) == 2

    def test_roundtrip_serialization(self):
        """Test roundtrip serialization."""
        original = Network(name="Roundtrip Test")
        original.add_point(Point(
            id="A", name="A", easting=100, northing=200,
            fixed_easting=True, fixed_northing=True
        ))
        original.add_point(Point(id="B", name="B", easting=200, northing=300))
        original.add_observation(DistanceObservation(
            id="D001", obs_type=ObservationType.DISTANCE,
            value=141.42, sigma=0.01,
            from_point_id="A", to_point_id="B"
        ))

        restored = Network.from_dict(original.to_dict())

        assert restored.name == original.name
        assert len(restored.points) == len(original.points)
        assert len(restored.observations) == len(original.observations)
        assert restored.get_point("A").fixed_easting == True


class TestNetworkModification:
    """Tests for Network modification methods."""

    def test_update_point(self):
        """Test updating an existing point."""
        network = Network()
        network.add_point(Point(id="A", name="A", easting=100, northing=200))

        updated = Point(id="A", name="A Updated", easting=101, northing=201)
        network.update_point(updated)

        assert network.get_point("A").name == "A Updated"
        assert network.get_point("A").easting == 101

    def test_update_nonexistent_point_raises_error(self):
        """Test that updating nonexistent point raises error."""
        network = Network()

        with pytest.raises(KeyError):
            network.update_point(Point(id="X", name="X", easting=0, northing=0))

    def test_remove_point(self):
        """Test removing a point."""
        network = Network()
        network.add_point(Point(id="A", name="A", easting=100, northing=200))

        network.remove_point("A")

        assert len(network.points) == 0

    def test_remove_nonexistent_point_raises_error(self):
        """Test that removing nonexistent point raises error."""
        network = Network()

        with pytest.raises(KeyError):
            network.remove_point("X")

    def test_remove_observation(self):
        """Test removing an observation."""
        network = Network()
        network.add_observation(DistanceObservation(
            id="D001", obs_type=ObservationType.DISTANCE,
            value=100.0, sigma=0.01,
            from_point_id="A", to_point_id="B"
        ))

        network.remove_observation("D001")

        assert len(network.observations) == 0

    def test_remove_nonexistent_observation_raises_error(self):
        """Test that removing nonexistent observation raises error."""
        network = Network()

        with pytest.raises(KeyError):
            network.remove_observation("X001")

    def test_get_observation(self):
        """Test getting an observation by ID."""
        network = Network()
        obs = DistanceObservation(
            id="D001", obs_type=ObservationType.DISTANCE,
            value=100.0, sigma=0.01,
            from_point_id="A", to_point_id="B"
        )
        network.add_observation(obs)

        retrieved = network.get_observation("D001")

        assert retrieved == obs

    def test_get_enabled_observations(self):
        """Test getting only enabled observations."""
        network = Network()
        network.add_observation(DistanceObservation(
            id="D001", obs_type=ObservationType.DISTANCE,
            value=100.0, sigma=0.01,
            from_point_id="A", to_point_id="B",
            enabled=True
        ))
        network.add_observation(DistanceObservation(
            id="D002", obs_type=ObservationType.DISTANCE,
            value=200.0, sigma=0.01,
            from_point_id="B", to_point_id="C",
            enabled=False
        ))

        enabled = network.get_enabled_observations()

        assert len(enabled) == 1
        assert enabled[0].id == "D001"


class TestNetworkRepr:
    """Tests for Network string representation."""

    def test_repr(self):
        """Test network repr."""
        network = Network(name="Test Network")
        network.add_point(Point(id="A", name="A", easting=0, northing=0))
        network.add_observation(DistanceObservation(
            id="D001", obs_type=ObservationType.DISTANCE,
            value=100.0, sigma=0.01,
            from_point_id="A", to_point_id="B"
        ))

        repr_str = repr(network)

        assert "Test Network" in repr_str
        assert "points=1" in repr_str
        assert "observations=1" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
