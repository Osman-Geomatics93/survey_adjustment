"""
Tests for the Observation classes.
"""

import pytest
import math
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from survey_adjustment.core.models.observation import (
    Observation,
    ObservationType,
    DistanceObservation,
    DirectionObservation,
    AngleObservation,
    degrees_to_radians,
    radians_to_degrees,
    arcseconds_to_radians,
    radians_to_arcseconds,
    dms_to_degrees,
    degrees_to_dms
)


class TestDistanceObservation:
    """Tests for DistanceObservation class."""

    def test_create_distance_observation(self):
        """Test creating a distance observation."""
        obs = DistanceObservation(
            id="D001",
            obs_type=ObservationType.DISTANCE,
            value=583.095,
            sigma=0.003,
            from_point_id="A",
            to_point_id="B"
        )

        assert obs.id == "D001"
        assert obs.obs_type == ObservationType.DISTANCE
        assert obs.value == 583.095
        assert obs.sigma == 0.003
        assert obs.from_point_id == "A"
        assert obs.to_point_id == "B"
        assert obs.enabled is True

    def test_distance_weight(self):
        """Test weight calculation from sigma."""
        obs = DistanceObservation(
            id="D001",
            obs_type=ObservationType.DISTANCE,
            value=100.0,
            sigma=0.01,  # 1 cm
            from_point_id="A",
            to_point_id="B"
        )

        expected_weight = 1.0 / (0.01 ** 2)  # 10000
        assert obs.weight == pytest.approx(expected_weight)

    def test_distance_validation_same_points(self):
        """Test that same from/to points raises error."""
        with pytest.raises(ValueError, match="cannot be the same"):
            DistanceObservation(
                id="D001",
                obs_type=ObservationType.DISTANCE,
                value=100.0,
                sigma=0.01,
                from_point_id="A",
                to_point_id="A"
            )

    def test_distance_validation_negative_value(self):
        """Test that negative distance raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            DistanceObservation(
                id="D001",
                obs_type=ObservationType.DISTANCE,
                value=-100.0,
                sigma=0.01,
                from_point_id="A",
                to_point_id="B"
            )

    def test_distance_validation_zero_sigma(self):
        """Test that zero sigma raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            DistanceObservation(
                id="D001",
                obs_type=ObservationType.DISTANCE,
                value=100.0,
                sigma=0.0,
                from_point_id="A",
                to_point_id="B"
            )

    def test_distance_from_dict(self):
        """Test creating distance observation from dictionary."""
        data = {
            "obs_id": "D001",
            "obs_type": "distance",
            "from_point": "A",
            "to_point": "B",
            "distance": 583.095,
            "sigma_distance": 0.003
        }

        obs = DistanceObservation.from_dict(data)

        assert obs.id == "D001"
        assert obs.from_point_id == "A"
        assert obs.to_point_id == "B"
        assert obs.value == 583.095
        assert obs.sigma == 0.003

    def test_distance_to_dict(self):
        """Test serialization to dictionary."""
        obs = DistanceObservation(
            id="D001",
            obs_type=ObservationType.DISTANCE,
            value=583.095,
            sigma=0.003,
            from_point_id="A",
            to_point_id="B"
        )

        data = obs.to_dict()

        assert data["id"] == "D001"
        assert data["obs_type"] == "distance"
        assert data["from_point_id"] == "A"
        assert data["to_point_id"] == "B"
        assert data["value"] == 583.095
        assert data["sigma"] == 0.003


class TestDirectionObservation:
    """Tests for DirectionObservation class."""

    def test_create_direction_observation(self):
        """Test creating a direction observation."""
        obs = DirectionObservation(
            id="DIR001",
            obs_type=ObservationType.DIRECTION,
            value=math.radians(45.0),
            sigma=0.00015,
            from_point_id="A",
            to_point_id="B",
            set_id="SET1"
        )

        assert obs.id == "DIR001"
        assert obs.obs_type == ObservationType.DIRECTION
        assert obs.value == pytest.approx(math.radians(45.0))
        assert obs.from_point_id == "A"
        assert obs.to_point_id == "B"
        assert obs.set_id == "SET1"

    def test_direction_value_degrees_property(self):
        """Test value_degrees property."""
        obs = DirectionObservation(
            id="DIR001",
            obs_type=ObservationType.DIRECTION,
            value=math.radians(90.0),
            sigma=0.00015,
            from_point_id="A",
            to_point_id="B",
            set_id="SET1"
        )

        assert obs.value_degrees == pytest.approx(90.0)

    def test_direction_default_set_id(self):
        """Test that default set_id is generated from from_point."""
        obs = DirectionObservation(
            id="DIR001",
            obs_type=ObservationType.DIRECTION,
            value=math.radians(45.0),
            sigma=0.00015,
            from_point_id="A",
            to_point_id="B",
            set_id=""
        )

        assert obs.set_id == "SET_A"

    def test_direction_from_dict_degrees(self):
        """Test creating direction from dictionary with degrees."""
        data = {
            "obs_id": "DIR001",
            "from_point": "A",
            "to_point": "B",
            "direction": 45.5,
            "sigma_direction": 0.00015,
            "set_id": "SET1"
        }

        obs = DirectionObservation.from_dict_degrees(data)

        assert obs.value == pytest.approx(math.radians(45.5))
        assert obs.value_degrees == pytest.approx(45.5)

    def test_direction_from_dict_arcseconds(self):
        """Test creating direction from dictionary with arc-seconds sigma."""
        data = {
            "obs_id": "DIR001",
            "from_point": "A",
            "to_point": "B",
            "direction": 45.0,
            "sigma_direction": 3.0,  # 3 arc-seconds
            "set_id": "SET1"
        }

        obs = DirectionObservation.from_dict_degrees(data, sigma_in_arcseconds=True)

        # 3 arc-seconds in radians
        expected_sigma = 3.0 * math.pi / (180.0 * 3600.0)
        assert obs.sigma == pytest.approx(expected_sigma)

    def test_direction_to_dict(self):
        """Test serialization to dictionary."""
        obs = DirectionObservation(
            id="DIR001",
            obs_type=ObservationType.DIRECTION,
            value=math.radians(45.0),
            sigma=0.00015,
            from_point_id="A",
            to_point_id="B",
            set_id="SET1"
        )

        data = obs.to_dict()

        assert data["id"] == "DIR001"
        assert data["obs_type"] == "direction"
        assert data["set_id"] == "SET1"


class TestAngleObservation:
    """Tests for AngleObservation class."""

    def test_create_angle_observation(self):
        """Test creating an angle observation."""
        obs = AngleObservation(
            id="ANG001",
            obs_type=ObservationType.ANGLE,
            value=math.radians(87.5432),
            sigma=0.00015,
            at_point_id="B",
            from_point_id="A",
            to_point_id="C"
        )

        assert obs.id == "ANG001"
        assert obs.obs_type == ObservationType.ANGLE
        assert obs.at_point_id == "B"
        assert obs.from_point_id == "A"
        assert obs.to_point_id == "C"

    def test_angle_value_degrees_property(self):
        """Test value_degrees property."""
        obs = AngleObservation(
            id="ANG001",
            obs_type=ObservationType.ANGLE,
            value=math.radians(90.0),
            sigma=0.00015,
            at_point_id="B",
            from_point_id="A",
            to_point_id="C"
        )

        assert obs.value_degrees == pytest.approx(90.0)

    def test_angle_validation_duplicate_points(self):
        """Test that duplicate points raises error."""
        with pytest.raises(ValueError, match="must all be different"):
            AngleObservation(
                id="ANG001",
                obs_type=ObservationType.ANGLE,
                value=math.radians(90.0),
                sigma=0.00015,
                at_point_id="A",
                from_point_id="A",
                to_point_id="C"
            )

    def test_angle_from_dict_degrees(self):
        """Test creating angle from dictionary with degrees."""
        data = {
            "obs_id": "ANG001",
            "at_point": "B",
            "from_point": "A",
            "to_point": "C",
            "angle": 87.5432,
            "sigma_angle": 0.00015
        }

        obs = AngleObservation.from_dict_degrees(data)

        assert obs.value == pytest.approx(math.radians(87.5432))
        assert obs.value_degrees == pytest.approx(87.5432)

    def test_angle_to_dict(self):
        """Test serialization to dictionary."""
        obs = AngleObservation(
            id="ANG001",
            obs_type=ObservationType.ANGLE,
            value=math.radians(90.0),
            sigma=0.00015,
            at_point_id="B",
            from_point_id="A",
            to_point_id="C"
        )

        data = obs.to_dict()

        assert data["id"] == "ANG001"
        assert data["obs_type"] == "angle"
        assert data["at_point_id"] == "B"
        assert data["from_point_id"] == "A"
        assert data["to_point_id"] == "C"


class TestObservationFactory:
    """Tests for Observation.from_dict factory method."""

    def test_factory_distance(self):
        """Test factory creates DistanceObservation."""
        data = {
            "obs_type": "distance",
            "obs_id": "D001",
            "from_point": "A",
            "to_point": "B",
            "distance": 100.0,
            "sigma_distance": 0.01
        }

        obs = Observation.from_dict(data)

        assert isinstance(obs, DistanceObservation)

    def test_factory_direction(self):
        """Test factory creates DirectionObservation."""
        data = {
            "obs_type": "direction",
            "obs_id": "DIR001",
            "from_point": "A",
            "to_point": "B",
            "direction": 45.0,
            "sigma_direction": 0.00015,
            "set_id": "SET1"
        }

        obs = Observation.from_dict(data)

        assert isinstance(obs, DirectionObservation)

    def test_factory_angle(self):
        """Test factory creates AngleObservation."""
        data = {
            "obs_type": "angle",
            "obs_id": "ANG001",
            "at_point": "B",
            "from_point": "A",
            "to_point": "C",
            "angle": 90.0,
            "sigma_angle": 0.00015
        }

        obs = Observation.from_dict(data)

        assert isinstance(obs, AngleObservation)

    def test_factory_unknown_type(self):
        """Test factory raises error for unknown type."""
        data = {
            "obs_type": "unknown",
            "obs_id": "X001"
        }

        with pytest.raises(ValueError, match="Unknown observation type"):
            Observation.from_dict(data)


class TestConversionFunctions:
    """Tests for angle conversion utility functions."""

    def test_degrees_to_radians(self):
        """Test degrees to radians conversion."""
        assert degrees_to_radians(180.0) == pytest.approx(math.pi)
        assert degrees_to_radians(90.0) == pytest.approx(math.pi / 2)
        assert degrees_to_radians(0.0) == pytest.approx(0.0)

    def test_radians_to_degrees(self):
        """Test radians to degrees conversion."""
        assert radians_to_degrees(math.pi) == pytest.approx(180.0)
        assert radians_to_degrees(math.pi / 2) == pytest.approx(90.0)
        assert radians_to_degrees(0.0) == pytest.approx(0.0)

    def test_arcseconds_to_radians(self):
        """Test arc-seconds to radians conversion."""
        # 1 degree = 3600 arc-seconds
        one_degree_arcsec = 3600.0
        assert arcseconds_to_radians(one_degree_arcsec) == pytest.approx(math.radians(1.0))

    def test_radians_to_arcseconds(self):
        """Test radians to arc-seconds conversion."""
        one_degree_rad = math.radians(1.0)
        assert radians_to_arcseconds(one_degree_rad) == pytest.approx(3600.0)

    def test_dms_to_degrees(self):
        """Test DMS to decimal degrees conversion."""
        # 45 30' 36" = 45.51 degrees
        assert dms_to_degrees(45, 30, 36) == pytest.approx(45.51)
        # 90 0' 0"
        assert dms_to_degrees(90, 0, 0) == pytest.approx(90.0)
        # Negative angle
        assert dms_to_degrees(-45, 30, 0) == pytest.approx(-45.5)

    def test_degrees_to_dms(self):
        """Test decimal degrees to DMS conversion."""
        d, m, s = degrees_to_dms(45.51)
        assert d == 45
        assert m == 30
        assert s == pytest.approx(36.0, abs=0.01)

    def test_roundtrip_dms_conversion(self):
        """Test roundtrip DMS conversion."""
        original = 123.456789
        d, m, s = degrees_to_dms(original)
        restored = dms_to_degrees(d, m, s)
        assert restored == pytest.approx(original, abs=0.0001)


class TestObservationRepr:
    """Tests for observation string representations."""

    def test_distance_repr(self):
        """Test distance observation repr."""
        obs = DistanceObservation(
            id="D001",
            obs_type=ObservationType.DISTANCE,
            value=100.0,
            sigma=0.01,
            from_point_id="A",
            to_point_id="B"
        )

        repr_str = repr(obs)
        assert "D001" in repr_str
        assert "A" in repr_str
        assert "B" in repr_str
        assert "100.000" in repr_str

    def test_direction_repr(self):
        """Test direction observation repr."""
        obs = DirectionObservation(
            id="DIR001",
            obs_type=ObservationType.DIRECTION,
            value=math.radians(45.0),
            sigma=0.00015,
            from_point_id="A",
            to_point_id="B",
            set_id="SET1"
        )

        repr_str = repr(obs)
        assert "DIR001" in repr_str
        assert "45.0000" in repr_str

    def test_angle_repr(self):
        """Test angle observation repr."""
        obs = AngleObservation(
            id="ANG001",
            obs_type=ObservationType.ANGLE,
            value=math.radians(90.0),
            sigma=0.00015,
            at_point_id="B",
            from_point_id="A",
            to_point_id="C"
        )

        repr_str = repr(obs)
        assert "ANG001" in repr_str
        assert "A-B-C" in repr_str
        assert "90.0000" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
