"""
Tests for the Point class.
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from survey_adjustment.core.models.point import Point


class TestPointCreation:
    """Tests for Point creation and validation."""

    def test_create_basic_point(self):
        """Test creating a point with minimal required fields."""
        point = Point(id="A", name="Station A", easting=1000.0, northing=2000.0)

        assert point.id == "A"
        assert point.name == "Station A"
        assert point.easting == 1000.0
        assert point.northing == 2000.0
        assert point.fixed_easting is False
        assert point.fixed_northing is False
        assert point.sigma_easting is None
        assert point.sigma_northing is None

    def test_create_fixed_point(self):
        """Test creating a fully fixed control point."""
        point = Point(
            id="CTRL1",
            name="Control Point 1",
            easting=1000.0,
            northing=2000.0,
            fixed_easting=True,
            fixed_northing=True
        )

        assert point.is_fixed is True
        assert point.is_free is False
        assert point.is_partially_fixed is False

    def test_create_free_point_with_sigma(self):
        """Test creating a free point with a priori standard deviations."""
        point = Point(
            id="P1",
            name="Point 1",
            easting=1500.0,
            northing=2500.0,
            fixed_easting=False,
            fixed_northing=False,
            sigma_easting=0.010,
            sigma_northing=0.015
        )

        assert point.is_free is True
        assert point.sigma_easting == 0.010
        assert point.sigma_northing == 0.015

    def test_create_partially_fixed_point(self):
        """Test creating a point with only easting fixed."""
        point = Point(
            id="P2",
            name="Point 2",
            easting=1000.0,
            northing=2000.0,
            fixed_easting=True,
            fixed_northing=False
        )

        assert point.is_fixed is False
        assert point.is_free is False
        assert point.is_partially_fixed is True

    def test_alphanumeric_point_id(self):
        """Test that alphanumeric point IDs are allowed."""
        point = Point(id="STA-001", name="Station 001", easting=100.0, northing=200.0)
        assert point.id == "STA-001"

    def test_coordinate_conversion_to_float(self):
        """Test that coordinates are converted to float."""
        point = Point(id="A", name="Test", easting="1000", northing="2000")
        assert isinstance(point.easting, float)
        assert isinstance(point.northing, float)


class TestPointValidation:
    """Tests for Point validation."""

    def test_empty_id_raises_error(self):
        """Test that empty point ID raises ValueError."""
        with pytest.raises(ValueError, match="Point ID cannot be empty"):
            Point(id="", name="Test", easting=0, northing=0)

    def test_negative_sigma_raises_error(self):
        """Test that negative standard deviation raises ValueError."""
        with pytest.raises(ValueError, match="sigma_easting cannot be negative"):
            Point(
                id="A",
                name="Test",
                easting=0,
                northing=0,
                sigma_easting=-0.01
            )

    def test_negative_sigma_northing_raises_error(self):
        """Test that negative sigma_northing raises ValueError."""
        with pytest.raises(ValueError, match="sigma_northing cannot be negative"):
            Point(
                id="A",
                name="Test",
                easting=0,
                northing=0,
                sigma_northing=-0.01
            )


class TestPointSerialization:
    """Tests for Point serialization/deserialization."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        point = Point(
            id="A",
            name="Station A",
            easting=1000.123,
            northing=2000.456,
            fixed_easting=True,
            fixed_northing=False,
            sigma_easting=0.005,
            sigma_northing=None
        )

        data = point.to_dict()

        assert data["id"] == "A"
        assert data["name"] == "Station A"
        assert data["easting"] == 1000.123
        assert data["northing"] == 2000.456
        assert data["fixed_easting"] is True
        assert data["fixed_northing"] is False
        assert data["sigma_easting"] == 0.005
        assert data["sigma_northing"] is None

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "id": "B",
            "name": "Station B",
            "easting": 1500.0,
            "northing": 2500.0,
            "fixed_easting": False,
            "fixed_northing": True,
            "sigma_easting": 0.010,
            "sigma_northing": 0.012
        }

        point = Point.from_dict(data)

        assert point.id == "B"
        assert point.name == "Station B"
        assert point.easting == 1500.0
        assert point.northing == 2500.0
        assert point.fixed_easting is False
        assert point.fixed_northing is True
        assert point.sigma_easting == 0.010
        assert point.sigma_northing == 0.012

    def test_from_dict_csv_format(self):
        """Test deserialization from CSV-style dictionary."""
        data = {
            "point_id": "C",
            "name": "Station C",
            "easting": 2000.0,
            "northing": 3000.0,
            "fixed_e": "true",
            "fixed_n": "false",
            "sigma_e": 0.008,
            "sigma_n": ""
        }

        point = Point.from_dict(data)

        assert point.id == "C"
        assert point.fixed_easting is True
        assert point.fixed_northing is False
        assert point.sigma_easting == 0.008
        assert point.sigma_northing is None

    def test_roundtrip_serialization(self):
        """Test that to_dict -> from_dict preserves all data."""
        original = Point(
            id="ROUNDTRIP",
            name="Roundtrip Test",
            easting=12345.678,
            northing=87654.321,
            fixed_easting=True,
            fixed_northing=True,
            sigma_easting=0.001,
            sigma_northing=0.002
        )

        restored = Point.from_dict(original.to_dict())

        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.easting == original.easting
        assert restored.northing == original.northing
        assert restored.fixed_easting == original.fixed_easting
        assert restored.fixed_northing == original.fixed_northing
        assert restored.sigma_easting == original.sigma_easting
        assert restored.sigma_northing == original.sigma_northing


class TestPointProperties:
    """Tests for Point properties."""

    def test_repr(self):
        """Test string representation."""
        point = Point(id="A", name="Test", easting=1000.0, northing=2000.0)
        repr_str = repr(point)

        assert "Point(A" in repr_str
        assert "1000.000" in repr_str
        assert "2000.000" in repr_str
        assert "free" in repr_str

    def test_repr_fixed(self):
        """Test string representation of fixed point."""
        point = Point(
            id="A",
            name="Test",
            easting=1000.0,
            northing=2000.0,
            fixed_easting=True,
            fixed_northing=True
        )

        assert "fixed" in repr(point)

    def test_repr_partial(self):
        """Test string representation of partially fixed point."""
        point = Point(
            id="A",
            name="Test",
            easting=1000.0,
            northing=2000.0,
            fixed_easting=True,
            fixed_northing=False
        )

        assert "partial" in repr(point)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
