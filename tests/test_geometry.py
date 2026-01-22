"""Tests for QGIS-free geometry utilities.

These tests verify the ellipse polygon and residual vector generation
without requiring QGIS runtime.
"""

import math
import pytest

from survey_adjustment.core.geometry import (
    ellipse_polygon_points,
    distance_residual_vector,
    direction_residual_vector,
    angle_residual_vector,
)


class TestEllipsePolygonPoints:
    """Tests for ellipse_polygon_points function."""

    def test_circle_centered_at_origin(self):
        """A circle (a == b) should have all points equidistant from center."""
        vertices = ellipse_polygon_points(
            center_e=0.0,
            center_n=0.0,
            semi_major=10.0,
            semi_minor=10.0,
            orientation=0.0,
            num_vertices=36,
        )

        # Should have 37 points (36 + 1 to close)
        assert len(vertices) == 37

        # First and last should be same
        assert vertices[0] == vertices[-1]

        # All points should be ~10m from center
        for e, n in vertices[:-1]:
            dist = math.sqrt(e * e + n * n)
            assert abs(dist - 10.0) < 1e-10

    def test_ellipse_semi_axes(self):
        """Verify ellipse has correct semi-major and semi-minor extents."""
        a = 20.0  # semi-major
        b = 10.0  # semi-minor
        vertices = ellipse_polygon_points(
            center_e=100.0,
            center_n=200.0,
            semi_major=a,
            semi_minor=b,
            orientation=0.0,  # oriented north
            num_vertices=72,
        )

        # Find min/max coordinates relative to center
        eastings = [v[0] - 100.0 for v in vertices[:-1]]
        northings = [v[1] - 200.0 for v in vertices[:-1]]

        max_e = max(eastings)
        min_e = min(eastings)
        max_n = max(northings)
        min_n = min(northings)

        # With orientation=0 (north), semi-major aligns with N axis
        # Semi-major (a=20) should be in N direction
        # Semi-minor (b=10) should be in E direction
        assert abs(max_n - a) < 0.5  # Tolerance for discrete approximation
        assert abs(min_n + a) < 0.5
        assert abs(max_e - b) < 0.5
        assert abs(min_e + b) < 0.5

    def test_ellipse_rotated_90_degrees(self):
        """An ellipse rotated 90 degrees should swap major/minor axes."""
        a = 20.0
        b = 10.0
        # Orientation of 90 degrees = pi/2 radians (clockwise from north = east)
        vertices = ellipse_polygon_points(
            center_e=0.0,
            center_n=0.0,
            semi_major=a,
            semi_minor=b,
            orientation=math.pi / 2,  # pointing east
            num_vertices=72,
        )

        eastings = [v[0] for v in vertices[:-1]]
        northings = [v[1] for v in vertices[:-1]]

        max_e = max(eastings)
        max_n = max(northings)

        # Now semi-major should be in E direction
        assert abs(max_e - a) < 0.5
        assert abs(max_n - b) < 0.5

    def test_ellipse_closure(self):
        """Polygon should be closed (first == last)."""
        vertices = ellipse_polygon_points(
            center_e=50.0,
            center_n=50.0,
            semi_major=5.0,
            semi_minor=3.0,
            orientation=0.5,
            num_vertices=36,
        )

        assert vertices[0][0] == vertices[-1][0]
        assert vertices[0][1] == vertices[-1][1]

    def test_minimum_vertices(self):
        """Should handle minimum vertex count gracefully."""
        vertices = ellipse_polygon_points(
            center_e=0.0,
            center_n=0.0,
            semi_major=1.0,
            semi_minor=1.0,
            orientation=0.0,
            num_vertices=2,  # Too few, should be bumped to 4
        )

        # Should have at least 5 points (4 + 1 to close)
        assert len(vertices) >= 5


class TestDistanceResidualVector:
    """Tests for distance_residual_vector function."""

    def test_positive_residual_along_line(self):
        """Positive residual should extend in direction of line."""
        start, end = distance_residual_vector(
            from_e=0.0, from_n=0.0,
            to_e=100.0, to_n=0.0,
            residual=0.01,  # 10mm too long
            scale=1000.0,   # 1mm -> 1m display
        )

        # Midpoint is at (50, 0)
        assert start == (50.0, 0.0)

        # End should be 10m east of midpoint (0.01 * 1000 = 10)
        assert abs(end[0] - 60.0) < 1e-10
        assert abs(end[1] - 0.0) < 1e-10

    def test_negative_residual(self):
        """Negative residual should extend opposite to line direction."""
        start, end = distance_residual_vector(
            from_e=0.0, from_n=0.0,
            to_e=100.0, to_n=0.0,
            residual=-0.01,  # 10mm too short
            scale=1000.0,
        )

        # End should be west of midpoint
        assert end[0] < start[0]

    def test_diagonal_line(self):
        """Test vector on diagonal observation line."""
        start, end = distance_residual_vector(
            from_e=0.0, from_n=0.0,
            to_e=100.0, to_n=100.0,
            residual=0.01,
            scale=1000.0,
        )

        # Midpoint at (50, 50)
        assert start == (50.0, 50.0)

        # Direction is 45 degrees, so offset should be equal in E and N
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        assert abs(dx - dy) < 1e-10

    def test_coincident_points(self):
        """Degenerate case: coincident points should return same point."""
        start, end = distance_residual_vector(
            from_e=10.0, from_n=20.0,
            to_e=10.0, to_n=20.0,
            residual=0.01,
            scale=1000.0,
        )

        assert start == end


class TestDirectionResidualVector:
    """Tests for direction_residual_vector function."""

    def test_positive_residual_clockwise(self):
        """Positive residual should offset clockwise."""
        start, end = direction_residual_vector(
            station_e=0.0, station_n=0.0,
            target_e=100.0, target_n=0.0,  # Target to east
            residual=0.001,  # Small positive angle
            scale=1000.0,
        )

        # Start should be along direction to target
        # End should be offset perpendicular (south for clockwise from east)
        assert end[1] < start[1]  # Moved south (clockwise offset from east)

    def test_negative_residual_counterclockwise(self):
        """Negative residual should offset counter-clockwise."""
        start, end = direction_residual_vector(
            station_e=0.0, station_n=0.0,
            target_e=100.0, target_n=0.0,
            residual=-0.001,
            scale=1000.0,
        )

        # End should be offset north (counter-clockwise from east)
        assert end[1] > start[1]

    def test_coincident_points_returns_same(self):
        """Degenerate case with coincident points."""
        start, end = direction_residual_vector(
            station_e=10.0, station_n=20.0,
            target_e=10.0, target_n=20.0,
            residual=0.001,
            scale=1000.0,
        )

        assert start == end


class TestAngleResidualVector:
    """Tests for angle_residual_vector function."""

    def test_angle_residual_same_as_direction(self):
        """Angle residual vector should behave like direction for foresight."""
        # angle_residual_vector delegates to direction_residual_vector
        # using the at->to direction

        angle_start, angle_end = angle_residual_vector(
            at_e=0.0, at_n=0.0,
            from_e=-100.0, from_n=0.0,  # Backsight to west
            to_e=0.0, to_n=100.0,       # Foresight to north
            residual=0.001,
            scale=1000.0,
        )

        dir_start, dir_end = direction_residual_vector(
            station_e=0.0, station_n=0.0,
            target_e=0.0, target_n=100.0,
            residual=0.001,
            scale=1000.0,
        )

        # Should be identical
        assert abs(angle_start[0] - dir_start[0]) < 1e-10
        assert abs(angle_start[1] - dir_start[1]) < 1e-10
        assert abs(angle_end[0] - dir_end[0]) < 1e-10
        assert abs(angle_end[1] - dir_end[1]) < 1e-10

    def test_right_angle_residual(self):
        """Test residual at a 90-degree angle setup."""
        start, end = angle_residual_vector(
            at_e=50.0, at_n=50.0,
            from_e=0.0, from_n=50.0,    # Backsight to west
            to_e=50.0, to_n=100.0,      # Foresight to north
            residual=0.002,
            scale=1000.0,
        )

        # Vector should start along the foresight direction
        # and be offset perpendicular to it
        assert start[0] == 50.0  # Same easting as station (north direction)
        assert start[1] > 50.0   # North of station


class TestGeometryIntegration:
    """Integration tests combining multiple geometry functions."""

    def test_ellipse_at_real_coordinates(self):
        """Test ellipse generation at realistic survey coordinates."""
        # Simulating UTM-like coordinates
        center_e = 500000.0
        center_n = 4500000.0

        vertices = ellipse_polygon_points(
            center_e=center_e,
            center_n=center_n,
            semi_major=0.025,  # 25mm error ellipse
            semi_minor=0.015,  # 15mm
            orientation=math.radians(45),  # 45 degrees from north
            num_vertices=64,
        )

        # All vertices should be very close to center
        for e, n in vertices:
            de = e - center_e
            dn = n - center_n
            dist = math.sqrt(de * de + dn * dn)
            assert dist <= 0.026  # Within semi-major + tolerance

    def test_residual_vectors_scale_correctly(self):
        """Verify scale factor produces expected vector lengths."""
        residual = 0.005  # 5mm

        # Test with scale 1000 (1mm -> 1m)
        _, end1 = distance_residual_vector(
            from_e=0.0, from_n=0.0,
            to_e=100.0, to_n=0.0,
            residual=residual,
            scale=1000.0,
        )

        # Test with scale 2000 (1mm -> 2m)
        _, end2 = distance_residual_vector(
            from_e=0.0, from_n=0.0,
            to_e=100.0, to_n=0.0,
            residual=residual,
            scale=2000.0,
        )

        # Vector with scale 2000 should be twice as long
        len1 = end1[0] - 50.0  # Distance from midpoint
        len2 = end2[0] - 50.0

        assert abs(len2 / len1 - 2.0) < 1e-10
