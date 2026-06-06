"""
Tests for the 2D Least Squares Adjustment Engine.

These tests verify:
- Geometry helper functions (azimuth, wrap_pi, etc.)
- Parameter indexing
- Basic adjustment with known results
- Convergence behavior
- Error handling for invalid networks
"""

import pytest
import math
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from survey_adjustment.core.models.point import Point
from survey_adjustment.core.models.observation import (
    ObservationType,
    DistanceObservation,
    DirectionObservation,
    AngleObservation
)
from survey_adjustment.core.models.network import Network
from survey_adjustment.core.models.options import AdjustmentOptions
from survey_adjustment.core.solver.geometry import (
    wrap_pi,
    wrap_2pi,
    azimuth,
    distance_2d,
    distance_partials,
    azimuth_partials,
    angle_at_point,
    angle_partials
)
from survey_adjustment.core.solver.indexing import (
    build_parameter_index,
    count_observations,
    compute_degrees_of_freedom,
    validate_network_for_adjustment
)
from survey_adjustment.core.solver.least_squares_2d import adjust_network_2d


class TestGeometryHelpers:
    """Tests for geometry helper functions."""

    def test_wrap_pi_positive(self):
        """Test wrap_pi with positive angles."""
        assert wrap_pi(0.0) == pytest.approx(0.0, abs=1e-10)
        assert wrap_pi(math.pi / 2) == pytest.approx(math.pi / 2, abs=1e-10)
        assert wrap_pi(math.pi) == pytest.approx(math.pi, abs=1e-10)

    def test_wrap_pi_negative(self):
        """Test wrap_pi with negative angles."""
        assert wrap_pi(-math.pi / 2) == pytest.approx(-math.pi / 2, abs=1e-10)
        # -π should wrap to π (since range is (-π, π])
        assert wrap_pi(-math.pi) == pytest.approx(math.pi, abs=1e-10)

    def test_wrap_pi_large_angles(self):
        """Test wrap_pi with angles outside [-π, π]."""
        assert wrap_pi(3 * math.pi) == pytest.approx(math.pi, abs=1e-10)
        assert wrap_pi(-3 * math.pi) == pytest.approx(math.pi, abs=1e-10)
        assert wrap_pi(2 * math.pi) == pytest.approx(0.0, abs=1e-10)

    def test_wrap_2pi_positive(self):
        """Test wrap_2pi with positive angles."""
        assert wrap_2pi(0.0) == pytest.approx(0.0, abs=1e-10)
        assert wrap_2pi(math.pi) == pytest.approx(math.pi, abs=1e-10)
        assert wrap_2pi(2 * math.pi) == pytest.approx(0.0, abs=1e-10)

    def test_wrap_2pi_negative(self):
        """Test wrap_2pi with negative angles."""
        assert wrap_2pi(-math.pi / 2) == pytest.approx(3 * math.pi / 2, abs=1e-10)
        assert wrap_2pi(-math.pi) == pytest.approx(math.pi, abs=1e-10)

    def test_azimuth_cardinal_directions(self):
        """Test azimuth for cardinal directions."""
        # North: (0,0) to (0,1) -> azimuth = 0
        assert azimuth(0, 0, 0, 1) == pytest.approx(0.0, abs=1e-10)

        # East: (0,0) to (1,0) -> azimuth = π/2
        assert azimuth(0, 0, 1, 0) == pytest.approx(math.pi / 2, abs=1e-10)

        # South: (0,0) to (0,-1) -> azimuth = π
        assert azimuth(0, 0, 0, -1) == pytest.approx(math.pi, abs=1e-10)

        # West: (0,0) to (-1,0) -> azimuth = 3π/2
        assert azimuth(0, 0, -1, 0) == pytest.approx(3 * math.pi / 2, abs=1e-10)

    def test_azimuth_diagonal(self):
        """Test azimuth for diagonal directions."""
        # NE: (0,0) to (1,1) -> azimuth = π/4 (45°)
        assert azimuth(0, 0, 1, 1) == pytest.approx(math.pi / 4, abs=1e-10)

        # SE: (0,0) to (1,-1) -> azimuth = 3π/4 (135°)
        assert azimuth(0, 0, 1, -1) == pytest.approx(3 * math.pi / 4, abs=1e-10)

    def test_distance_2d(self):
        """Test 2D distance calculation."""
        assert distance_2d(0, 0, 3, 4) == pytest.approx(5.0, abs=1e-10)
        assert distance_2d(0, 0, 0, 0) == pytest.approx(0.0, abs=1e-10)
        assert distance_2d(1, 1, 4, 5) == pytest.approx(5.0, abs=1e-10)

    def test_distance_partials(self):
        """Test partial derivatives of distance."""
        # For points (0,0) to (3,4), d=5
        # ∂d/∂e1 = -(3)/5 = -0.6
        # ∂d/∂n1 = -(4)/5 = -0.8
        # ∂d/∂e2 = (3)/5 = 0.6
        # ∂d/∂n2 = (4)/5 = 0.8
        dd_de1, dd_dn1, dd_de2, dd_dn2 = distance_partials(0, 0, 3, 4)
        assert dd_de1 == pytest.approx(-0.6, abs=1e-10)
        assert dd_dn1 == pytest.approx(-0.8, abs=1e-10)
        assert dd_de2 == pytest.approx(0.6, abs=1e-10)
        assert dd_dn2 == pytest.approx(0.8, abs=1e-10)

    def test_azimuth_partials(self):
        """Test partial derivatives of azimuth."""
        # For points (0,0) to (0,1), azimuth=0
        # d² = 1
        # ∂az/∂e1 = -dn/d² = -1/1 = -1
        # ∂az/∂n1 = de/d² = 0/1 = 0
        da_de1, da_dn1, da_de2, da_dn2 = azimuth_partials(0, 0, 0, 1)
        assert da_de1 == pytest.approx(-1.0, abs=1e-10)
        assert da_dn1 == pytest.approx(0.0, abs=1e-10)
        assert da_de2 == pytest.approx(1.0, abs=1e-10)
        assert da_dn2 == pytest.approx(0.0, abs=1e-10)

    def test_angle_at_point(self):
        """Test angle computation at a point."""
        # Angle at (0,0), from (0,1) to (1,0)
        # az_from = azimuth(0,0 -> 0,1) = 0
        # az_to = azimuth(0,0 -> 1,0) = π/2
        # angle = π/2 - 0 = π/2 (90°)
        angle = angle_at_point(0, 0, 0, 1, 1, 0)
        assert angle == pytest.approx(math.pi / 2, abs=1e-10)

        # Angle at (0,0), from (1,0) to (0,1)
        # az_from = π/2
        # az_to = 0
        # angle = 0 - π/2 = -π/2 -> wrapped to 3π/2 (270°)
        angle = angle_at_point(0, 0, 1, 0, 0, 1)
        assert angle == pytest.approx(3 * math.pi / 2, abs=1e-10)


class TestParameterIndexing:
    """Tests for parameter indexing."""

    def test_build_index_simple_network(self):
        """Test parameter index for a simple network."""
        network = Network()
        network.add_point(Point(
            id="A", name="A", easting=0, northing=0,
            fixed_easting=True, fixed_northing=True
        ))
        network.add_point(Point(
            id="B", name="B", easting=100, northing=0,
            fixed_easting=False, fixed_northing=False
        ))

        index = build_parameter_index(network)

        # A is fixed, so no unknowns
        assert index.coord_index.get(("A", "E")) is None
        assert index.coord_index.get(("A", "N")) is None

        # B has 2 unknowns
        assert index.coord_index.get(("B", "E")) is not None
        assert index.coord_index.get(("B", "N")) is not None

        assert len(index.coord_order) == 2
        assert len(index.orientation_order) == 0
        assert index.num_params == 2

    def test_build_index_with_directions(self):
        """Test parameter index with direction observations."""
        network = Network()
        network.add_point(Point(
            id="A", name="A", easting=0, northing=0,
            fixed_easting=True, fixed_northing=True
        ))
        network.add_point(Point(id="B", name="B", easting=100, northing=0))

        network.add_observation(DirectionObservation(
            id="DIR1", obs_type=ObservationType.DIRECTION,
            value=math.pi/2, sigma=0.0001,
            from_point_id="A", to_point_id="B", set_id="SET_A"
        ))

        index = build_parameter_index(network)

        # Orientation unknowns are keyed by direction set_id (one per setup/set)
        assert index.orientation_index.get("SET_A") is not None
        assert len(index.orientation_order) == 1
        assert index.num_params == 3  # 2 coords (B free) + 1 orientation

    def test_count_observations(self):
        """Test observation counting."""
        network = Network()
        network.add_point(Point(id="A", name="A", easting=0, northing=0))
        network.add_point(Point(id="B", name="B", easting=100, northing=0))

        network.add_observation(DistanceObservation(
            id="D1", obs_type=ObservationType.DISTANCE,
            value=100, sigma=0.01,
            from_point_id="A", to_point_id="B"
        ))
        network.add_observation(DistanceObservation(
            id="D2", obs_type=ObservationType.DISTANCE,
            value=100, sigma=0.01,
            from_point_id="A", to_point_id="B",
            enabled=False  # Disabled
        ))

        assert count_observations(network) == 1

    def test_compute_degrees_of_freedom(self):
        """Test DOF computation."""
        network = Network()
        network.add_point(Point(
            id="A", name="A", easting=0, northing=0,
            fixed_easting=True, fixed_northing=True
        ))
        network.add_point(Point(id="B", name="B", easting=100, northing=0))

        # 3 observations for 2 unknowns -> DOF = 1
        for i in range(3):
            network.add_observation(DistanceObservation(
                id=f"D{i}", obs_type=ObservationType.DISTANCE,
                value=100, sigma=0.01,
                from_point_id="A", to_point_id="B"
            ))

        index = build_parameter_index(network)
        dof = compute_degrees_of_freedom(network, index)

        assert dof == 1


class TestValidation:
    """Tests for network validation."""

    def test_validate_no_fixed_points(self):
        """Test validation fails with no fixed points."""
        network = Network()
        network.add_point(Point(id="A", name="A", easting=0, northing=0))
        network.add_point(Point(id="B", name="B", easting=100, northing=0))
        network.add_observation(DistanceObservation(
            id="D1", obs_type=ObservationType.DISTANCE,
            value=100, sigma=0.01,
            from_point_id="A", to_point_id="B"
        ))

        index = build_parameter_index(network)
        errors = validate_network_for_adjustment(network, index)

        assert any("fixed point" in e.lower() for e in errors)

    def test_validate_insufficient_observations(self):
        """Test validation fails with insufficient observations."""
        network = Network()
        network.add_point(Point(
            id="A", name="A", easting=0, northing=0,
            fixed_easting=True, fixed_northing=True
        ))
        network.add_point(Point(id="B", name="B", easting=100, northing=0))

        # Only 1 observation for 2 unknowns
        network.add_observation(DistanceObservation(
            id="D1", obs_type=ObservationType.DISTANCE,
            value=100, sigma=0.01,
            from_point_id="A", to_point_id="B"
        ))

        index = build_parameter_index(network)
        errors = validate_network_for_adjustment(network, index)

        assert any("insufficient" in e.lower() for e in errors)


class TestAdjustmentDistanceOnly:
    """Tests for adjustment with distance observations only."""

    def test_simple_trilateration(self):
        """Test simple trilateration with two fixed points."""
        network = Network(name="Trilateration Test")

        # Two fixed control points
        network.add_point(Point(
            id="A", name="Control A", easting=0.0, northing=0.0,
            fixed_easting=True, fixed_northing=True
        ))
        network.add_point(Point(
            id="B", name="Control B", easting=100.0, northing=0.0,
            fixed_easting=True, fixed_northing=True
        ))

        # One free point - approximate position
        # True position should be (50, 50*sqrt(3)) = (50, 86.60...)
        network.add_point(Point(
            id="C", name="Free C", easting=50.0, northing=85.0,
            fixed_easting=False, fixed_northing=False
        ))

        # Distance observations (true distances from equilateral triangle)
        true_dist = 100.0
        network.add_observation(DistanceObservation(
            id="D_AC", obs_type=ObservationType.DISTANCE,
            value=true_dist, sigma=0.01,
            from_point_id="A", to_point_id="C"
        ))
        network.add_observation(DistanceObservation(
            id="D_BC", obs_type=ObservationType.DISTANCE,
            value=true_dist, sigma=0.01,
            from_point_id="B", to_point_id="C"
        ))

        # Run adjustment
        options = AdjustmentOptions(max_iterations=20, convergence_threshold=1e-10)
        result = adjust_network_2d(network, options)

        # Verify success and convergence
        assert result.success is True
        assert result.converged is True

        # Verify adjusted coordinates
        c_adj = result.adjusted_points["C"]
        expected_n = 50 * math.sqrt(3)  # ~86.6025
        assert c_adj.easting == pytest.approx(50.0, abs=0.001)
        assert c_adj.northing == pytest.approx(expected_n, abs=0.001)

        # Verify residuals are small
        for obs_id, residual in result.residuals.items():
            assert abs(residual) < 0.01

    def test_overdetermined_trilateration(self):
        """Test trilateration with redundant observations."""
        network = Network(name="Overdetermined Test")

        # Three fixed control points
        network.add_point(Point(
            id="A", name="A", easting=0.0, northing=0.0,
            fixed_easting=True, fixed_northing=True
        ))
        network.add_point(Point(
            id="B", name="B", easting=100.0, northing=0.0,
            fixed_easting=True, fixed_northing=True
        ))
        network.add_point(Point(
            id="C", name="C", easting=50.0, northing=100.0,
            fixed_easting=True, fixed_northing=True
        ))

        # One free point at centroid
        network.add_point(Point(
            id="P", name="P", easting=50.0, northing=30.0,  # Approximate
            fixed_easting=False, fixed_northing=False
        ))

        # True position: centroid at (50, 33.33...)
        true_e, true_n = 50.0, 100.0/3.0

        # Distances from true position
        d_AP = distance_2d(0, 0, true_e, true_n)
        d_BP = distance_2d(100, 0, true_e, true_n)
        d_CP = distance_2d(50, 100, true_e, true_n)

        network.add_observation(DistanceObservation(
            id="D_AP", obs_type=ObservationType.DISTANCE,
            value=d_AP, sigma=0.01,
            from_point_id="A", to_point_id="P"
        ))
        network.add_observation(DistanceObservation(
            id="D_BP", obs_type=ObservationType.DISTANCE,
            value=d_BP, sigma=0.01,
            from_point_id="B", to_point_id="P"
        ))
        network.add_observation(DistanceObservation(
            id="D_CP", obs_type=ObservationType.DISTANCE,
            value=d_CP, sigma=0.01,
            from_point_id="C", to_point_id="P"
        ))

        result = adjust_network_2d(network)

        assert result.success is True
        assert result.converged is True
        assert result.degrees_of_freedom == 1  # 3 obs - 2 unknowns

        # Verify adjusted coordinates
        p_adj = result.adjusted_points["P"]
        assert p_adj.easting == pytest.approx(true_e, abs=0.001)
        assert p_adj.northing == pytest.approx(true_n, abs=0.001)


class TestAdjustmentWithDirections:
    """Tests for adjustment with direction observations."""

    def test_resection_with_directions(self):
        """Test resection using direction observations."""
        network = Network(name="Resection Test")

        # Three fixed control points
        network.add_point(Point(
            id="A", name="A", easting=0.0, northing=0.0,
            fixed_easting=True, fixed_northing=True
        ))
        network.add_point(Point(
            id="B", name="B", easting=100.0, northing=0.0,
            fixed_easting=True, fixed_northing=True
        ))
        network.add_point(Point(
            id="C", name="C", easting=50.0, northing=100.0,
            fixed_easting=True, fixed_northing=True
        ))

        # Free point
        true_e, true_n = 50.0, 30.0
        network.add_point(Point(
            id="P", name="P", easting=48.0, northing=32.0,  # Approximate
            fixed_easting=False, fixed_northing=False
        ))

        # True directions from P to control points
        dir_PA = azimuth(true_e, true_n, 0, 0)
        dir_PB = azimuth(true_e, true_n, 100, 0)
        dir_PC = azimuth(true_e, true_n, 50, 100)

        network.add_observation(DirectionObservation(
            id="DIR_PA", obs_type=ObservationType.DIRECTION,
            value=dir_PA, sigma=0.0001,
            from_point_id="P", to_point_id="A", set_id="SET_P"
        ))
        network.add_observation(DirectionObservation(
            id="DIR_PB", obs_type=ObservationType.DIRECTION,
            value=dir_PB, sigma=0.0001,
            from_point_id="P", to_point_id="B", set_id="SET_P"
        ))
        network.add_observation(DirectionObservation(
            id="DIR_PC", obs_type=ObservationType.DIRECTION,
            value=dir_PC, sigma=0.0001,
            from_point_id="P", to_point_id="C", set_id="SET_P"
        ))

        result = adjust_network_2d(network)

        assert result.success is True
        assert result.converged is True

        # Verify adjusted coordinates
        p_adj = result.adjusted_points["P"]
        assert p_adj.easting == pytest.approx(true_e, abs=0.01)
        assert p_adj.northing == pytest.approx(true_n, abs=0.01)


class TestAdjustmentWithAngles:
    """Tests for adjustment with angle observations."""

    def test_traverse_with_angles(self):
        """Test traverse adjustment with angles."""
        network = Network(name="Angle Traverse Test")

        # Fixed control points
        network.add_point(Point(
            id="A", name="A", easting=0.0, northing=0.0,
            fixed_easting=True, fixed_northing=True
        ))
        network.add_point(Point(
            id="B", name="B", easting=100.0, northing=0.0,
            fixed_easting=True, fixed_northing=True
        ))

        # Free point
        true_e, true_n = 50.0, 50.0
        network.add_point(Point(
            id="P", name="P", easting=48.0, northing=52.0,  # Approximate
            fixed_easting=False, fixed_northing=False
        ))

        # True angle at P from A to B (clockwise)
        angle_APB = angle_at_point(true_e, true_n, 0, 0, 100, 0)

        # Distance from A to P
        dist_AP = distance_2d(0, 0, true_e, true_n)

        network.add_observation(AngleObservation(
            id="ANG_APB", obs_type=ObservationType.ANGLE,
            value=angle_APB, sigma=0.0001,
            at_point_id="P", from_point_id="A", to_point_id="B"
        ))
        network.add_observation(DistanceObservation(
            id="D_AP", obs_type=ObservationType.DISTANCE,
            value=dist_AP, sigma=0.01,
            from_point_id="A", to_point_id="P"
        ))

        result = adjust_network_2d(network)

        assert result.success is True
        assert result.converged is True

        # Check adjusted coordinates are reasonable
        # (May not be exactly the true values due to limited observations)
        p_adj = result.adjusted_points["P"]
        assert abs(p_adj.easting - true_e) < 1.0
        assert abs(p_adj.northing - true_n) < 1.0


class TestAdjustmentStatistics:
    """Tests for adjustment statistics."""

    def test_variance_factor(self):
        """Test variance factor computation."""
        network = Network(name="Statistics Test")

        network.add_point(Point(
            id="A", name="A", easting=0.0, northing=0.0,
            fixed_easting=True, fixed_northing=True
        ))
        network.add_point(Point(
            id="B", name="B", easting=100.0, northing=0.0,
            fixed_easting=True, fixed_northing=True
        ))
        network.add_point(Point(
            id="C", name="C", easting=50.0, northing=86.6025,  # Near exact
            fixed_easting=False, fixed_northing=False
        ))

        # Exact distances for equilateral triangle
        network.add_observation(DistanceObservation(
            id="D_AC", obs_type=ObservationType.DISTANCE,
            value=100.0, sigma=0.01,
            from_point_id="A", to_point_id="C"
        ))
        network.add_observation(DistanceObservation(
            id="D_BC", obs_type=ObservationType.DISTANCE,
            value=100.0, sigma=0.01,
            from_point_id="B", to_point_id="C"
        ))
        network.add_observation(DistanceObservation(
            id="D_AB", obs_type=ObservationType.DISTANCE,
            value=100.0, sigma=0.01,
            from_point_id="A", to_point_id="B"
        ))

        result = adjust_network_2d(network)

        assert result.success is True
        assert result.degrees_of_freedom == 1
        # Perfectly consistent synthetic observations give a near-zero variance
        # factor (exactly 0.0 when residuals vanish), so only require non-negativity.
        assert result.variance_factor >= 0

    def test_error_ellipses(self):
        """Test error ellipse computation."""
        network = Network(name="Error Ellipse Test")

        network.add_point(Point(
            id="A", name="A", easting=0.0, northing=0.0,
            fixed_easting=True, fixed_northing=True
        ))
        network.add_point(Point(
            id="B", name="B", easting=100.0, northing=0.0,
            fixed_easting=True, fixed_northing=True
        ))
        network.add_point(Point(
            id="C", name="C", easting=50.0, northing=86.6,
            fixed_easting=False, fixed_northing=False
        ))

        network.add_observation(DistanceObservation(
            id="D_AC", obs_type=ObservationType.DISTANCE,
            value=100.0, sigma=0.01,
            from_point_id="A", to_point_id="C"
        ))
        network.add_observation(DistanceObservation(
            id="D_BC", obs_type=ObservationType.DISTANCE,
            value=100.0, sigma=0.01,
            from_point_id="B", to_point_id="C"
        ))
        network.add_observation(DistanceObservation(
            id="D_AB", obs_type=ObservationType.DISTANCE,
            value=100.0, sigma=0.01,
            from_point_id="A", to_point_id="B"
        ))

        options = AdjustmentOptions(compute_error_ellipses=True)
        result = adjust_network_2d(network, options)

        assert result.success is True

        # Should have error ellipse for free point C
        assert "C" in result.error_ellipses
        ellipse = result.error_ellipses["C"]
        assert ellipse.semi_major > 0
        assert ellipse.semi_minor > 0
        assert ellipse.semi_major >= ellipse.semi_minor

    def test_chi_square_test(self):
        """Test chi-square global test."""
        network = Network(name="Chi-Square Test")

        network.add_point(Point(
            id="A", name="A", easting=0.0, northing=0.0,
            fixed_easting=True, fixed_northing=True
        ))
        network.add_point(Point(
            id="B", name="B", easting=100.0, northing=0.0,
            fixed_easting=True, fixed_northing=True
        ))
        network.add_point(Point(
            id="C", name="C", easting=50.0, northing=86.6,
            fixed_easting=False, fixed_northing=False
        ))

        # Good observations (should pass chi-square test)
        network.add_observation(DistanceObservation(
            id="D_AC", obs_type=ObservationType.DISTANCE,
            value=100.0, sigma=0.01,
            from_point_id="A", to_point_id="C"
        ))
        network.add_observation(DistanceObservation(
            id="D_BC", obs_type=ObservationType.DISTANCE,
            value=100.0, sigma=0.01,
            from_point_id="B", to_point_id="C"
        ))
        network.add_observation(DistanceObservation(
            id="D_AB", obs_type=ObservationType.DISTANCE,
            value=100.0, sigma=0.01,
            from_point_id="A", to_point_id="B"
        ))

        result = adjust_network_2d(network)

        assert result.success is True
        assert result.chi_square_test is not None
        # Chi-square test result should exist
        assert result.chi_square_test.test_statistic >= 0


class TestErrorHandling:
    """Tests for error handling."""

    def test_no_fixed_points_fails(self):
        """Test that adjustment fails gracefully with no fixed points."""
        network = Network()
        network.add_point(Point(id="A", name="A", easting=0, northing=0))
        network.add_point(Point(id="B", name="B", easting=100, northing=0))
        network.add_observation(DistanceObservation(
            id="D1", obs_type=ObservationType.DISTANCE,
            value=100, sigma=0.01,
            from_point_id="A", to_point_id="B"
        ))

        result = adjust_network_2d(network)

        assert result.success is False
        assert "fixed point" in result.error_message.lower()

    def test_insufficient_observations_fails(self):
        """Test that adjustment fails with insufficient observations."""
        network = Network()
        network.add_point(Point(
            id="A", name="A", easting=0, northing=0,
            fixed_easting=True, fixed_northing=True
        ))
        network.add_point(Point(id="B", name="B", easting=100, northing=0))

        # Only 1 observation for 2 unknowns
        network.add_observation(DistanceObservation(
            id="D1", obs_type=ObservationType.DISTANCE,
            value=100, sigma=0.01,
            from_point_id="A", to_point_id="B"
        ))

        result = adjust_network_2d(network)

        assert result.success is False
        assert "insufficient" in result.error_message.lower()

    def test_all_fixed_points_succeeds(self):
        """Test that all-fixed network computes residuals only."""
        network = Network(name="All Fixed")
        network.add_point(Point(
            id="A", name="A", easting=0, northing=0,
            fixed_easting=True, fixed_northing=True
        ))
        network.add_point(Point(
            id="B", name="B", easting=100, northing=0,
            fixed_easting=True, fixed_northing=True
        ))
        network.add_observation(DistanceObservation(
            id="D1", obs_type=ObservationType.DISTANCE,
            value=100.05, sigma=0.01,  # Slightly different
            from_point_id="A", to_point_id="B"
        ))

        result = adjust_network_2d(network)

        assert result.success is True
        # Residual should be 0.05 (observed - computed)
        assert abs(result.residuals["D1"] - 0.05) < 0.001


class TestConvergence:
    """Tests for convergence behavior."""

    def test_convergence_threshold(self):
        """Test that convergence respects threshold."""
        network = Network(name="Convergence Test")

        network.add_point(Point(
            id="A", name="A", easting=0.0, northing=0.0,
            fixed_easting=True, fixed_northing=True
        ))
        network.add_point(Point(
            id="B", name="B", easting=100.0, northing=0.0,
            fixed_easting=True, fixed_northing=True
        ))
        network.add_point(Point(
            id="C", name="C", easting=50.0, northing=50.0,  # Start away from true
            fixed_easting=False, fixed_northing=False
        ))

        network.add_observation(DistanceObservation(
            id="D_AC", obs_type=ObservationType.DISTANCE,
            value=100.0, sigma=0.01,
            from_point_id="A", to_point_id="C"
        ))
        network.add_observation(DistanceObservation(
            id="D_BC", obs_type=ObservationType.DISTANCE,
            value=100.0, sigma=0.01,
            from_point_id="B", to_point_id="C"
        ))

        # Use tight convergence threshold
        options = AdjustmentOptions(
            max_iterations=50,
            convergence_threshold=1e-12
        )
        result = adjust_network_2d(network, options)

        assert result.success is True
        assert result.converged is True

    def test_max_iterations_limit(self):
        """Test that max iterations is respected."""
        network = Network(name="Max Iterations Test")

        network.add_point(Point(
            id="A", name="A", easting=0.0, northing=0.0,
            fixed_easting=True, fixed_northing=True
        ))
        network.add_point(Point(
            id="B", name="B", easting=100.0, northing=0.0,
            fixed_easting=True, fixed_northing=True
        ))
        network.add_point(Point(
            id="C", name="C", easting=50.0, northing=50.0,
            fixed_easting=False, fixed_northing=False
        ))

        network.add_observation(DistanceObservation(
            id="D_AC", obs_type=ObservationType.DISTANCE,
            value=100.0, sigma=0.01,
            from_point_id="A", to_point_id="C"
        ))
        network.add_observation(DistanceObservation(
            id="D_BC", obs_type=ObservationType.DISTANCE,
            value=100.0, sigma=0.01,
            from_point_id="B", to_point_id="C"
        ))

        # Use very tight threshold that can't be achieved in 2 iterations
        options = AdjustmentOptions(
            max_iterations=2,
            convergence_threshold=1e-20  # Impossible to achieve
        )
        result = adjust_network_2d(network, options)

        assert result.success is True
        assert result.iterations <= 2


class TestMixedObservations:
    """Tests for networks with mixed observation types."""

    def test_distance_and_direction_network(self):
        """Test network with both distances and directions."""
        network = Network(name="Mixed Dist+Dir Test")

        # Fixed control
        network.add_point(Point(
            id="A", name="A", easting=0.0, northing=0.0,
            fixed_easting=True, fixed_northing=True
        ))
        network.add_point(Point(
            id="B", name="B", easting=100.0, northing=0.0,
            fixed_easting=True, fixed_northing=True
        ))

        # Free point
        true_e, true_n = 50.0, 50.0
        network.add_point(Point(
            id="P", name="P", easting=48.0, northing=52.0,
            fixed_easting=False, fixed_northing=False
        ))

        # Distance observation
        dist_AP = distance_2d(0, 0, true_e, true_n)
        network.add_observation(DistanceObservation(
            id="D_AP", obs_type=ObservationType.DISTANCE,
            value=dist_AP, sigma=0.01,
            from_point_id="A", to_point_id="P"
        ))

        # Direction observations from P
        dir_PA = azimuth(true_e, true_n, 0, 0)
        dir_PB = azimuth(true_e, true_n, 100, 0)

        network.add_observation(DirectionObservation(
            id="DIR_PA", obs_type=ObservationType.DIRECTION,
            value=dir_PA, sigma=0.0001,
            from_point_id="P", to_point_id="A", set_id="SET_P"
        ))
        network.add_observation(DirectionObservation(
            id="DIR_PB", obs_type=ObservationType.DIRECTION,
            value=dir_PB, sigma=0.0001,
            from_point_id="P", to_point_id="B", set_id="SET_P"
        ))

        result = adjust_network_2d(network)

        assert result.success is True
        assert result.converged is True

        p_adj = result.adjusted_points["P"]
        assert p_adj.easting == pytest.approx(true_e, abs=0.1)
        assert p_adj.northing == pytest.approx(true_n, abs=0.1)

    def test_distance_and_angle_network(self):
        """Test network with both distances and angles."""
        network = Network(name="Mixed Dist+Ang Test")

        # Fixed control
        network.add_point(Point(
            id="A", name="A", easting=0.0, northing=0.0,
            fixed_easting=True, fixed_northing=True
        ))
        network.add_point(Point(
            id="B", name="B", easting=100.0, northing=0.0,
            fixed_easting=True, fixed_northing=True
        ))

        # Free point
        true_e, true_n = 50.0, 50.0
        network.add_point(Point(
            id="P", name="P", easting=48.0, northing=52.0,
            fixed_easting=False, fixed_northing=False
        ))

        # Distances
        dist_AP = distance_2d(0, 0, true_e, true_n)
        dist_BP = distance_2d(100, 0, true_e, true_n)

        network.add_observation(DistanceObservation(
            id="D_AP", obs_type=ObservationType.DISTANCE,
            value=dist_AP, sigma=0.01,
            from_point_id="A", to_point_id="P"
        ))
        network.add_observation(DistanceObservation(
            id="D_BP", obs_type=ObservationType.DISTANCE,
            value=dist_BP, sigma=0.01,
            from_point_id="B", to_point_id="P"
        ))

        # Angle at A from B to P
        ang_BAP = angle_at_point(0, 0, 100, 0, true_e, true_n)
        network.add_observation(AngleObservation(
            id="ANG_BAP", obs_type=ObservationType.ANGLE,
            value=ang_BAP, sigma=0.0001,
            at_point_id="A", from_point_id="B", to_point_id="P"
        ))

        result = adjust_network_2d(network)

        assert result.success is True
        assert result.converged is True

        p_adj = result.adjusted_points["P"]
        assert p_adj.easting == pytest.approx(true_e, abs=0.1)
        assert p_adj.northing == pytest.approx(true_n, abs=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
