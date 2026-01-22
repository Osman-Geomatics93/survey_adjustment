"""Tests for Phase 7B: Constraint Health Analysis and Auto-Datum.

Tests the constraint health module including:
- Constraint health analysis for 2D, 1D, 3D GNSS, and mixed adjustments
- Auto-datum feature that applies minimal constraints
- Integration with solvers and AdjustmentResult
"""

import pytest
import math

from survey_adjustment.core.models.network import Network
from survey_adjustment.core.models.point import Point
from survey_adjustment.core.models.observation import (
    DistanceObservation,
    DirectionObservation,
    HeightDifferenceObservation,
    GnssBaselineObservation,
)
from survey_adjustment.core.models.options import AdjustmentOptions
from survey_adjustment.core.validation import (
    ConstraintStatus,
    ConstraintHealth,
    AppliedConstraint,
    analyze_constraint_health,
    apply_minimal_constraints,
    format_validation_message,
)
from survey_adjustment.core.solver.least_squares_2d import adjust_network_2d
from survey_adjustment.core.solver.least_squares_1d import adjust_leveling_1d
from survey_adjustment.core.solver.least_squares_3d import adjust_gnss_3d
from survey_adjustment.core.solver.least_squares_mixed import adjust_network_mixed


def make_distance(obs_id, from_pt, to_pt, value, sigma):
    """Helper to create DistanceObservation with keyword args."""
    return DistanceObservation(
        id=obs_id, obs_type=None, value=value, sigma=sigma,
        from_point_id=from_pt, to_point_id=to_pt
    )


def make_height_diff(obs_id, from_pt, to_pt, dh, sigma):
    """Helper to create HeightDifferenceObservation with keyword args."""
    return HeightDifferenceObservation(
        id=obs_id, obs_type=None, value=dh, sigma=sigma,
        from_point_id=from_pt, to_point_id=to_pt
    )


def make_gnss_baseline(obs_id, from_pt, to_pt, dE, dN, dH, sigma=0.01):
    """Helper to create GnssBaselineObservation with keyword args."""
    # Create diagonal covariance matrix with given sigma
    cov = [
        [sigma**2, 0.0, 0.0],
        [0.0, sigma**2, 0.0],
        [0.0, 0.0, sigma**2]
    ]
    return GnssBaselineObservation(
        id=obs_id, obs_type=None, value=0.0, sigma=sigma,
        from_point_id=from_pt, to_point_id=to_pt,
        dE=dE, dN=dN, dH=dH,
        cov_EE=cov[0][0], cov_EN=cov[0][1], cov_EH=cov[0][2],
        cov_NN=cov[1][1], cov_NH=cov[1][2], cov_HH=cov[2][2]
    )


class TestConstraintStatus:
    """Tests for ConstraintStatus enum."""

    def test_status_values(self):
        """Test enum values are correct."""
        assert ConstraintStatus.OK.value == "ok"
        assert ConstraintStatus.WARNING.value == "warning"
        assert ConstraintStatus.ERROR.value == "error"


class TestAppliedConstraint:
    """Tests for AppliedConstraint dataclass."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        constraint = AppliedConstraint(
            point_id="A",
            constraint_type="fixed_easting",
            value=1000.0,
            reason="Auto-datum: translation constraint (E)"
        )
        d = constraint.to_dict()
        assert d["point_id"] == "A"
        assert d["constraint_type"] == "fixed_easting"
        assert d["value"] == 1000.0
        assert "Auto-datum" in d["reason"]


class TestConstraintHealth:
    """Tests for ConstraintHealth dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        health = ConstraintHealth()
        assert health.is_solvable == False
        assert health.horizontal_status == ConstraintStatus.ERROR
        assert health.dof == 0

    def test_to_dict(self):
        """Test serialization to dictionary."""
        health = ConstraintHealth(
            is_solvable=True,
            horizontal_status=ConstraintStatus.OK,
            horizontal_message="Fixed E: ['A'], Fixed N: ['A']",
            fixed_easting_points=["A"],
            fixed_northing_points=["A"],
            dof=5,
            num_observations=10,
            num_unknowns=5,
        )
        d = health.to_dict()
        assert d["is_solvable"] == True
        assert d["horizontal"]["status"] == "ok"
        assert d["horizontal"]["fixed_easting_points"] == ["A"]
        assert d["degrees_of_freedom"]["value"] == 5


class TestAnalyze2DConstraintHealth:
    """Tests for 2D classical adjustment constraint health analysis."""

    def test_valid_network(self):
        """Test analysis of a properly constrained 2D network."""
        # Create a simple trilateration with 2 fixed points (overdetermined)
        points = {
            "A": Point("A", "A", 0.0, 0.0, fixed_easting=True, fixed_northing=True),
            "B": Point("B", "B", 100.0, 0.0, fixed_easting=True, fixed_northing=True),
            "C": Point("C", "C", 50.0, 86.6),  # Apex of equilateral triangle
        }
        net = Network(points=points)
        # 3 observations for 2 unknowns -> DOF = 1
        net.add_observation(make_distance("D1", "A", "C", 100.0, 0.003))
        net.add_observation(make_distance("D2", "B", "C", 100.0, 0.003))
        net.add_observation(make_distance("D3", "A", "B", 100.0, 0.003))

        health = analyze_constraint_health(net, adjustment_type="2d")

        assert health.is_solvable == True
        assert health.horizontal_status == ConstraintStatus.OK
        assert "A" in health.fixed_easting_points
        assert "A" in health.fixed_northing_points
        assert health.connectivity_status == ConstraintStatus.OK
        assert health.dof >= 0

    def test_missing_horizontal_datum(self):
        """Test detection of missing horizontal datum."""
        points = {
            "A": Point("A", "A", 0.0, 0.0),  # Not fixed
            "B": Point("B", "B", 100.0, 0.0),
        }
        net = Network(points=points)
        net.add_observation(make_distance("D1", "A", "B", 100.0, 0.003))

        health = analyze_constraint_health(net, adjustment_type="2d")

        assert health.is_solvable == False
        assert health.horizontal_status == ConstraintStatus.ERROR
        assert len(health.errors) > 0
        assert "horizontal datum" in health.errors[0].lower()

    def test_underdetermined_system(self):
        """Test detection of underdetermined system (DOF < 0)."""
        points = {
            "A": Point("A", "A", 0.0, 0.0, fixed_easting=True, fixed_northing=True),
            "B": Point("B", "B", 100.0, 0.0),
            "C": Point("C", "C", 100.0, 100.0),
            "D": Point("D", "D", 0.0, 100.0),
        }
        net = Network(points=points)
        # Only 2 observations but 6 unknowns (3 free points * 2 coords)
        net.add_observation(make_distance("D1", "A", "B", 100.0, 0.003))
        net.add_observation(make_distance("D2", "B", "C", 100.0, 0.003))

        health = analyze_constraint_health(net, adjustment_type="2d")

        assert health.dof < 0
        assert health.dof_status == ConstraintStatus.ERROR


class TestAnalyze1DConstraintHealth:
    """Tests for 1D leveling adjustment constraint health analysis."""

    def test_valid_leveling_network(self):
        """Test analysis of a properly constrained leveling network."""
        points = {
            "BM1": Point("BM1", "BM1", 0.0, 0.0, height=100.0, fixed_height=True),
            "P1": Point("P1", "P1", 100.0, 0.0, height=101.5),
            "P2": Point("P2", "P2", 200.0, 0.0, height=102.0),
        }
        net = Network(points=points)
        net.add_observation(make_height_diff("HD1", "BM1", "P1", 1.5, 0.002))
        net.add_observation(make_height_diff("HD2", "P1", "P2", 0.5, 0.002))
        net.add_observation(make_height_diff("HD3", "BM1", "P2", 2.0, 0.003))

        health = analyze_constraint_health(net, adjustment_type="1d")

        assert health.is_solvable == True
        assert health.height_status == ConstraintStatus.OK
        assert health.height_required == True
        assert "BM1" in health.fixed_height_points
        assert health.dof > 0

    def test_missing_height_datum(self):
        """Test detection of missing height datum."""
        points = {
            "P1": Point("P1", "P1", 0.0, 0.0, height=100.0),  # Not fixed
            "P2": Point("P2", "P2", 100.0, 0.0, height=101.0),
        }
        net = Network(points=points)
        net.add_observation(make_height_diff("HD1", "P1", "P2", 1.0, 0.002))

        health = analyze_constraint_health(net, adjustment_type="1d")

        assert health.is_solvable == False
        assert health.height_status == ConstraintStatus.ERROR
        assert len(health.errors) > 0
        assert "height datum" in health.errors[0].lower()


class TestAnalyze3DConstraintHealth:
    """Tests for 3D GNSS adjustment constraint health analysis."""

    def test_valid_gnss_network(self):
        """Test analysis of a properly constrained GNSS network."""
        points = {
            "BASE": Point("BASE", "BASE", 0.0, 0.0, height=100.0,
                         fixed_easting=True, fixed_northing=True, fixed_height=True),
            "P1": Point("P1", "P1", 100.0, 0.0, height=100.0),
            "P2": Point("P2", "P2", 100.0, 100.0, height=100.0),
        }
        net = Network(points=points)

        # Add GNSS baselines
        net.add_observation(make_gnss_baseline("B1", "BASE", "P1", dE=100.0, dN=0.0, dH=0.0))
        net.add_observation(make_gnss_baseline("B2", "BASE", "P2", dE=100.0, dN=100.0, dH=0.0))

        health = analyze_constraint_health(net, adjustment_type="3d")

        assert health.is_solvable == True
        assert health.horizontal_status == ConstraintStatus.OK
        assert health.height_status == ConstraintStatus.OK
        assert "BASE" in health.fixed_easting_points
        assert "BASE" in health.fixed_height_points

    def test_missing_3d_datum(self):
        """Test detection of incomplete 3D datum."""
        points = {
            "BASE": Point("BASE", "BASE", 0.0, 0.0, height=100.0),  # Not fixed
            "P1": Point("P1", "P1", 100.0, 0.0, height=100.0),
        }
        net = Network(points=points)
        net.add_observation(make_gnss_baseline("B1", "BASE", "P1", dE=100.0, dN=0.0, dH=0.0))

        health = analyze_constraint_health(net, adjustment_type="3d")

        assert health.is_solvable == False
        assert health.horizontal_status == ConstraintStatus.ERROR
        assert health.height_status == ConstraintStatus.ERROR


class TestApplyMinimalConstraints:
    """Tests for auto-datum feature."""

    def test_apply_2d_constraints(self):
        """Test applying minimal 2D constraints."""
        points = {
            "A": Point("A", "A", 1000.0, 2000.0),
            "B": Point("B", "B", 1100.0, 2000.0),
        }
        net = Network(points=points)
        net.add_observation(make_distance("D1", "A", "B", 100.0, 0.003))

        applied = apply_minimal_constraints(net, adjustment_type="2d")

        # For distance-only (no directions), should fix:
        # - E and N of first point (A) for translation
        # - E of second point (B) for rotation
        assert len(applied) == 3
        assert any(c.point_id == "A" and c.constraint_type == "fixed_easting" for c in applied)
        assert any(c.point_id == "A" and c.constraint_type == "fixed_northing" for c in applied)
        assert any(c.point_id == "B" and c.constraint_type == "fixed_easting" for c in applied)

        # Check that the points were actually modified
        assert net.points["A"].fixed_easting == True
        assert net.points["A"].fixed_northing == True
        assert net.points["B"].fixed_easting == True

    def test_apply_1d_constraints(self):
        """Test applying minimal 1D constraints."""
        points = {
            "P1": Point("P1", "P1", 0.0, 0.0, height=100.0),
            "P2": Point("P2", "P2", 100.0, 0.0, height=101.0),
        }
        net = Network(points=points)
        net.add_observation(make_height_diff("HD1", "P1", "P2", 1.0, 0.002))

        applied = apply_minimal_constraints(net, adjustment_type="1d")

        assert len(applied) == 1
        assert applied[0].point_id == "P1"
        assert applied[0].constraint_type == "fixed_height"
        assert net.points["P1"].fixed_height == True

    def test_apply_3d_constraints(self):
        """Test applying minimal 3D constraints."""
        points = {
            "BASE": Point("BASE", "BASE", 0.0, 0.0, height=100.0),
            "P1": Point("P1", "P1", 100.0, 0.0, height=100.0),
        }
        net = Network(points=points)
        net.add_observation(make_gnss_baseline("B1", "BASE", "P1", dE=100.0, dN=0.0, dH=0.0))

        applied = apply_minimal_constraints(net, adjustment_type="3d")

        # Should fix E, N, H of first point alphabetically (BASE)
        assert len(applied) == 3
        assert net.points["BASE"].fixed_easting == True
        assert net.points["BASE"].fixed_northing == True
        assert net.points["BASE"].fixed_height == True


class TestFormatValidationMessage:
    """Tests for format_validation_message function."""

    def test_format_errors_and_warnings(self):
        """Test formatting of errors and warnings."""
        health = ConstraintHealth(
            errors=["Missing horizontal datum"],
            warnings=["Rotation may be weakly constrained"],
        )
        msg = format_validation_message(health)

        assert "Errors:" in msg
        assert "Missing horizontal datum" in msg
        assert "Warnings:" in msg
        assert "Rotation" in msg

    def test_format_applied_constraints(self):
        """Test formatting of applied constraints."""
        health = ConstraintHealth(
            applied_constraints=[
                AppliedConstraint("A", "fixed_easting", 1000.0, "Auto-datum")
            ]
        )
        msg = format_validation_message(health)

        assert "Auto-applied constraints:" in msg
        assert "A" in msg
        assert "fixed_easting" in msg

    def test_format_ok_message(self):
        """Test formatting when everything is OK."""
        health = ConstraintHealth(is_solvable=True)
        msg = format_validation_message(health)

        assert msg == "Constraints OK"


class TestSolverIntegration:
    """Tests for constraint health integration with solvers."""

    def test_2d_solver_includes_datum_summary(self):
        """Test that 2D solver includes datum_summary in result."""
        # Need overdetermined network: 2 fixed points + 1 free point with 3 obs
        points = {
            "A": Point("A", "A", 0.0, 0.0, fixed_easting=True, fixed_northing=True),
            "B": Point("B", "B", 100.0, 0.0, fixed_easting=True, fixed_northing=True),
            "C": Point("C", "C", 50.0, 86.6),
        }
        net = Network(points=points)
        # 3 observations for 2 unknowns = DOF 1
        net.add_observation(make_distance("D1", "A", "C", 100.0, 0.003))
        net.add_observation(make_distance("D2", "B", "C", 100.0, 0.003))
        net.add_observation(make_distance("D3", "A", "B", 100.0, 0.003))

        result = adjust_network_2d(net)

        assert result.datum_summary is not None
        assert "is_solvable" in result.datum_summary
        assert "horizontal" in result.datum_summary
        assert result.datum_summary["is_solvable"] == True

    def test_1d_solver_includes_datum_summary(self):
        """Test that 1D solver includes datum_summary in result."""
        points = {
            "BM1": Point("BM1", "BM1", 0.0, 0.0, height=100.0, fixed_height=True),
            "P1": Point("P1", "P1", 100.0, 0.0, height=101.5),
        }
        net = Network(points=points)
        net.add_observation(make_height_diff("HD1", "BM1", "P1", 1.5, 0.002))

        result = adjust_leveling_1d(net)

        assert result.datum_summary is not None
        assert result.datum_summary["height"]["status"] == "ok"

    def test_auto_datum_in_2d_solver(self):
        """Test auto-datum feature in 2D solver."""
        # Network with 3 points, no fixed, but enough observations for DOF=0 after auto-datum
        points = {
            "A": Point("A", "A", 0.0, 0.0),  # Not fixed - will be auto-fixed
            "B": Point("B", "B", 100.0, 0.0),
            "C": Point("C", "C", 50.0, 86.6),
        }
        net = Network(points=points)
        # 5 observations: with 2 constraints on A (E, N), we have 4 unknowns, DOF = 1
        net.add_observation(make_distance("D1", "A", "B", 100.0, 0.003))
        net.add_observation(make_distance("D2", "A", "C", 100.0, 0.003))
        net.add_observation(make_distance("D3", "B", "C", 100.0, 0.003))
        net.add_observation(make_distance("D4", "A", "B", 100.0, 0.003))  # Redundant
        net.add_observation(make_distance("D5", "A", "C", 100.0, 0.003))  # Redundant

        options = AdjustmentOptions(auto_datum=True)
        result = adjust_network_2d(net, options)

        # Adjustment should succeed with auto-datum
        assert result.success == True
        assert len(result.applied_auto_constraints) > 0

        # Check that constraint info is in result
        applied = result.applied_auto_constraints
        assert any(c["constraint_type"] == "fixed_easting" for c in applied)
        assert any(c["constraint_type"] == "fixed_northing" for c in applied)

    def test_auto_datum_in_1d_solver(self):
        """Test auto-datum feature in 1D solver."""
        points = {
            "P1": Point("P1", "P1", 0.0, 0.0, height=100.0),  # Not fixed
            "P2": Point("P2", "P2", 100.0, 0.0, height=101.0),
        }
        net = Network(points=points)
        net.add_observation(make_height_diff("HD1", "P1", "P2", 1.0, 0.002))

        options = AdjustmentOptions(auto_datum=True)
        result = adjust_leveling_1d(net, options)

        assert result.success == True
        assert len(result.applied_auto_constraints) == 1
        assert result.applied_auto_constraints[0]["constraint_type"] == "fixed_height"

    def test_auto_datum_disabled_by_default(self):
        """Test that auto-datum is disabled by default."""
        points = {
            "A": Point("A", "A", 0.0, 0.0),  # Not fixed
            "B": Point("B", "B", 100.0, 0.0),
        }
        net = Network(points=points)
        net.add_observation(make_distance("D1", "A", "B", 100.0, 0.003))

        # Default options have auto_datum=False
        result = adjust_network_2d(net)

        # Should fail because no datum is defined
        assert result.success == False
        assert result.datum_summary is not None


class TestJSONSerialization:
    """Tests for JSON serialization of constraint health."""

    def test_adjustment_result_to_dict_includes_datum_summary(self):
        """Test that to_dict() includes datum_summary."""
        # Need overdetermined network
        points = {
            "A": Point("A", "A", 0.0, 0.0, fixed_easting=True, fixed_northing=True),
            "B": Point("B", "B", 100.0, 0.0, fixed_easting=True, fixed_northing=True),
            "C": Point("C", "C", 50.0, 86.6),
        }
        net = Network(points=points)
        net.add_observation(make_distance("D1", "A", "C", 100.0, 0.003))
        net.add_observation(make_distance("D2", "B", "C", 100.0, 0.003))
        net.add_observation(make_distance("D3", "A", "B", 100.0, 0.003))

        result = adjust_network_2d(net)
        result_dict = result.to_dict()

        assert "datum_summary" in result_dict
        assert result_dict["datum_summary"] is not None
        assert "is_solvable" in result_dict["datum_summary"]

    def test_adjustment_result_from_dict_parses_datum_summary(self):
        """Test that from_dict() correctly parses datum_summary."""
        from survey_adjustment.core.results.adjustment_result import AdjustmentResult

        data = {
            "metadata": {"plugin_version": "1.0.0"},
            "adjustment": {"success": True, "iterations": 1, "converged": True},
            "datum_summary": {
                "is_solvable": True,
                "horizontal": {"status": "ok", "message": "OK"},
            },
            "applied_auto_constraints": [
                {"point_id": "A", "constraint_type": "fixed_easting", "value": 1000.0, "reason": "Auto"}
            ],
            "adjusted_points": [],
            "residuals": [],
        }

        result = AdjustmentResult.from_dict(data)

        assert result.datum_summary is not None
        assert result.datum_summary["is_solvable"] == True
        assert len(result.applied_auto_constraints) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
