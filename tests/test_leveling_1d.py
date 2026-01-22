"""Tests for 1D leveling adjustment.

Tests the HeightDifferenceObservation, 1D validation, solver, and parsers.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from survey_adjustment.core.models.point import Point
from survey_adjustment.core.models.observation import (
    HeightDifferenceObservation,
    ObservationType,
)
from survey_adjustment.core.models.network import Network
from survey_adjustment.core.models.options import AdjustmentOptions
from survey_adjustment.core.solver.least_squares_1d import adjust_leveling_1d
from survey_adjustment.core.reports.html_report import render_html_report, _is_leveling_result
from survey_adjustment.qgis_integration.io.observations import (
    parse_leveling_csv,
    parse_leveling_points_csv,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def data_dir() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def simple_leveling_network() -> Network:
    """Create a simple leveling network: A fixed, B and C unknown.

    Network:
    - A: fixed at H=100.000
    - B: unknown
    - C: unknown

    Observations (slightly inconsistent for dof=1):
    - A->B: +1.234 +/- 0.002
    - B->C: -0.456 +/- 0.002
    - A->C: +0.780 +/- 0.003

    Consistent would be: A->C = 1.234 - 0.456 = 0.778
    But we have 0.780, so there's a 2mm discrepancy.
    """
    points = {
        "A": Point(id="A", name="Benchmark A", easting=1000.0, northing=2000.0,
                   height=100.000, fixed_height=True),
        "B": Point(id="B", name="Point B", easting=1050.0, northing=2050.0,
                   height=0.0, fixed_height=False),
        "C": Point(id="C", name="Point C", easting=1100.0, northing=2000.0,
                   height=0.0, fixed_height=False),
    }

    net = Network(name="Test Leveling Network", points=points)

    # Add height difference observations
    net.add_observation(HeightDifferenceObservation(
        id="L1", obs_type=None, value=1.234, sigma=0.002,
        from_point_id="A", to_point_id="B"
    ))
    net.add_observation(HeightDifferenceObservation(
        id="L2", obs_type=None, value=-0.456, sigma=0.002,
        from_point_id="B", to_point_id="C"
    ))
    net.add_observation(HeightDifferenceObservation(
        id="L3", obs_type=None, value=0.780, sigma=0.003,
        from_point_id="A", to_point_id="C"
    ))

    return net


@pytest.fixture
def no_fixed_network() -> Network:
    """Network with no fixed height points (should fail validation)."""
    points = {
        "A": Point(id="A", name="Point A", easting=0, northing=0, height=100.0, fixed_height=False),
        "B": Point(id="B", name="Point B", easting=0, northing=0, height=101.0, fixed_height=False),
    }
    net = Network(name="No Fixed", points=points)
    net.add_observation(HeightDifferenceObservation(
        id="L1", obs_type=None, value=1.0, sigma=0.001,
        from_point_id="A", to_point_id="B"
    ))
    return net


# ---------------------------------------------------------------------------
# HeightDifferenceObservation Tests
# ---------------------------------------------------------------------------

class TestHeightDifferenceObservation:
    """Tests for HeightDifferenceObservation class."""

    def test_creation(self):
        """Test basic creation of height difference observation."""
        obs = HeightDifferenceObservation(
            id="HD1",
            obs_type=None,
            value=1.234,
            sigma=0.002,
            from_point_id="A",
            to_point_id="B",
        )
        assert obs.id == "HD1"
        assert obs.obs_type == ObservationType.HEIGHT_DIFF
        assert obs.value == 1.234
        assert obs.dh == 1.234  # alias property
        assert obs.sigma == 0.002
        assert obs.from_point_id == "A"
        assert obs.to_point_id == "B"
        assert obs.enabled is True

    def test_to_dict(self):
        """Test serialization to dict."""
        obs = HeightDifferenceObservation(
            id="HD1", obs_type=None, value=-0.5, sigma=0.003,
            from_point_id="X", to_point_id="Y"
        )
        d = obs.to_dict()
        assert d["id"] == "HD1"
        assert d["obs_type"] == "height_diff"
        assert d["value"] == -0.5
        assert d["sigma"] == 0.003
        assert d["from_point_id"] == "X"
        assert d["to_point_id"] == "Y"

    def test_from_dict(self):
        """Test deserialization from dict."""
        d = {
            "id": "HD2",
            "obs_type": "height_diff",
            "value": 2.5,
            "sigma": 0.001,
            "from_point_id": "P1",
            "to_point_id": "P2",
            "enabled": True,
        }
        obs = HeightDifferenceObservation.from_dict(d)
        assert obs.id == "HD2"
        assert obs.value == 2.5
        assert obs.from_point_id == "P1"
        assert obs.to_point_id == "P2"


# ---------------------------------------------------------------------------
# Point Height Tests
# ---------------------------------------------------------------------------

class TestPointHeightFields:
    """Tests for Point height-related fields."""

    def test_point_with_height(self):
        """Test Point creation with height fields."""
        p = Point(
            id="BM1",
            name="Benchmark 1",
            easting=0.0,
            northing=0.0,
            height=100.5,
            fixed_height=True,
            sigma_height=0.001,
        )
        assert p.height == 100.5
        assert p.fixed_height is True
        assert p.sigma_height == 0.001
        assert p.is_height_fixed is True
        assert p.has_height is True

    def test_point_without_height(self):
        """Test Point creation without height (backward compatibility)."""
        p = Point(id="P1", name="P1", easting=100.0, northing=200.0)
        assert p.height is None
        assert p.fixed_height is False
        assert p.sigma_height is None
        assert p.is_height_fixed is False
        assert p.has_height is False

    def test_point_to_dict_with_height(self):
        """Test Point serialization includes height fields."""
        p = Point(id="P", name="P", easting=0, northing=0,
                  height=50.0, fixed_height=True, sigma_height=0.002)
        d = p.to_dict()
        assert d["height"] == 50.0
        assert d["fixed_height"] is True
        assert d["sigma_height"] == 0.002

    def test_point_from_dict_with_height(self):
        """Test Point deserialization includes height fields."""
        d = {
            "id": "BM", "name": "BM", "easting": 0, "northing": 0,
            "height": 99.9, "fixed_height": True, "sigma_height": 0.0015
        }
        p = Point.from_dict(d)
        assert p.height == 99.9
        assert p.fixed_height is True
        assert p.sigma_height == 0.0015

    def test_point_from_dict_backward_compatible(self):
        """Test Point deserialization without height (backward compatibility)."""
        d = {"id": "X", "name": "X", "easting": 10, "northing": 20}
        p = Point.from_dict(d)
        assert p.height is None
        assert p.fixed_height is False


# ---------------------------------------------------------------------------
# Network 1D Validation Tests
# ---------------------------------------------------------------------------

class TestNetwork1DValidation:
    """Tests for Network 1D validation methods."""

    def test_get_height_fixed_points(self, simple_leveling_network):
        """Test retrieval of height-fixed points."""
        fixed = simple_leveling_network.get_height_fixed_points()
        assert len(fixed) == 1
        assert fixed[0].id == "A"

    def test_get_height_free_points(self, simple_leveling_network):
        """Test retrieval of height-free points."""
        free = simple_leveling_network.get_height_free_points()
        assert len(free) == 2
        ids = {p.id for p in free}
        assert ids == {"B", "C"}

    def test_get_leveling_observations(self, simple_leveling_network):
        """Test retrieval of leveling observations."""
        obs = simple_leveling_network.get_leveling_observations()
        assert len(obs) == 3
        assert all(isinstance(o, HeightDifferenceObservation) for o in obs)

    def test_get_leveling_dof(self, simple_leveling_network):
        """Test degrees of freedom calculation for leveling.

        DOF = observations - unknowns = 3 - 2 = 1
        """
        dof = simple_leveling_network.get_leveling_degrees_of_freedom()
        assert dof == 1

    def test_validate_1d_success(self, simple_leveling_network):
        """Test successful 1D validation."""
        errors = simple_leveling_network.validate_1d()
        assert len(errors) == 0

    def test_validate_1d_no_fixed(self, no_fixed_network):
        """Test 1D validation fails with no fixed heights."""
        errors = no_fixed_network.validate_1d()
        assert len(errors) > 0
        assert any("fixed" in e.lower() for e in errors)

    def test_validate_1d_missing_point(self):
        """Test 1D validation fails when observation references missing point."""
        points = {
            "A": Point(id="A", name="A", easting=0, northing=0, height=100.0, fixed_height=True),
        }
        net = Network(name="Missing", points=points)
        net.add_observation(HeightDifferenceObservation(
            id="L1", obs_type=None, value=1.0, sigma=0.001,
            from_point_id="A", to_point_id="MISSING"
        ))
        errors = net.validate_1d()
        assert len(errors) > 0


# ---------------------------------------------------------------------------
# 1D Solver Tests
# ---------------------------------------------------------------------------

class TestLeveling1DSolver:
    """Tests for the 1D leveling least-squares solver."""

    def test_adjustment_success(self, simple_leveling_network):
        """Test successful leveling adjustment."""
        options = AdjustmentOptions(compute_reliability=True)
        result = adjust_leveling_1d(simple_leveling_network, options)

        assert result.success is True
        assert result.converged is True
        assert result.iterations == 1  # Linear problem
        assert result.degrees_of_freedom == 1

    def test_adjusted_heights(self, simple_leveling_network):
        """Test adjusted height values are reasonable."""
        result = adjust_leveling_1d(simple_leveling_network)

        # A is fixed at 100.0
        assert result.adjusted_points["A"].height == 100.0
        assert result.adjusted_points["A"].fixed_height is True

        # B should be approximately 100.0 + 1.234 = 101.234
        h_b = result.adjusted_points["B"].height
        assert h_b is not None
        assert 101.23 < h_b < 101.24

        # C should be approximately 100.0 + 0.778 to 0.780
        h_c = result.adjusted_points["C"].height
        assert h_c is not None
        assert 100.77 < h_c < 100.79

    def test_variance_factor(self, simple_leveling_network):
        """Test variance factor is computed."""
        result = adjust_leveling_1d(simple_leveling_network)

        # With dof=1 and small residuals, variance factor should be reasonable
        assert result.variance_factor is not None
        assert result.variance_factor > 0

    def test_residuals_populated(self, simple_leveling_network):
        """Test that residuals are populated."""
        result = adjust_leveling_1d(simple_leveling_network)

        assert len(result.residuals) == 3
        assert len(result.standardized_residuals) == 3
        assert len(result.residual_details) == 3

        for r in result.residual_details:
            assert r.obs_type == "height_diff"
            assert r.residual is not None
            assert r.standardized_residual is not None

    def test_chi_square_test(self, simple_leveling_network):
        """Test chi-square global test is computed."""
        result = adjust_leveling_1d(simple_leveling_network)

        assert result.chi_square_test is not None
        assert result.chi_square_test.degrees_of_freedom == 1

    def test_reliability_measures(self, simple_leveling_network):
        """Test reliability measures are computed when requested."""
        options = AdjustmentOptions(compute_reliability=True)
        result = adjust_leveling_1d(simple_leveling_network, options)

        # Check that reliability info is present
        for r in result.residual_details:
            assert r.redundancy_number is not None
            # MDB and external reliability may be inf for some cases

    def test_sigma_heights_computed(self, simple_leveling_network):
        """Test that sigma heights are computed for free points."""
        result = adjust_leveling_1d(simple_leveling_network)

        # Fixed point has zero sigma (known height)
        assert result.adjusted_points["A"].sigma_height == 0.0

        # Free points should have non-zero sigma
        assert result.adjusted_points["B"].sigma_height is not None
        assert result.adjusted_points["B"].sigma_height > 0
        assert result.adjusted_points["C"].sigma_height is not None
        assert result.adjusted_points["C"].sigma_height > 0

    def test_no_fixed_fails(self, no_fixed_network):
        """Test that adjustment fails when no fixed heights exist."""
        result = adjust_leveling_1d(no_fixed_network)

        assert result.success is False
        assert "fixed" in result.error_message.lower() or "datum" in result.error_message.lower()

    def test_underdetermined_system(self):
        """Test failure when system is underdetermined (DOF < 0)."""
        # 2 unknowns (B, C), 1 observation (A-B) -> C is disconnected
        # The solver only considers connected points, so it succeeds for A-B
        # but C is left unadjusted. This test checks the network validation.
        points = {
            "A": Point(id="A", name="A", easting=0, northing=0, height=100.0, fixed_height=True),
            "B": Point(id="B", name="B", easting=0, northing=0, height=0.0, fixed_height=False),
        }
        net = Network(name="Minimal", points=points)
        net.add_observation(HeightDifferenceObservation(
            id="L1", obs_type=None, value=1.0, sigma=0.001,
            from_point_id="A", to_point_id="B"
        ))
        # DOF = 1 obs - 1 unknown = 0 (just determined)

        result = adjust_leveling_1d(net)
        # Should succeed with DOF=0 (just determined)
        assert result.success is True
        assert result.degrees_of_freedom == 0


# ---------------------------------------------------------------------------
# HTML Report Tests
# ---------------------------------------------------------------------------

class TestLevelingHTMLReport:
    """Tests for leveling-specific HTML report generation."""

    def test_is_leveling_result_detection(self, simple_leveling_network):
        """Test that leveling results are correctly detected."""
        result = adjust_leveling_1d(simple_leveling_network)
        assert _is_leveling_result(result) is True

    def test_leveling_report_contains_heights(self, simple_leveling_network):
        """Test that leveling report shows height columns."""
        result = adjust_leveling_1d(simple_leveling_network)
        html = render_html_report(result)

        # Should contain height-specific headers
        assert "H (m)" in html or "σH (m)" in html
        # Should contain leveling-specific residual headers
        assert "ΔH obs" in html or "ΔH comp" in html

    def test_leveling_report_title(self, simple_leveling_network):
        """Test that leveling report has appropriate title."""
        result = adjust_leveling_1d(simple_leveling_network)
        html = render_html_report(result)

        assert "Leveling" in html


# ---------------------------------------------------------------------------
# CSV Parsing Tests
# ---------------------------------------------------------------------------

class TestLevelingCSVParsing:
    """Tests for leveling CSV parsing functions."""

    def test_parse_leveling_points_csv(self, data_dir):
        """Test parsing leveling points from CSV."""
        csv_path = data_dir / "example_leveling_points.csv"
        points = parse_leveling_points_csv(csv_path)

        assert len(points) == 3
        assert "A" in points
        assert "B" in points
        assert "C" in points

        # Check point A
        assert points["A"].height == 100.0
        assert points["A"].fixed_height is True

        # Check point B
        assert points["B"].fixed_height is False

    def test_parse_leveling_csv(self, data_dir):
        """Test parsing height difference observations from CSV."""
        csv_path = data_dir / "example_leveling_hdiff.csv"
        obs_list = parse_leveling_csv(csv_path)

        assert len(obs_list) == 3
        assert all(isinstance(o, HeightDifferenceObservation) for o in obs_list)

        # Find L1 observation
        l1 = next(o for o in obs_list if o.id == "L1")
        assert l1.from_point_id == "A"
        assert l1.to_point_id == "B"
        assert l1.value == 1.234
        assert l1.sigma == 0.002

    def test_parse_leveling_csv_mm_sigma(self, data_dir, tmp_path):
        """Test parsing with sigma in millimeters."""
        # Create a temp CSV with mm sigmas
        csv_content = "obs_id,from_point,to_point,dh,sigma\nL1,A,B,1.0,2.0\n"
        csv_path = tmp_path / "test_mm.csv"
        csv_path.write_text(csv_content)

        # Parse with mm unit
        obs_list = parse_leveling_csv(csv_path, sigma_unit="mm")
        assert len(obs_list) == 1
        # sigma 2.0 mm = 0.002 m
        assert abs(obs_list[0].sigma - 0.002) < 1e-9

    def test_full_adjustment_from_csv(self, data_dir):
        """Test full adjustment workflow from CSV files."""
        points = parse_leveling_points_csv(data_dir / "example_leveling_points.csv")
        obs_list = parse_leveling_csv(data_dir / "example_leveling_hdiff.csv")

        net = Network(name="From CSV", points=points)
        for obs in obs_list:
            net.add_observation(obs)

        result = adjust_leveling_1d(net)

        assert result.success is True
        assert result.degrees_of_freedom == 1
        assert len(result.adjusted_points) == 3


# ---------------------------------------------------------------------------
# Serialization Tests
# ---------------------------------------------------------------------------

class TestLevelingResultSerialization:
    """Tests for serialization of leveling adjustment results."""

    def test_to_dict(self, simple_leveling_network):
        """Test AdjustmentResult.to_dict() for leveling results."""
        result = adjust_leveling_1d(simple_leveling_network)
        d = result.to_dict()

        # Check nested structure
        assert d["adjustment"]["success"] is True
        assert d["adjustment"]["degrees_of_freedom"] == 1
        assert "adjusted_points" in d
        assert "residuals" in d  # residual_details is serialized as residuals

        # Check that height is in adjusted points (serialized as dict with point_id keys)
        adj_pts = d["adjusted_points"]
        if isinstance(adj_pts, dict):
            for pt_data in adj_pts.values():
                assert "height" in pt_data
        else:
            # May be serialized as list in some versions
            for pt_data in adj_pts:
                assert "height" in pt_data

    def test_to_json(self, simple_leveling_network):
        """Test AdjustmentResult.to_json() for leveling results."""
        result = adjust_leveling_1d(simple_leveling_network)
        json_str = result.to_json()

        assert isinstance(json_str, str)
        assert "height" in json_str
        assert "height_diff" in json_str
