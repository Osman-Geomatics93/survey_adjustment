"""Tests for 3D GNSS baseline adjustment.

Tests the GnssBaselineObservation, 3D validation, solver, and parsers.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from survey_adjustment.core.models.point import Point
from survey_adjustment.core.models.observation import (
    GnssBaselineObservation,
    ObservationType,
)
from survey_adjustment.core.models.network import Network
from survey_adjustment.core.models.options import AdjustmentOptions
from survey_adjustment.core.solver.least_squares_3d import adjust_gnss_3d
from survey_adjustment.core.reports.html_report import render_html_report, _is_gnss_result
from survey_adjustment.qgis_integration.io.observations import (
    parse_gnss_baselines_csv,
    parse_gnss_points_csv,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def data_dir() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def simple_gnss_network() -> Network:
    """Create a simple GNSS baseline network: BASE fixed, P1 and P2 unknown.

    Network:
    - BASE: fixed at E=500000, N=4000000, H=100 (all fixed)
    - P1: unknown
    - P2: unknown

    Baselines (slightly inconsistent for DOF > 0):
    - BASE->P1: dE=100.002, dN=50.001, dH=5.001
    - BASE->P2: dE=200.001, dN=100.002, dH=10.003
    - P1->P2: dE=99.998, dN=50.000, dH=5.000

    Consistent would be P1->P2 = (200.001-100.002, 100.002-50.001, 10.003-5.001) = (99.999, 50.001, 5.002)
    But we have (99.998, 50.000, 5.000), small inconsistencies.
    """
    points = {
        "BASE": Point(
            id="BASE", name="Base Station",
            easting=500000.0, northing=4000000.0, height=100.0,
            fixed_easting=True, fixed_northing=True, fixed_height=True
        ),
        "P1": Point(
            id="P1", name="Point 1",
            easting=500100.0, northing=4000050.0, height=105.0,
            fixed_easting=False, fixed_northing=False, fixed_height=False
        ),
        "P2": Point(
            id="P2", name="Point 2",
            easting=500200.0, northing=4000100.0, height=110.0,
            fixed_easting=False, fixed_northing=False, fixed_height=False
        ),
    }

    net = Network(name="Test GNSS Network", points=points)

    # Add GNSS baseline observations with covariance
    # sigma ~ 2mm for E,N, 3mm for H
    cov_ee = 0.000004  # (0.002)^2
    cov_nn = 0.000004
    cov_hh = 0.000009  # (0.003)^2

    net.add_observation(GnssBaselineObservation(
        id="BL1", obs_type=None, value=0.0, sigma=1.0,
        from_point_id="BASE", to_point_id="P1",
        dE=100.002, dN=50.001, dH=5.001,
        cov_EE=cov_ee, cov_EN=0.0000005, cov_EH=0.0000003,
        cov_NN=cov_nn, cov_NH=0.0000004, cov_HH=cov_hh
    ))
    net.add_observation(GnssBaselineObservation(
        id="BL2", obs_type=None, value=0.0, sigma=1.0,
        from_point_id="BASE", to_point_id="P2",
        dE=200.001, dN=100.002, dH=10.003,
        cov_EE=cov_ee, cov_EN=0.0000006, cov_EH=0.0000004,
        cov_NN=cov_nn, cov_NH=0.0000005, cov_HH=cov_hh
    ))
    net.add_observation(GnssBaselineObservation(
        id="BL3", obs_type=None, value=0.0, sigma=1.0,
        from_point_id="P1", to_point_id="P2",
        dE=99.998, dN=50.000, dH=5.000,
        cov_EE=cov_ee, cov_EN=0.0000005, cov_EH=0.0000003,
        cov_NN=cov_nn, cov_NH=0.0000004, cov_HH=cov_hh
    ))

    return net


@pytest.fixture
def no_fixed_network() -> Network:
    """Network with no fixed 3D point (should fail validation)."""
    points = {
        "A": Point(id="A", name="Point A", easting=0, northing=0, height=100.0,
                   fixed_easting=False, fixed_northing=False, fixed_height=False),
        "B": Point(id="B", name="Point B", easting=100, northing=0, height=100.0,
                   fixed_easting=False, fixed_northing=False, fixed_height=False),
    }
    net = Network(name="No Fixed", points=points)
    net.add_observation(GnssBaselineObservation(
        id="BL1", obs_type=None, value=0.0, sigma=1.0,
        from_point_id="A", to_point_id="B",
        dE=100.0, dN=0.0, dH=0.0,
        cov_EE=0.000001, cov_EN=0.0, cov_EH=0.0,
        cov_NN=0.000001, cov_NH=0.0, cov_HH=0.000001
    ))
    return net


@pytest.fixture
def partial_fixed_network() -> Network:
    """Network with only E,N fixed (no height datum)."""
    points = {
        "A": Point(id="A", name="Point A", easting=0, northing=0, height=100.0,
                   fixed_easting=True, fixed_northing=True, fixed_height=False),
        "B": Point(id="B", name="Point B", easting=100, northing=0, height=100.0,
                   fixed_easting=False, fixed_northing=False, fixed_height=False),
    }
    net = Network(name="Partial Fixed", points=points)
    net.add_observation(GnssBaselineObservation(
        id="BL1", obs_type=None, value=0.0, sigma=1.0,
        from_point_id="A", to_point_id="B",
        dE=100.0, dN=0.0, dH=0.0,
        cov_EE=0.000001, cov_EN=0.0, cov_EH=0.0,
        cov_NN=0.000001, cov_NH=0.0, cov_HH=0.000001
    ))
    return net


# ---------------------------------------------------------------------------
# GnssBaselineObservation Tests
# ---------------------------------------------------------------------------

class TestGnssBaselineObservation:
    """Tests for GnssBaselineObservation class."""

    def test_creation(self):
        """Test basic creation of GNSS baseline observation."""
        obs = GnssBaselineObservation(
            id="BL1",
            obs_type=None,
            value=0.0,
            sigma=1.0,
            from_point_id="A",
            to_point_id="B",
            dE=100.5,
            dN=50.2,
            dH=5.1,
            cov_EE=0.000004,
            cov_EN=0.0000005,
            cov_EH=0.0000003,
            cov_NN=0.000004,
            cov_NH=0.0000004,
            cov_HH=0.000009,
        )
        assert obs.id == "BL1"
        assert obs.obs_type == ObservationType.GNSS_BASELINE
        assert obs.dE == 100.5
        assert obs.dN == 50.2
        assert obs.dH == 5.1
        assert obs.cov_EE == 0.000004
        assert obs.from_point_id == "A"
        assert obs.to_point_id == "B"
        assert obs.enabled is True

    def test_covariance_matrix(self):
        """Test covariance matrix retrieval."""
        obs = GnssBaselineObservation(
            id="BL1", obs_type=None, value=0.0, sigma=1.0,
            from_point_id="A", to_point_id="B",
            dE=100.0, dN=50.0, dH=5.0,
            cov_EE=0.000004, cov_EN=0.0000005, cov_EH=0.0000003,
            cov_NN=0.000004, cov_NH=0.0000004, cov_HH=0.000009,
        )
        cov = obs.covariance_matrix
        # Returns a 3x3 list of lists
        assert len(cov) == 3
        assert len(cov[0]) == 3
        assert cov[0][0] == 0.000004  # cov_EE
        assert cov[1][1] == 0.000004  # cov_NN
        assert cov[2][2] == 0.000009  # cov_HH
        assert cov[0][1] == cov[1][0]  # symmetric
        assert cov[0][2] == cov[2][0]  # symmetric
        assert cov[1][2] == cov[2][1]  # symmetric

    def test_baseline_length(self):
        """Test baseline length computation."""
        obs = GnssBaselineObservation(
            id="BL1", obs_type=None, value=0.0, sigma=1.0,
            from_point_id="A", to_point_id="B",
            dE=30.0, dN=40.0, dH=0.0,
            cov_EE=0.000001, cov_EN=0.0, cov_EH=0.0,
            cov_NN=0.000001, cov_NH=0.0, cov_HH=0.000001,
        )
        # 3D length = sqrt(30^2 + 40^2 + 0^2) = 50
        assert abs(obs.baseline_length - 50.0) < 1e-9

    def test_to_dict(self):
        """Test serialization to dict."""
        obs = GnssBaselineObservation(
            id="BL1", obs_type=None, value=0.0, sigma=1.0,
            from_point_id="X", to_point_id="Y",
            dE=10.0, dN=20.0, dH=3.0,
            cov_EE=0.000004, cov_EN=0.0000005, cov_EH=0.0,
            cov_NN=0.000004, cov_NH=0.0, cov_HH=0.000009,
        )
        d = obs.to_dict()
        assert d["id"] == "BL1"
        assert d["obs_type"] == "gnss_baseline"
        assert d["from_point_id"] == "X"
        assert d["to_point_id"] == "Y"
        assert d["dE"] == 10.0
        assert d["dN"] == 20.0
        assert d["dH"] == 3.0
        assert d["cov_EE"] == 0.000004

    def test_from_dict(self):
        """Test deserialization from dict."""
        d = {
            "id": "BL2",
            "obs_type": "gnss_baseline",
            "value": 0.0,
            "sigma": 1.0,
            "from_point_id": "P1",
            "to_point_id": "P2",
            "dE": 100.0,
            "dN": 50.0,
            "dH": 5.0,
            "cov_EE": 0.000004,
            "cov_EN": 0.0,
            "cov_EH": 0.0,
            "cov_NN": 0.000004,
            "cov_NH": 0.0,
            "cov_HH": 0.000009,
            "enabled": True,
        }
        obs = GnssBaselineObservation.from_dict(d)
        assert obs.id == "BL2"
        assert obs.dE == 100.0
        assert obs.from_point_id == "P1"
        assert obs.to_point_id == "P2"
        assert obs.cov_EE == 0.000004


# ---------------------------------------------------------------------------
# Network 3D Validation Tests
# ---------------------------------------------------------------------------

class TestNetwork3DValidation:
    """Tests for Network 3D validation methods."""

    def test_get_gnss_baseline_observations(self, simple_gnss_network):
        """Test retrieval of GNSS baseline observations."""
        obs = simple_gnss_network.get_gnss_baseline_observations()
        assert len(obs) == 3
        assert all(isinstance(o, GnssBaselineObservation) for o in obs)

    def test_get_3d_fixed_points(self, simple_gnss_network):
        """Test retrieval of fully fixed 3D points."""
        fixed = simple_gnss_network.get_3d_fixed_points()
        assert len(fixed) == 1
        assert fixed[0].id == "BASE"

    def test_get_3d_dof(self, simple_gnss_network):
        """Test degrees of freedom calculation for 3D GNSS.

        3 baselines x 3 components = 9 observations
        2 free points x 3 coords = 6 unknowns
        DOF = 9 - 6 = 3
        """
        dof = simple_gnss_network.get_3d_degrees_of_freedom()
        assert dof == 3

    def test_validate_3d_success(self, simple_gnss_network):
        """Test successful 3D validation."""
        errors = simple_gnss_network.validate_3d()
        assert len(errors) == 0

    def test_validate_3d_no_fixed(self, no_fixed_network):
        """Test 3D validation fails with no fixed 3D point."""
        errors = no_fixed_network.validate_3d()
        assert len(errors) > 0
        # Should mention missing E, N, or H datum
        error_text = " ".join(errors).lower()
        assert "easting" in error_text or "northing" in error_text or "height" in error_text or "datum" in error_text

    def test_validate_3d_partial_fixed(self, partial_fixed_network):
        """Test 3D validation fails with no height datum."""
        errors = partial_fixed_network.validate_3d()
        assert len(errors) > 0
        # Should mention missing height datum
        error_text = " ".join(errors).lower()
        assert "height" in error_text

    def test_validate_3d_missing_point(self):
        """Test 3D validation fails when observation references missing point."""
        points = {
            "A": Point(id="A", name="A", easting=0, northing=0, height=100.0,
                       fixed_easting=True, fixed_northing=True, fixed_height=True),
        }
        net = Network(name="Missing", points=points)
        net.add_observation(GnssBaselineObservation(
            id="BL1", obs_type=None, value=0.0, sigma=1.0,
            from_point_id="A", to_point_id="MISSING",
            dE=100.0, dN=0.0, dH=0.0,
            cov_EE=0.000001, cov_EN=0.0, cov_EH=0.0,
            cov_NN=0.000001, cov_NH=0.0, cov_HH=0.000001
        ))
        errors = net.validate_3d()
        assert len(errors) > 0


# ---------------------------------------------------------------------------
# 3D Solver Tests
# ---------------------------------------------------------------------------

class TestGnss3DSolver:
    """Tests for the 3D GNSS least-squares solver."""

    def test_adjustment_success(self, simple_gnss_network):
        """Test successful GNSS adjustment."""
        options = AdjustmentOptions(compute_reliability=True)
        result = adjust_gnss_3d(simple_gnss_network, options)

        assert result.success is True
        assert result.converged is True
        assert result.iterations == 1  # Linear problem
        assert result.degrees_of_freedom == 3

    def test_adjusted_coordinates(self, simple_gnss_network):
        """Test adjusted coordinate values are reasonable."""
        result = adjust_gnss_3d(simple_gnss_network)

        # BASE is fixed
        base = result.adjusted_points["BASE"]
        assert base.easting == 500000.0
        assert base.northing == 4000000.0
        assert base.height == 100.0
        assert base.fixed_easting is True
        assert base.fixed_northing is True
        assert base.fixed_height is True

        # P1 should be approximately BASE + (100, 50, 5)
        p1 = result.adjusted_points["P1"]
        assert p1.easting is not None
        assert abs(p1.easting - 500100.0) < 0.01
        assert abs(p1.northing - 4000050.0) < 0.01
        assert abs(p1.height - 105.0) < 0.01

        # P2 should be approximately BASE + (200, 100, 10)
        p2 = result.adjusted_points["P2"]
        assert abs(p2.easting - 500200.0) < 0.01
        assert abs(p2.northing - 4000100.0) < 0.01
        assert abs(p2.height - 110.0) < 0.01

    def test_variance_factor(self, simple_gnss_network):
        """Test variance factor is computed."""
        result = adjust_gnss_3d(simple_gnss_network)

        assert result.variance_factor is not None
        assert result.variance_factor > 0

    def test_residuals_populated(self, simple_gnss_network):
        """Test that residuals are populated."""
        result = adjust_gnss_3d(simple_gnss_network)

        # 3 baselines
        assert len(result.residual_details) == 3

        for r in result.residual_details:
            assert r.obs_type == "gnss_baseline"
            assert r.residual is not None  # 3D magnitude
            assert r.standardized_residual is not None

    def test_chi_square_test(self, simple_gnss_network):
        """Test chi-square global test is computed."""
        result = adjust_gnss_3d(simple_gnss_network)

        assert result.chi_square_test is not None
        assert result.chi_square_test.degrees_of_freedom == 3

    def test_reliability_measures(self, simple_gnss_network):
        """Test reliability measures are computed when requested."""
        options = AdjustmentOptions(compute_reliability=True)
        result = adjust_gnss_3d(simple_gnss_network, options)

        # Check that redundancy info is present
        for r in result.residual_details:
            assert r.redundancy_number is not None

    def test_sigma_coords_computed(self, simple_gnss_network):
        """Test that sigma values are computed for free points."""
        result = adjust_gnss_3d(simple_gnss_network)

        # Fixed point has zero sigma
        base = result.adjusted_points["BASE"]
        assert base.sigma_easting == 0.0
        assert base.sigma_northing == 0.0
        assert base.sigma_height == 0.0

        # Free points should have non-zero sigma
        p1 = result.adjusted_points["P1"]
        assert p1.sigma_easting is not None and p1.sigma_easting > 0
        assert p1.sigma_northing is not None and p1.sigma_northing > 0
        assert p1.sigma_height is not None and p1.sigma_height > 0

    def test_error_ellipses_computed(self, simple_gnss_network):
        """Test that horizontal error ellipses are computed for free points."""
        result = adjust_gnss_3d(simple_gnss_network)

        # Should have ellipses for P1 and P2 (free points)
        assert "P1" in result.error_ellipses
        assert "P2" in result.error_ellipses

        # BASE is fully fixed, should not have ellipse
        assert "BASE" not in result.error_ellipses

        # Check ellipse properties
        e1 = result.error_ellipses["P1"]
        assert e1.semi_major > 0
        assert e1.semi_minor > 0
        assert e1.semi_major >= e1.semi_minor

    def test_no_fixed_fails(self, no_fixed_network):
        """Test that adjustment fails when no fixed 3D point exists."""
        result = adjust_gnss_3d(no_fixed_network)

        assert result.success is False
        error_lower = result.error_message.lower()
        assert "datum" in error_lower or "fixed" in error_lower or "easting" in error_lower

    def test_partial_fixed_fails(self, partial_fixed_network):
        """Test that adjustment fails when height datum is missing."""
        result = adjust_gnss_3d(partial_fixed_network)

        assert result.success is False
        assert "height" in result.error_message.lower()

    def test_just_determined_system(self):
        """Test DOF=0 case (just-determined system)."""
        # 1 baseline, 1 free point -> 3 obs, 3 unknowns -> DOF=0
        points = {
            "BASE": Point(id="BASE", name="Base", easting=0, northing=0, height=0,
                          fixed_easting=True, fixed_northing=True, fixed_height=True),
            "P1": Point(id="P1", name="P1", easting=100, northing=50, height=5,
                        fixed_easting=False, fixed_northing=False, fixed_height=False),
        }
        net = Network(name="Just Determined", points=points)
        net.add_observation(GnssBaselineObservation(
            id="BL1", obs_type=None, value=0.0, sigma=1.0,
            from_point_id="BASE", to_point_id="P1",
            dE=100.0, dN=50.0, dH=5.0,
            cov_EE=0.000004, cov_EN=0.0, cov_EH=0.0,
            cov_NN=0.000004, cov_NH=0.0, cov_HH=0.000009
        ))

        result = adjust_gnss_3d(net)
        assert result.success is True
        assert result.degrees_of_freedom == 0


# ---------------------------------------------------------------------------
# HTML Report Tests
# ---------------------------------------------------------------------------

class TestGnssHTMLReport:
    """Tests for GNSS-specific HTML report generation."""

    def test_is_gnss_result_detection(self, simple_gnss_network):
        """Test that GNSS results are correctly detected."""
        result = adjust_gnss_3d(simple_gnss_network)
        assert _is_gnss_result(result) is True

    def test_gnss_report_contains_3d_coords(self, simple_gnss_network):
        """Test that GNSS report shows E, N, H columns."""
        result = adjust_gnss_3d(simple_gnss_network)
        html = render_html_report(result)

        # Should contain 3D coordinate headers
        assert "σE" in html or "σN" in html or "σH" in html

    def test_gnss_report_contains_baseline_residuals(self, simple_gnss_network):
        """Test that GNSS report shows baseline residual table."""
        result = adjust_gnss_3d(simple_gnss_network)
        html = render_html_report(result)

        # Should contain baseline-specific columns
        assert "Length" in html or "w_max" in html

    def test_gnss_report_title(self, simple_gnss_network):
        """Test that GNSS report has appropriate title."""
        result = adjust_gnss_3d(simple_gnss_network)
        html = render_html_report(result)

        assert "GNSS" in html


# ---------------------------------------------------------------------------
# CSV Parsing Tests
# ---------------------------------------------------------------------------

class TestGnssCSVParsing:
    """Tests for GNSS CSV parsing functions."""

    def test_parse_gnss_points_csv(self, data_dir):
        """Test parsing GNSS points from CSV."""
        csv_path = data_dir / "example_gnss_points.csv"
        points = parse_gnss_points_csv(csv_path)

        assert len(points) == 4
        assert "BASE" in points
        assert "P1" in points
        assert "P2" in points
        assert "P3" in points

        # Check BASE point
        base = points["BASE"]
        assert base.easting == 500000.0
        assert base.northing == 4000000.0
        assert base.height == 100.0
        assert base.fixed_easting is True
        assert base.fixed_northing is True
        assert base.fixed_height is True

        # Check P1 point (free)
        p1 = points["P1"]
        assert p1.fixed_easting is False
        assert p1.fixed_northing is False
        assert p1.fixed_height is False

    def test_parse_gnss_baselines_csv_full(self, data_dir):
        """Test parsing GNSS baselines with full covariance format."""
        csv_path = data_dir / "example_gnss_baselines.csv"
        obs_list = parse_gnss_baselines_csv(csv_path, covariance_format="full")

        assert len(obs_list) == 6
        assert all(isinstance(o, GnssBaselineObservation) for o in obs_list)

        # Find BL1 observation
        bl1 = next(o for o in obs_list if o.id == "BL1")
        assert bl1.from_point_id == "BASE"
        assert bl1.to_point_id == "P1"
        assert bl1.dE == 100.002
        assert bl1.dN == 50.001
        assert bl1.dH == 5.001
        assert bl1.cov_EE == 0.000004

    def test_parse_gnss_baselines_csv_sigmas(self, data_dir):
        """Test parsing GNSS baselines with sigmas+correlations format."""
        csv_path = data_dir / "example_gnss_baselines_sigmas.csv"
        obs_list = parse_gnss_baselines_csv(csv_path, covariance_format="sigmas_corr")

        assert len(obs_list) == 3
        assert all(isinstance(o, GnssBaselineObservation) for o in obs_list)

        # Find BL1 observation
        bl1 = next(o for o in obs_list if o.id == "BL1")
        assert bl1.from_point_id == "BASE"
        assert bl1.to_point_id == "P1"
        assert bl1.dE == 100.002

        # Check converted covariance: cov_EE = sigma_E^2 = 0.002^2 = 0.000004
        assert abs(bl1.cov_EE - 0.000004) < 1e-9
        assert abs(bl1.cov_NN - 0.000004) < 1e-9
        assert abs(bl1.cov_HH - 0.000009) < 1e-9

        # Check correlation conversion: cov_EN = rho_EN * sigma_E * sigma_N
        # rho_EN = 0.05, sigma_E = sigma_N = 0.002
        # cov_EN = 0.05 * 0.002 * 0.002 = 0.0000002
        assert abs(bl1.cov_EN - 0.0000002) < 1e-10

    def test_full_adjustment_from_csv(self, data_dir):
        """Test full adjustment workflow from CSV files."""
        points = parse_gnss_points_csv(data_dir / "example_gnss_points.csv")
        obs_list = parse_gnss_baselines_csv(data_dir / "example_gnss_baselines.csv", covariance_format="full")

        net = Network(name="From CSV", points=points)
        for obs in obs_list:
            net.add_observation(obs)

        result = adjust_gnss_3d(net)

        assert result.success is True
        # DOF = 6 baselines * 3 - 3 free points * 3 = 18 - 9 = 9
        assert result.degrees_of_freedom == 9
        assert len(result.adjusted_points) == 4


# ---------------------------------------------------------------------------
# Serialization Tests
# ---------------------------------------------------------------------------

class TestGnssResultSerialization:
    """Tests for serialization of GNSS adjustment results."""

    def test_to_dict(self, simple_gnss_network):
        """Test AdjustmentResult.to_dict() for GNSS results."""
        result = adjust_gnss_3d(simple_gnss_network)
        d = result.to_dict()

        # Check nested structure
        assert d["adjustment"]["success"] is True
        assert d["adjustment"]["degrees_of_freedom"] == 3
        assert "adjusted_points" in d
        assert "residuals" in d

        # Check that 3D coords are in adjusted points
        adj_pts = d["adjusted_points"]
        if isinstance(adj_pts, dict):
            for pt_data in adj_pts.values():
                assert "easting" in pt_data
                assert "northing" in pt_data
                assert "height" in pt_data

    def test_to_json(self, simple_gnss_network):
        """Test AdjustmentResult.to_json() for GNSS results."""
        result = adjust_gnss_3d(simple_gnss_network)
        json_str = result.to_json()

        assert isinstance(json_str, str)
        assert "gnss_baseline" in json_str
        assert "height" in json_str
