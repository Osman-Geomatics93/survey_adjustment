import math
from pathlib import Path

import pytest

from survey_adjustment.core.models.network import Network
from survey_adjustment.core.models.point import Point
from survey_adjustment.core.models.observation import DistanceObservation
from survey_adjustment.core.solver.least_squares_2d import adjust_network_2d
from survey_adjustment.core.models.options import AdjustmentOptions
from survey_adjustment.core.reports.html_report import render_html_report
from survey_adjustment.qgis_integration.io.observations import (
    parse_points_csv,
    parse_distances_csv,
    parse_directions_csv,
    parse_angles_csv,
    parse_traverse_file,
)


DATA_DIR = Path(__file__).parent / "data"


def test_parse_points_csv_basic():
    pts = parse_points_csv(DATA_DIR / "example_points.csv")
    assert set(pts.keys()) == {"A", "B", "C", "D", "E"}
    assert pts["A"].fixed_easting is True
    assert pts["B"].fixed_easting is False


def test_parse_distances_csv_count_and_values():
    obs = parse_distances_csv(DATA_DIR / "example_distances.csv")
    assert len(obs) == 8
    assert obs[0].from_point_id == "A"
    assert obs[0].to_point_id == "B"
    assert abs(obs[0].value - 583.095) < 1e-9


def test_parse_directions_csv_unit_conversion_degrees_to_radians():
    obs = parse_directions_csv(DATA_DIR / "example_directions.csv", direction_unit="degrees", sigma_unit="radians")
    assert len(obs) == 8
    # 36.8699 degrees is ~0.6435 rad
    assert abs(obs[0].value - math.radians(36.8699)) < 1e-9


def test_parse_angles_csv_unit_conversion_degrees_to_radians(tmp_path):
    # make a tiny angles CSV to test arcsecond sigma conversion
    p = tmp_path / "angles.csv"
    p.write_text("obs_id,at_point,from_point,to_point,angle,sigma_angle\nA1,B,A,C,90,30\n", encoding="utf-8")
    obs = parse_angles_csv(p, angle_unit="degrees", sigma_unit="arcseconds")
    assert len(obs) == 1
    assert abs(obs[0].value - math.pi / 2) < 1e-12
    # 30 arcsec in radians
    assert abs(obs[0].sigma - (30 * math.pi / (180.0 * 3600.0))) < 1e-15


def test_parse_traverse_file_builds_network():
    net = parse_traverse_file(DATA_DIR / "example_traverse.csv", angle_unit="degrees")
    assert len(net.points) == 5
    assert len(net.get_enabled_observations()) == 10


def _build_dof1_network_with_residuals() -> Network:
    # Two fixed points define datum; one unknown point C.
    pts = {
        "A": Point(id="A", name="A", easting=0.0, northing=0.0, fixed_easting=True, fixed_northing=True),
        "B": Point(id="B", name="B", easting=100.0, northing=0.0, fixed_easting=True, fixed_northing=True),
        "C": Point(id="C", name="C", easting=50.0, northing=50.0, fixed_easting=False, fixed_northing=False),
    }
    net = Network(points=pts)

    # True distances AC and BC are ~70.710678. Use slightly inconsistent values.
    net.add_observation(DistanceObservation(id="AC", obs_type=None, value=70.72, sigma=0.01, from_point_id="A", to_point_id="C"))
    net.add_observation(DistanceObservation(id="BC", obs_type=None, value=70.70, sigma=0.01, from_point_id="B", to_point_id="C"))
    # Add a redundant fixed-fixed observation to force dof=1 and non-zero vTPv.
    net.add_observation(DistanceObservation(id="AB", obs_type=None, value=100.02, sigma=0.01, from_point_id="A", to_point_id="B"))

    return net


def test_end_to_end_adjustment_has_dof1_and_nonzero_variance_factor():
    net = _build_dof1_network_with_residuals()
    options = AdjustmentOptions(max_iterations=25, convergence_threshold=1e-10)
    result = adjust_network_2d(net, options)
    assert result.success is True
    assert result.degrees_of_freedom == 1
    assert result.variance_factor > 0


def test_html_report_contains_sections():
    net = _build_dof1_network_with_residuals()
    result = adjust_network_2d(net, AdjustmentOptions(max_iterations=25, convergence_threshold=1e-10))
    html = render_html_report(result)
    assert "Adjusted Points" in html
    assert "Residuals" in html


def test_html_report_marks_flagged_rows_when_present():
    net = _build_dof1_network_with_residuals()
    result = adjust_network_2d(net, AdjustmentOptions(max_iterations=25, convergence_threshold=1e-10))
    # Force a flag for test
    if result.residual_details:
        result.residual_details[0].flagged = True
    html = render_html_report(result)
    assert "class='flag'" in html


def test_adjustment_result_timestamp_is_utc_z():
    net = _build_dof1_network_with_residuals()
    result = adjust_network_2d(net, AdjustmentOptions(max_iterations=25, convergence_threshold=1e-10))
    assert result.timestamp.endswith("Z")


def test_residualinfo_to_dict_json_safe_for_inf():
    net = _build_dof1_network_with_residuals()
    result = adjust_network_2d(net, AdjustmentOptions(max_iterations=25, convergence_threshold=1e-10))
    if result.residual_details:
        result.residual_details[0].mdb = float("inf")
        d = result.residual_details[0].to_dict()
        assert d["mdb"] is None


def test_full_workflow_traverse_file_to_html(tmp_path):
    net = parse_traverse_file(DATA_DIR / "example_traverse.csv", angle_unit="degrees")
    result = adjust_network_2d(net, AdjustmentOptions(max_iterations=50, convergence_threshold=1e-8))
    html = render_html_report(result)
    out = tmp_path / "report.html"
    out.write_text(html, encoding="utf-8")
    assert out.exists()
    assert "Survey Adjustment Report" in html
