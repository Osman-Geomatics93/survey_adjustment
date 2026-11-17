"""
Tests for the global chi-square model test in every solver.

Each network has exactly 3 degrees of freedom, so the two-sided critical values
can be checked against published chi-square table values.
"""

import pytest

from survey_adjustment.core.models.point import Point
from survey_adjustment.core.models.observation import (
    ObservationType,
    DistanceObservation,
    HeightDifferenceObservation,
    GnssBaselineObservation,
)
from survey_adjustment.core.models.network import Network
from survey_adjustment.core.models.options import AdjustmentOptions
from survey_adjustment.core.solver.least_squares_1d import adjust_leveling_1d
from survey_adjustment.core.solver.least_squares_2d import adjust_network_2d
from survey_adjustment.core.solver.least_squares_3d import adjust_gnss_3d
from survey_adjustment.core.solver.least_squares_mixed import adjust_network_mixed

# chi-square quantiles for 3 degrees of freedom (standard tables)
CRITICAL = {
    0.95: (0.2158, 9.3484),   # chi2(0.025), chi2(0.975)
    0.99: (0.0717, 12.8382),  # chi2(0.005), chi2(0.995)
}


def _fixed(pid, e, n, h=0.0):
    return Point(id=pid, name=pid, easting=e, northing=n, height=h,
                 fixed_easting=True, fixed_northing=True, fixed_height=True)


def _leveling_network():
    network = Network(name="1d")
    network.add_point(Point(id="A", name="A", easting=0.0, northing=0.0,
                            height=100.0, fixed_height=True))
    network.add_point(Point(id="B", name="B", easting=0.0, northing=0.0, height=101.0))
    network.add_point(Point(id="C", name="C", easting=0.0, northing=0.0, height=102.0))
    lines = [("A", "B", 1.0011), ("B", "C", 0.9993), ("C", "A", -2.0006),
             ("A", "B", 1.0004), ("B", "C", 1.0009)]
    for k, (f, t, value) in enumerate(lines, 1):
        network.add_observation(HeightDifferenceObservation(
            id=f"L{k}", obs_type=ObservationType.HEIGHT_DIFF, value=value, sigma=0.001,
            from_point_id=f, to_point_id=t))
    return network


def _distance_network():
    network = Network(name="2d")
    network.add_point(_fixed("A", 0.0, 0.0))
    network.add_point(_fixed("B", 100.0, 0.0))
    network.add_point(Point(id="C", name="C", easting=50.2, northing=79.7))
    distances = [("A", "C", 94.341), ("B", "C", 94.338), ("A", "C", 94.345),
                 ("B", "C", 94.343), ("A", "C", 94.339)]
    for k, (f, t, value) in enumerate(distances, 1):
        network.add_observation(DistanceObservation(
            id=f"D{k}", obs_type=ObservationType.DISTANCE, value=value, sigma=0.003,
            from_point_id=f, to_point_id=t))
    return network


def _gnss_network():
    network = Network(name="3d")
    network.add_point(_fixed("A", 0.0, 0.0, 0.0))
    network.add_point(Point(id="B", name="B", easting=100.1, northing=49.9, height=10.2))
    for k, vector in enumerate([(100.003, 50.001, 10.004), (99.998, 50.003, 9.995)], 1):
        network.add_observation(GnssBaselineObservation.from_sigmas_correlations(
            "A", "B", *vector, 0.003, 0.003, 0.006, obs_id=f"G{k}"))
    return network


CASES = [
    pytest.param(adjust_leveling_1d, _leveling_network, id="1d"),
    pytest.param(adjust_network_2d, _distance_network, id="2d"),
    pytest.param(adjust_gnss_3d, _gnss_network, id="3d"),
    pytest.param(adjust_network_mixed, _gnss_network, id="mixed"),
]


@pytest.mark.parametrize("confidence", sorted(CRITICAL))
@pytest.mark.parametrize("solver, build", CASES)
def test_critical_values_match_chi_square_table(solver, build, confidence):
    result = solver(build(), AdjustmentOptions(confidence_level=confidence))
    assert result.success, result.error_message
    assert result.degrees_of_freedom == 3

    test = result.chi_square_test
    lower, upper = CRITICAL[confidence]
    assert test.critical_lower == pytest.approx(lower, abs=1e-3)
    assert test.critical_upper == pytest.approx(upper, abs=1e-3)
    assert test.confidence_level == pytest.approx(confidence)


@pytest.mark.parametrize("solver, build", CASES)
def test_decision_matches_critical_values(solver, build):
    result = solver(build())
    test = result.chi_square_test
    assert test.passed == (test.critical_lower <= test.test_statistic <= test.critical_upper)
