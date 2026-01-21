import math

import pytest

from survey_adjustment.core.models.network import Network
from survey_adjustment.core.models.point import Point
from survey_adjustment.core.models.observation import (
    DistanceObservation,
    DirectionObservation,
    AngleObservation,
)
from survey_adjustment.core.models.options import AdjustmentOptions
from survey_adjustment.core.solver import adjust_network_2d
from survey_adjustment.core.solver.geometry import wrap_pi, wrap_2pi, azimuth


def _build_small_network() -> Network:
    """A small constrained network with redundancy.

    A, B are fixed; C, D are free.
    Mix of distance + directions + one angle.
    """
    net = Network(name="test_net")
    # Fixed control points
    net.add_point(Point("A", "A", 0.0, 0.0, fixed_easting=True, fixed_northing=True))
    net.add_point(Point("B", "B", 100.0, 0.0, fixed_easting=True, fixed_northing=True))
    # Unknowns with rough initial approximations
    net.add_point(Point("C", "C", 30.0, 35.0))
    net.add_point(Point("D", "D", 65.0, 75.0))

    # True-ish positions used to generate synthetic observations
    C_true = (30.0, 40.0)
    D_true = (60.0, 80.0)

    # Distances (add a small perturbation to create non-zero residuals)
    def dist(p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    A_xy = (0.0, 0.0)
    B_xy = (100.0, 0.0)

    net.add_observation(DistanceObservation(id="d_AC", obs_type=None, value=dist(A_xy, C_true) + 0.003, sigma=0.005, from_point_id="A", to_point_id="C"))
    net.add_observation(DistanceObservation(id="d_BC", obs_type=None, value=dist(B_xy, C_true), sigma=0.005, from_point_id="B", to_point_id="C"))
    net.add_observation(DistanceObservation(id="d_AD", obs_type=None, value=dist(A_xy, D_true), sigma=0.005, from_point_id="A", to_point_id="D"))
    net.add_observation(DistanceObservation(id="d_BD", obs_type=None, value=dist(B_xy, D_true) - 0.004, sigma=0.005, from_point_id="B", to_point_id="D"))

    # Directions from A set (orientation unknown)
    net.add_observation(DirectionObservation(id="dir_AB", obs_type=None, value=azimuth(*A_xy, *B_xy), sigma=1e-5, from_point_id="A", to_point_id="B", set_id="SET_A"))
    net.add_observation(DirectionObservation(id="dir_AC", obs_type=None, value=azimuth(*A_xy, *C_true), sigma=1e-5, from_point_id="A", to_point_id="C", set_id="SET_A"))
    net.add_observation(DirectionObservation(id="dir_AD", obs_type=None, value=azimuth(*A_xy, *D_true), sigma=1e-5, from_point_id="A", to_point_id="D", set_id="SET_A"))

    # Directions from B set
    net.add_observation(DirectionObservation(id="dir_BA", obs_type=None, value=azimuth(*B_xy, *A_xy), sigma=1e-5, from_point_id="B", to_point_id="A", set_id="SET_B"))
    net.add_observation(DirectionObservation(id="dir_BC", obs_type=None, value=azimuth(*B_xy, *C_true), sigma=1e-5, from_point_id="B", to_point_id="C", set_id="SET_B"))
    net.add_observation(DirectionObservation(id="dir_BD", obs_type=None, value=azimuth(*B_xy, *D_true), sigma=1e-5, from_point_id="B", to_point_id="D", set_id="SET_B"))

    # One angle at C: from A to B (synthetic)
    # Compute as angle at point C from ray CA to CB
    az_CA = azimuth(*C_true, *A_xy)
    az_CB = azimuth(*C_true, *B_xy)
    ang = wrap_2pi(az_CB - az_CA)
    net.add_observation(AngleObservation(id="ang_ACB", obs_type=None, value=ang, sigma=2e-5, at_point_id="C", from_point_id="A", to_point_id="B"))

    return net


def test_wrap_helpers_basic():
    assert wrap_pi(math.pi) == pytest.approx(math.pi)
    assert wrap_pi(-math.pi) == pytest.approx(math.pi)
    assert 0.0 <= wrap_2pi(-0.1) < 2 * math.pi


def test_adjust_distance_only_trilateration_converges():
    net = Network(name="tri")
    net.add_point(Point("A", "A", 0.0, 0.0, fixed_easting=True, fixed_northing=True))
    net.add_point(Point("B", "B", 100.0, 0.0, fixed_easting=True, fixed_northing=True))
    net.add_point(Point("C", "C", 40.0, 30.0))

    C_true = (30.0, 40.0)
    A_xy = (0.0, 0.0)
    B_xy = (100.0, 0.0)

    net.add_observation(DistanceObservation(id="d_AC", obs_type=None, value=math.hypot(C_true[0], C_true[1]), sigma=0.005, from_point_id="A", to_point_id="C"))
    net.add_observation(DistanceObservation(id="d_BC", obs_type=None, value=math.hypot(C_true[0]-100.0, C_true[1]-0.0), sigma=0.005, from_point_id="B", to_point_id="C"))

    res = adjust_network_2d(net, AdjustmentOptions(max_iterations=20, convergence_threshold=1e-10))
    assert res.success is True
    assert res.converged is True
    assert "C" in res.adjusted_points
    assert res.adjusted_points["C"].easting == pytest.approx(C_true[0], abs=1e-4)
    assert res.adjusted_points["C"].northing == pytest.approx(C_true[1], abs=1e-4)


def test_adjust_mixed_observations_produces_stats_and_ellipses():
    net = _build_small_network()
    opts = AdjustmentOptions(max_iterations=30, convergence_threshold=1e-10, outlier_threshold=6.0)
    res = adjust_network_2d(net, opts)
    assert res.success is True
    assert res.converged is True
    assert res.degrees_of_freedom > 0
    assert math.isfinite(res.variance_factor) and res.variance_factor > 0
    assert res.chi_square_test is not None
    assert "C" in res.error_ellipses
    assert "D" in res.error_ellipses
    assert len(res.residual_details) == len(net.get_enabled_observations())
    # No hard outliers in this synthetic dataset
    assert res.flagged_observations == []


def test_adjust_fails_without_fixed_points():
    net = Network(name="bad")
    net.add_point(Point("A", "A", 0.0, 0.0))
    net.add_point(Point("B", "B", 10.0, 0.0))
    net.add_observation(DistanceObservation(id="d", obs_type=None, value=10.0, sigma=0.01, from_point_id="A", to_point_id="B"))

    res = adjust_network_2d(net, AdjustmentOptions())
    assert res.success is False
    assert res.error_message
