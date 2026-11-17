"""
Robust estimation in the 2D, 3D GNSS, and mixed solvers.

Each test builds a well-determined network with small random noise, adds one
gross error, and checks that every robust method converges, gives the
blundered observation the smallest weight, and recovers the coordinates far
better than ordinary least squares (regression tests for issue #16).
"""

import math

import numpy as np
import pytest

from survey_adjustment.core.models.point import Point
from survey_adjustment.core.models.observation import (
    ObservationType,
    DistanceObservation,
    DirectionObservation,
    HeightDifferenceObservation,
    GnssBaselineObservation,
)
from survey_adjustment.core.models.network import Network
from survey_adjustment.core.models.options import AdjustmentOptions, RobustEstimator
from survey_adjustment.core.solver.geometry import azimuth
from survey_adjustment.core.solver.least_squares_2d import adjust_network_2d
from survey_adjustment.core.solver.least_squares_3d import adjust_gnss_3d
from survey_adjustment.core.solver.least_squares_mixed import adjust_network_mixed

ESTIMATORS = [RobustEstimator.HUBER, RobustEstimator.DANISH, RobustEstimator.IGG3]

TRUTH = {
    "A": (0.0, 0.0, 100.0),
    "B": (400.0, 0.0, 102.0),
    "C": (100.0, 300.0, 105.0),
    "D": (350.0, 320.0, 103.0),
    "E": (220.0, 150.0, 101.5),
}
FIXED = ("A", "B")
FREE = ("C", "D", "E")
SIGMA_DIST = 0.002
SIGMA_DIR = math.radians(2 / 3600)
SIGMA_LEV = 0.001


def _add_points(network, dims):
    for pid, (e, n, h) in TRUTH.items():
        fixed = pid in FIXED
        if fixed:
            point = Point(id=pid, name=pid, easting=e, northing=n, height=h,
                          fixed_easting=True, fixed_northing=True, fixed_height=True)
        else:
            point = Point(id=pid, name=pid, easting=e + 0.2, northing=n - 0.3,
                          height=h + 0.1 if dims == 3 else h)
        network.add_point(point)


def _add_classical(network, rng, blunder_id=None, blunder=0.0):
    pairs = [(f, t) for i, f in enumerate(TRUTH) for t in list(TRUTH)[i + 1:]
             if not (f in FIXED and t in FIXED)]
    for k, (f, t) in enumerate(pairs, 1):
        (e1, n1, _), (e2, n2, _) = TRUTH[f], TRUTH[t]
        obs_id = f"D{k}"
        value = math.hypot(e2 - e1, n2 - n1) + rng.normal(0, SIGMA_DIST / 2)
        value += blunder if obs_id == blunder_id else 0.0
        network.add_observation(DistanceObservation(
            id=obs_id, obs_type=ObservationType.DISTANCE, value=value, sigma=SIGMA_DIST,
            from_point_id=f, to_point_id=t))
    for station, orientation in (("A", 0.4), ("B", 2.9), ("E", 5.1)):
        for t in TRUTH:
            if t == station:
                continue
            (e1, n1, _), (e2, n2, _) = TRUTH[station], TRUTH[t]
            value = (azimuth(e1, n1, e2, n2) - orientation) % (2 * math.pi)
            network.add_observation(DirectionObservation(
                id=f"R{station}{t}", obs_type=ObservationType.DIRECTION,
                value=value + rng.normal(0, SIGMA_DIR / 2), sigma=SIGMA_DIR,
                from_point_id=station, to_point_id=t, set_id=f"SET_{station}"))


def _max_error(result, dims):
    return max(
        np.max(np.abs(np.array([getattr(result.adjusted_points[pid], c)
                                for c in ("easting", "northing", "height")[:dims]])
                      - np.array(TRUTH[pid][:dims])))
        for pid in FREE)


def _check(network_factory, solver, blunder_id, dims, estimator):
    ls = solver(network_factory())
    robust = solver(network_factory(), AdjustmentOptions(robust_estimator=estimator))
    assert ls.success and robust.success

    assert robust.robust_converged, robust.robust_message
    weights = {r.obs_id: r.weight_factor for r in robust.residual_details}
    assert min(weights, key=weights.get) == blunder_id

    ls_error, robust_error = _max_error(ls, dims), _max_error(robust, dims)
    assert robust_error < ls_error / 3
    return robust_error


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_2d_isolates_distance_blunder(estimator):
    def build():
        network = Network(name="2d robust")
        _add_points(network, dims=2)
        _add_classical(network, np.random.default_rng(7), blunder_id="D6", blunder=0.05)
        return network

    error = _check(build, adjust_network_2d, "D6", dims=2, estimator=estimator)
    assert error < 0.003


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_3d_isolates_baseline_blunder(estimator):
    def build():
        rng = np.random.default_rng(11)
        network = Network(name="3d robust")
        _add_points(network, dims=3)
        pairs = [("A", "C"), ("A", "E"), ("B", "D"), ("B", "E"), ("C", "D"),
                 ("C", "E"), ("D", "E"), ("A", "D")]
        for k, (f, t) in enumerate(pairs, 1):
            vector = np.subtract(TRUTH[t], TRUTH[f]) + rng.normal(0, 0.001, 3)
            if k == 5:
                vector[0] += 0.05
            network.add_observation(GnssBaselineObservation.from_sigmas_correlations(
                f, t, *vector, 0.002, 0.002, 0.004, rho_EN=0.2, obs_id=f"G{k}"))
        return network

    error = _check(build, adjust_gnss_3d, "G5", dims=3, estimator=estimator)
    assert error < 0.003


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_mixed_isolates_leveling_blunder(estimator):
    def build():
        rng = np.random.default_rng(3)
        network = Network(name="mixed robust")
        _add_points(network, dims=3)
        _add_classical(network, rng)
        lines = [(f, t) for i, f in enumerate(TRUTH) for t in list(TRUTH)[i + 1:]
                 if not (f in FIXED and t in FIXED)]
        for k, (f, t) in enumerate(lines, 1):
            value = TRUTH[t][2] - TRUTH[f][2] + rng.normal(0, SIGMA_LEV / 2)
            if k == 4:
                value += 0.03
            network.add_observation(HeightDifferenceObservation(
                id=f"L{k}", obs_type=ObservationType.HEIGHT_DIFF, value=value, sigma=SIGMA_LEV,
                from_point_id=f, to_point_id=t))
        return network

    error = _check(build, adjust_network_mixed, "L4", dims=3, estimator=estimator)
    assert error < 0.003
