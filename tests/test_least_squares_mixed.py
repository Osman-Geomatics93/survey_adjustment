"""
Tests for the mixed (classical + GNSS + leveling) least-squares adjustment.

These tests verify:
- Recovery of known 3D coordinates from exact mixed observations
- Degrees of freedom with orientation unknowns
- Agreement with the specialised 1D and 3D solvers on single-type networks
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
from survey_adjustment.core.solver.geometry import azimuth
from survey_adjustment.core.solver.least_squares_1d import adjust_leveling_1d
from survey_adjustment.core.solver.least_squares_3d import adjust_gnss_3d
from survey_adjustment.core.solver.least_squares_mixed import adjust_network_mixed

TRUTH = {
    "A": (0.0, 0.0, 100.0),
    "B": (300.0, 0.0, 102.0),
    "C": (150.0, 200.0, 105.0),
    "D": (320.0, 250.0, 103.0),
}
FIXED = ("A", "B")


def _xyz(point):
    return np.array([point.easting, point.northing, point.height])


def _distance(obs_id, f, t):
    (e1, n1, _), (e2, n2, _) = TRUTH[f], TRUTH[t]
    return DistanceObservation(
        id=obs_id, obs_type=ObservationType.DISTANCE,
        value=math.hypot(e2 - e1, n2 - n1), sigma=0.003,
        from_point_id=f, to_point_id=t,
    )


def _direction(obs_id, f, t, set_id, orientation):
    (e1, n1, _), (e2, n2, _) = TRUTH[f], TRUTH[t]
    value = (azimuth(e1, n1, e2, n2) - orientation) % (2 * math.pi)
    return DirectionObservation(
        id=obs_id, obs_type=ObservationType.DIRECTION,
        value=value, sigma=math.radians(2 / 3600),
        from_point_id=f, to_point_id=t, set_id=set_id,
    )


def _height_diff(obs_id, f, t):
    return HeightDifferenceObservation(
        id=obs_id, obs_type=ObservationType.HEIGHT_DIFF,
        value=TRUTH[t][2] - TRUTH[f][2], sigma=0.001,
        from_point_id=f, to_point_id=t,
    )


def _baseline(obs_id, f, t):
    vector = np.subtract(TRUTH[t], TRUTH[f])
    return GnssBaselineObservation.from_sigmas_correlations(
        f, t, *vector, 0.004, 0.004, 0.008, rho_EN=0.2, obs_id=obs_id)


@pytest.fixture
def mixed_network():
    """Exact observations of every type; free points start 0.2-0.5 m off."""
    network = Network(name="mixed")
    for pid, (e, n, h) in TRUTH.items():
        if pid in FIXED:
            network.add_point(Point(id=pid, name=pid, easting=e, northing=n, height=h,
                                    fixed_easting=True, fixed_northing=True, fixed_height=True))
        else:
            network.add_point(Point(id=pid, name=pid, easting=e + 0.3, northing=n - 0.4,
                                    height=h + 0.2))

    for i, (f, t) in enumerate([("A", "C"), ("B", "C"), ("B", "D"), ("C", "D"), ("A", "D")], 1):
        network.add_observation(_distance(f"D{i}", f, t))
    # Two setups with arbitrary (unknown) orientations.
    for i, t in enumerate("BCD", 1):
        network.add_observation(_direction(f"RA{i}", "A", t, "SET_A", orientation=0.7))
    for i, t in enumerate("ACD", 1):
        network.add_observation(_direction(f"RB{i}", "B", t, "SET_B", orientation=4.1))
    for i, (f, t) in enumerate([("A", "C"), ("C", "D"), ("B", "D")], 1):
        network.add_observation(_height_diff(f"L{i}", f, t))
    network.add_observation(_baseline("G1", "A", "D"))
    return network


class TestExactMixedNetwork:

    def test_recovers_true_coordinates(self, mixed_network):
        result = adjust_network_mixed(mixed_network)
        assert result.success, result.error_message
        assert result.converged
        for pid, xyz in TRUTH.items():
            np.testing.assert_allclose(_xyz(result.adjusted_points[pid]), xyz, atol=1e-6)

    def test_residuals_vanish(self, mixed_network):
        result = adjust_network_mixed(mixed_network)
        for r in result.residual_details:
            assert r.residual == pytest.approx(0.0, abs=1e-6)

    def test_degrees_of_freedom(self, mixed_network):
        """17 observations (5 dist, 6 dir, 3 lev, 3 GNSS components) minus
        8 unknowns (E, N, H of C and D, plus two orientations)."""
        result = adjust_network_mixed(mixed_network)
        assert result.degrees_of_freedom == 17 - 8

    def test_every_observation_reported(self, mixed_network):
        result = adjust_network_mixed(mixed_network)
        reported = {r.obs_id for r in result.residual_details}
        expected = {obs.id for obs in mixed_network.get_enabled_observations()}
        assert reported == expected


class TestAgreementWithSpecialisedSolvers:
    """On a single-type network the mixed solver must match the dedicated solver."""

    def test_gnss_only_matches_3d_solver(self):
        def build():
            network = Network(name="gnss")
            network.add_point(Point(id="A", name="A", easting=0.0, northing=0.0, height=100.0,
                                    fixed_easting=True, fixed_northing=True, fixed_height=True))
            network.add_point(Point(id="D", name="D", easting=320.5, northing=249.5, height=103.4))
            network.add_point(Point(id="C", name="C", easting=150.2, northing=200.3, height=104.9))
            baselines = {"G1": ("A", "D", 0.004), "G2": ("A", "C", -0.003),
                         "G3": ("C", "D", 0.005)}
            for obs_id, (f, t, noise) in baselines.items():
                vector = np.subtract(TRUTH[t], TRUTH[f]) + noise
                network.add_observation(GnssBaselineObservation.from_sigmas_correlations(
                    f, t, *vector, 0.004, 0.005, 0.009, rho_EN=0.25, rho_NH=-0.1, obs_id=obs_id))
            return network

        mixed = adjust_network_mixed(build())
        dedicated = adjust_gnss_3d(build())
        assert mixed.success and dedicated.success
        for pid in "ACD":
            np.testing.assert_allclose(_xyz(mixed.adjusted_points[pid]),
                                       _xyz(dedicated.adjusted_points[pid]), atol=1e-9)
        assert mixed.variance_factor == pytest.approx(dedicated.variance_factor, rel=1e-8)

    def test_leveling_only_matches_1d_solver(self):
        def build():
            network = Network(name="leveling")
            network.add_point(Point(id="A", name="A", easting=0.0, northing=0.0, height=100.0,
                                    fixed_easting=True, fixed_northing=True, fixed_height=True))
            for pid, (e, n, h) in {"B": (300.0, 0.0, 102.3), "C": (150.0, 200.0, 104.6)}.items():
                network.add_point(Point(id=pid, name=pid, easting=e, northing=n, height=h,
                                        fixed_easting=True, fixed_northing=True))
            lines = {"L1": ("A", "B", 2.0012), "L2": ("B", "C", 2.9979),
                     "L3": ("C", "A", -5.0021), "L4": ("A", "C", 5.0004)}
            for obs_id, (f, t, value) in lines.items():
                network.add_observation(HeightDifferenceObservation(
                    id=obs_id, obs_type=ObservationType.HEIGHT_DIFF, value=value, sigma=0.001,
                    from_point_id=f, to_point_id=t))
            return network

        mixed = adjust_network_mixed(build())
        dedicated = adjust_leveling_1d(build())
        assert mixed.success, mixed.error_message
        assert dedicated.success
        for pid in "ABC":
            assert mixed.adjusted_points[pid].height == pytest.approx(
                dedicated.adjusted_points[pid].height, abs=1e-9)
