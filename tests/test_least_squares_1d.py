"""
Tests for the 1D leveling least-squares adjustment.

These tests verify:
- Closed-form results (weighted mean, loop misclosure distribution)
- Residual sign convention (v = observed - computed)
- Covariance scaling and redundancy numbers
- Datum handling (missing fixed height, auto-datum)
"""

import math

import pytest

from survey_adjustment.core.models.point import Point
from survey_adjustment.core.models.observation import (
    ObservationType,
    HeightDifferenceObservation,
)
from survey_adjustment.core.models.network import Network
from survey_adjustment.core.models.options import AdjustmentOptions
from survey_adjustment.core.solver.least_squares_1d import adjust_leveling_1d


def _point(pid, height, fixed=False):
    return Point(id=pid, name=pid, easting=0.0, northing=0.0,
                 height=height, fixed_height=fixed)


def _dh(obs_id, from_id, to_id, value, sigma):
    return HeightDifferenceObservation(
        id=obs_id, obs_type=ObservationType.HEIGHT_DIFF,
        value=value, sigma=sigma,
        from_point_id=from_id, to_point_id=to_id,
    )


def _residual(result, obs_id):
    return next(r for r in result.residual_details if r.obs_id == obs_id)


@pytest.fixture
def weighted_mean_network():
    """Two height differences A->B with different precision (A fixed)."""
    network = Network(name="weighted mean")
    network.add_point(_point("A", 100.0, fixed=True))
    network.add_point(_point("B", 100.5))  # deliberately poor approximation
    network.add_observation(_dh("L1", "A", "B", 1.000, 0.001))
    network.add_observation(_dh("L2", "A", "B", 1.006, 0.002))
    return network


class TestWeightedMean:
    """A single unknown observed twice reduces to a weighted mean."""

    def test_adjusted_height_is_weighted_mean(self, weighted_mean_network):
        result = adjust_leveling_1d(weighted_mean_network)
        assert result.success

        p1, p2 = 1 / 0.001 ** 2, 1 / 0.002 ** 2
        expected_dh = (p1 * 1.000 + p2 * 1.006) / (p1 + p2)
        assert result.adjusted_points["B"].height == pytest.approx(100.0 + expected_dh, abs=1e-10)

    def test_fixed_height_unchanged(self, weighted_mean_network):
        result = adjust_leveling_1d(weighted_mean_network)
        assert result.adjusted_points["A"].height == pytest.approx(100.0, abs=1e-12)

    def test_residual_sign_convention(self, weighted_mean_network):
        """v = observed - computed, where computed = H_to - H_from."""
        result = adjust_leveling_1d(weighted_mean_network)
        adjusted_dh = result.adjusted_points["B"].height - 100.0
        assert _residual(result, "L1").residual == pytest.approx(1.000 - adjusted_dh, abs=1e-10)
        assert _residual(result, "L2").residual == pytest.approx(1.006 - adjusted_dh, abs=1e-10)

    def test_degrees_of_freedom(self, weighted_mean_network):
        result = adjust_leveling_1d(weighted_mean_network)
        assert result.degrees_of_freedom == 1

    def test_variance_factor(self, weighted_mean_network):
        """sigma0^2 = v'Pv / r."""
        result = adjust_leveling_1d(weighted_mean_network)
        v1 = _residual(result, "L1").residual
        v2 = _residual(result, "L2").residual
        vtpv = v1 ** 2 / 0.001 ** 2 + v2 ** 2 / 0.002 ** 2
        assert result.variance_factor == pytest.approx(vtpv / 1, rel=1e-9)

    def test_covariance_is_scaled_inverse_normal_matrix(self, weighted_mean_network):
        """Sigma_x = sigma0^2 * N^-1, with N = sum of weights."""
        result = adjust_leveling_1d(weighted_mean_network)
        n_inv = 1 / (1 / 0.001 ** 2 + 1 / 0.002 ** 2)
        expected_var = result.variance_factor * n_inv
        assert result.point_covariances["B"][0][0] == pytest.approx(expected_var, rel=1e-9)
        assert result.adjusted_points["B"].sigma_height == pytest.approx(
            math.sqrt(expected_var), rel=1e-9)

    def test_redundancy_numbers(self, weighted_mean_network):
        """r_i = 1 - p_i / sum(p); the more precise observation is less controlled."""
        result = adjust_leveling_1d(weighted_mean_network)
        p1, p2 = 1 / 0.001 ** 2, 1 / 0.002 ** 2
        assert _residual(result, "L1").redundancy_number == pytest.approx(1 - p1 / (p1 + p2), abs=1e-9)
        assert _residual(result, "L2").redundancy_number == pytest.approx(1 - p2 / (p1 + p2), abs=1e-9)


class TestLevelingLoop:
    """A closed loop distributes its misclosure over the observations."""

    def test_equal_weights_share_misclosure_equally(self):
        network = Network(name="loop")
        network.add_point(_point("A", 100.0, fixed=True))
        network.add_point(_point("B", 101.0))
        network.add_point(_point("C", 102.0))
        observed = {"L1": ("A", "B", 1.000), "L2": ("B", "C", 1.000), "L3": ("C", "A", -1.994)}
        for obs_id, (f, t, value) in observed.items():
            network.add_observation(_dh(obs_id, f, t, value, 0.001))

        result = adjust_leveling_1d(network)
        assert result.success

        misclosure = sum(value for _, _, value in observed.values())
        for obs_id in observed:
            assert _residual(result, obs_id).residual == pytest.approx(misclosure / 3, abs=1e-10)

        # The adjusted loop closes exactly.
        h = {pid: result.adjusted_points[pid].height for pid in "ABC"}
        assert (h["B"] - h["A"]) + (h["C"] - h["B"]) + (h["A"] - h["C"]) == pytest.approx(0.0, abs=1e-12)


class TestNetworkProperties:
    """Properties that hold for any leveling network."""

    @pytest.fixture
    def network(self):
        truth = {"A": 100.0, "B": 101.234, "C": 99.876, "D": 102.5, "E": 100.75}
        noise = {"L1": 0.0004, "L2": -0.0007, "L3": 0.0011, "L4": -0.0002,
                 "L5": 0.0006, "L6": -0.0009, "L7": 0.0003}
        lines = {"L1": ("A", "B"), "L2": ("B", "C"), "L3": ("C", "D"), "L4": ("D", "E"),
                 "L5": ("E", "A"), "L6": ("A", "C"), "L7": ("B", "D")}
        network = Network(name="properties")
        network.add_point(_point("A", truth["A"], fixed=True))
        for pid in "BCDE":
            network.add_point(_point(pid, truth[pid] + 0.3))
        for obs_id, (f, t) in lines.items():
            network.add_observation(
                _dh(obs_id, f, t, truth[t] - truth[f] + noise[obs_id], 0.001))
        return network

    def test_redundancy_numbers_sum_to_degrees_of_freedom(self, network):
        result = adjust_leveling_1d(network)
        assert result.success
        assert result.degrees_of_freedom == 7 - 4
        total = sum(r.redundancy_number for r in result.residual_details)
        assert total == pytest.approx(result.degrees_of_freedom, abs=1e-9)

    def test_redundancy_numbers_in_unit_interval(self, network):
        result = adjust_leveling_1d(network)
        for r in result.residual_details:
            assert 0.0 <= r.redundancy_number <= 1.0

    def test_exact_observations_give_zero_residuals(self):
        truth = {"A": 50.0, "B": 51.5, "C": 49.25}
        network = Network(name="exact")
        network.add_point(_point("A", truth["A"], fixed=True))
        network.add_point(_point("B", 0.0))
        network.add_point(_point("C", 0.0))
        for obs_id, (f, t) in {"L1": ("A", "B"), "L2": ("B", "C"), "L3": ("C", "A")}.items():
            network.add_observation(_dh(obs_id, f, t, truth[t] - truth[f], 0.002))

        result = adjust_leveling_1d(network)
        assert result.success
        for pid, height in truth.items():
            assert result.adjusted_points[pid].height == pytest.approx(height, abs=1e-10)
        for r in result.residual_details:
            assert r.residual == pytest.approx(0.0, abs=1e-10)


class TestDatum:
    """A leveling network needs at least one fixed height."""

    @pytest.fixture
    def free_network(self):
        network = Network(name="no datum")
        network.add_point(_point("A", 100.0))
        network.add_point(_point("B", 101.0))
        network.add_observation(_dh("L1", "A", "B", 1.0, 0.001))
        network.add_observation(_dh("L2", "A", "B", 1.002, 0.001))
        return network

    def test_missing_datum_fails_gracefully(self, free_network):
        result = adjust_leveling_1d(free_network)
        assert not result.success
        assert result.error_message

    def test_auto_datum_fixes_one_height(self, free_network):
        result = adjust_leveling_1d(free_network, AdjustmentOptions(auto_datum=True))
        assert result.success
        assert len(result.applied_auto_constraints) == 1
        dh = result.adjusted_points["B"].height - result.adjusted_points["A"].height
        assert dh == pytest.approx(1.001, abs=1e-10)
