"""
Tests for the 3D GNSS baseline least-squares adjustment.

These tests verify:
- Recovery of known coordinates from exact baselines
- Correct use of the full 3x3 baseline covariance (closed-form check)
- Covariance scaling and degrees of freedom
- Datum handling
"""

import numpy as np
import pytest

from survey_adjustment.core.models.point import Point
from survey_adjustment.core.models.observation import GnssBaselineObservation
from survey_adjustment.core.models.network import Network
from survey_adjustment.core.models.options import AdjustmentOptions
from survey_adjustment.core.solver.least_squares_3d import adjust_gnss_3d
from survey_adjustment.core.solver.least_squares_mixed import adjust_network_mixed


def _fixed(pid, e, n, h):
    return Point(id=pid, name=pid, easting=e, northing=n, height=h,
                 fixed_easting=True, fixed_northing=True, fixed_height=True)


def _free(pid, e, n, h):
    return Point(id=pid, name=pid, easting=e, northing=n, height=h)


def _baseline(obs_id, from_id, to_id, vector, sigmas, rho=(0.0, 0.0, 0.0)):
    return GnssBaselineObservation.from_sigmas_correlations(
        from_id, to_id, *vector, *sigmas,
        rho_EN=rho[0], rho_EH=rho[1], rho_NH=rho[2], obs_id=obs_id,
    )


def _xyz(point):
    return np.array([point.easting, point.northing, point.height])


class TestTwoBaselineIntersection:
    """One new point fixed by two correlated baselines: a matrix-weighted mean."""

    P1 = np.array([0.0, 0.0, 0.0])
    P2 = np.array([100.0, 0.0, 0.0])
    B1 = (np.array([50.0, 50.0, 10.0]), (0.005, 0.005, 0.010), (0.3, 0.0, 0.0))
    B2 = (np.array([-50.0, 50.01, 10.02]), (0.004, 0.006, 0.012), (0.0, 0.0, -0.2))

    @staticmethod
    def _covariance(sigmas, rho):
        s = np.diag(sigmas)
        r = np.array([[1.0, rho[0], rho[1]],
                      [rho[0], 1.0, rho[2]],
                      [rho[1], rho[2], 1.0]])
        return s @ r @ s

    @pytest.fixture
    def network(self):
        network = Network(name="intersection")
        network.add_point(_fixed("P1", *self.P1))
        network.add_point(_fixed("P2", *self.P2))
        network.add_point(_free("X", 50.3, 49.6, 10.4))
        network.add_observation(_baseline("B1", "P1", "X", self.B1[0], self.B1[1], self.B1[2]))
        network.add_observation(_baseline("B2", "P2", "X", self.B2[0], self.B2[1], self.B2[2]))
        return network

    def _expected(self):
        w1 = np.linalg.inv(self._covariance(self.B1[1], self.B1[2]))
        w2 = np.linalg.inv(self._covariance(self.B2[1], self.B2[2]))
        n = w1 + w2
        x = np.linalg.solve(n, w1 @ (self.P1 + self.B1[0]) + w2 @ (self.P2 + self.B2[0]))
        return x, np.linalg.inv(n)

    def test_adjusted_point_matches_closed_form(self, network):
        result = adjust_gnss_3d(network)
        assert result.success
        expected, _ = self._expected()
        np.testing.assert_allclose(_xyz(result.adjusted_points["X"]), expected, atol=1e-9)

    def test_correlations_change_the_solution(self, network):
        """Dropping the correlations must give a different point, so they are really used."""
        w1 = np.diag(1 / np.square(self.B1[1]))
        w2 = np.diag(1 / np.square(self.B2[1]))
        uncorrelated = np.linalg.solve(
            w1 + w2, w1 @ (self.P1 + self.B1[0]) + w2 @ (self.P2 + self.B2[0]))
        result = adjust_gnss_3d(network)
        assert np.max(np.abs(_xyz(result.adjusted_points["X"]) - uncorrelated)) > 1e-5

    def test_degrees_of_freedom(self, network):
        result = adjust_gnss_3d(network)
        assert result.degrees_of_freedom == 6 - 3

    def test_point_covariance_is_scaled_inverse_normal_matrix(self, network):
        result = adjust_gnss_3d(network)
        _, n_inv = self._expected()
        np.testing.assert_allclose(
            np.asarray(result.point_covariances["X"]),
            result.variance_factor * n_inv, rtol=1e-8)

    def test_fixed_points_unchanged(self, network):
        result = adjust_gnss_3d(network)
        np.testing.assert_allclose(_xyz(result.adjusted_points["P1"]), self.P1, atol=1e-12)
        np.testing.assert_allclose(_xyz(result.adjusted_points["P2"]), self.P2, atol=1e-12)

    def test_redundancy_numbers_sum_to_dof_with_correlations(self, network):
        """The per-baseline value is the mean of its three component numbers."""
        result = adjust_gnss_3d(network)
        total = sum(3 * r.redundancy_number for r in result.residual_details)
        assert total == pytest.approx(result.degrees_of_freedom, abs=1e-9)

    def test_standardized_residuals_use_a_priori_sigma0(self, network):
        """w = v / (sigma0 * sqrt(qvv)): quadrupling sigma0^2 halves w."""
        base = adjust_gnss_3d(network)
        scaled = adjust_gnss_3d(network, AdjustmentOptions(a_priori_variance=4.0))
        for b, s in zip(base.residual_details, scaled.residual_details):
            assert s.standardized_residual == pytest.approx(b.standardized_residual / 2, rel=1e-9)

    def test_standardized_residuals_match_mixed_solver(self, network):
        dedicated = adjust_gnss_3d(network)
        mixed = adjust_network_mixed(network)
        w_mixed = {r.obs_id: r.standardized_residual for r in mixed.residual_details}
        for r in dedicated.residual_details:
            assert r.standardized_residual == pytest.approx(w_mixed[r.obs_id], rel=1e-8)


class TestExactNetwork:
    """Exact baselines reproduce the true coordinates from poor approximations."""

    TRUTH = {
        "A": (1000.0, 2000.0, 50.0),
        "B": (1800.0, 2100.0, 62.5),
        "C": (1400.0, 2700.0, 48.2),
        "D": (900.0, 2500.0, 55.9),
    }
    LINES = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A"), ("A", "C"), ("B", "D")]

    @pytest.fixture
    def network(self):
        network = Network(name="exact")
        network.add_point(_fixed("A", *self.TRUTH["A"]))
        for pid in "BCD":
            e, n, h = self.TRUTH[pid]
            network.add_point(_free(pid, e + 0.4, n - 0.3, h + 0.2))
        for i, (f, t) in enumerate(self.LINES, start=1):
            vector = np.subtract(self.TRUTH[t], self.TRUTH[f])
            network.add_observation(_baseline(f"G{i}", f, t, vector, (0.003, 0.003, 0.006)))
        return network

    def test_recovers_true_coordinates(self, network):
        result = adjust_gnss_3d(network)
        assert result.success
        for pid, xyz in self.TRUTH.items():
            np.testing.assert_allclose(_xyz(result.adjusted_points[pid]), xyz, atol=1e-8)

    @pytest.mark.filterwarnings("error::RuntimeWarning")
    def test_residuals_vanish(self, network):
        result = adjust_gnss_3d(network)
        for r in result.residual_details:
            assert r.residual == pytest.approx(0.0, abs=1e-8)
            assert np.isfinite(r.standardized_residual)

    def test_degrees_of_freedom(self, network):
        result = adjust_gnss_3d(network)
        assert result.degrees_of_freedom == 3 * len(self.LINES) - 3 * 3


def test_missing_datum_fails_gracefully():
    network = Network(name="no datum")
    network.add_point(_free("A", 0.0, 0.0, 0.0))
    network.add_point(_free("B", 10.0, 0.0, 0.0))
    network.add_observation(_baseline("G1", "A", "B", (10.0, 0.0, 0.0), (0.003, 0.003, 0.006)))
    result = adjust_gnss_3d(network)
    assert not result.success
    assert result.error_message
