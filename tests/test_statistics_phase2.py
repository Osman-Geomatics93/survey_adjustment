import math

import numpy as np
import pytest

from survey_adjustment.core.statistics.distributions import normal_ppf, chi2_cdf, chi2_ppf
from survey_adjustment.core.statistics.tests import chi_square_global_test, standardized_residuals, local_outlier_threshold
from survey_adjustment.core.statistics.reliability import redundancy_numbers, mdb_values, external_reliability

from survey_adjustment.core.models.network import Network
from survey_adjustment.core.models.point import Point
from survey_adjustment.core.models.observation import DistanceObservation
from survey_adjustment.core.models.options import AdjustmentOptions
from survey_adjustment.core.solver.least_squares_2d import adjust_network_2d


@pytest.mark.parametrize(
    "p, expected",
    [
        (0.5, 0.0),
        (0.975, 1.959963984540054),
        (0.025, -1.959963984540054),
    ],
)
def test_normal_ppf_matches_stdlib(p, expected):
    assert normal_ppf(p) == pytest.approx(expected, rel=0, abs=1e-12)


@pytest.mark.parametrize(
    "p, df, expected",
    [
        (0.95, 1, 3.841458820694124),
        (0.95, 2, 5.991464547107979),
        (0.95, 10, 18.307038053275146),
        (0.99, 30, 50.89218131147588, ),  # table value; allow looser tolerance
    ],
)
def test_chi2_ppf_known_values(p, df, expected):
    tol = 1e-8 if df in (1, 2, 10) else 5e-3
    assert chi2_ppf(p, df) == pytest.approx(expected, abs=tol)


def test_chi2_cdf_roundtrip_for_known_value():
    x = 3.841458820694124
    assert chi2_cdf(x, 1) == pytest.approx(0.95, abs=5e-7)


def test_chi_square_global_test_returns_p_value_and_dof():
    res = chi_square_global_test(vTPv=10.0, dof=10, alpha=0.05, a_priori_variance=1.0)
    assert res.degrees_of_freedom == 10
    assert res.p_value is not None
    assert 0.0 <= res.p_value <= 1.0
    assert res.critical_lower < res.test_statistic < res.critical_upper
    assert res.passed is True


def test_standardized_residuals_formula():
    v = np.array([0.01, -0.02])
    qvv = np.array([0.0001, 0.0004])
    w = standardized_residuals(v, sigma0_hat=2.0, qvv_diag=qvv)
    # w_i = v_i / (2 * sqrt(qvv_i))
    assert w[0] == pytest.approx(0.01 / (2.0 * math.sqrt(0.0001)))
    assert w[1] == pytest.approx(-0.02 / (2.0 * math.sqrt(0.0004)))


def test_local_outlier_threshold_two_sided():
    k = local_outlier_threshold(0.01)
    # should be about 2.5758
    assert k == pytest.approx(2.5758293035489004, abs=1e-9)


def test_redundancy_numbers_in_unit_interval():
    qvv = np.array([0.5, 0.1, 2.0])
    p = np.array([1.0, 5.0, 0.1])
    r = redundancy_numbers(qvv, p)
    assert np.all(r >= 0.0)
    assert np.all(r <= 1.0)


def test_mdb_values_increases_when_redundancy_decreases():
    k_alpha, k_beta = 2.0, 1.0
    sigma0 = 1.0
    sigma = np.array([0.01, 0.01])
    r_hi = np.array([1.0, 0.5])
    mdb = mdb_values(k_alpha, k_beta, sigma0, sigma, r_hi)
    assert mdb[1] > mdb[0]


def test_external_reliability_simple_case():
    # Two parameters, one observation
    Qxx = np.eye(2)
    A = np.array([[1.0, 2.0]])
    p = np.array([4.0])
    mdb = np.array([0.5])
    er = external_reliability(Qxx, A, p, mdb, coordinate_param_indices=[0, 1])
    # dx = Qxx*(p*A_i)*mdb = [4,8]*0.5 = [2,4] => max abs is 4
    assert er[0] == pytest.approx(4.0)


def _simple_distance_network(with_outlier: bool = False) -> Network:
    net = Network(name="dist_net")
    net.add_point(Point("A", "A", 0.0, 0.0, fixed_easting=True, fixed_northing=True))
    net.add_point(Point("B", "B", 100.0, 0.0, fixed_easting=True, fixed_northing=True))
    net.add_point(Point("D", "D", 0.0, 100.0, fixed_easting=True, fixed_northing=True))
    net.add_point(Point("C", "C", 60.0, 80.0))

    # Three distances constrain C (redundant)
    true_ac = math.hypot(60.0, 80.0)
    true_bc = math.hypot(60.0 - 100.0, 80.0)
    true_dc = math.hypot(60.0 - 0.0, 80.0 - 100.0)

    val_ac = true_ac
    val_bc = true_bc
    val_dc = true_dc
    if with_outlier:
        val_bc += 0.5  # very large for 5mm sigma

    net.add_observation(DistanceObservation(id="d_AC", obs_type=None, value=val_ac, sigma=0.005, from_point_id="A", to_point_id="C"))
    net.add_observation(DistanceObservation(id="d_BC", obs_type=None, value=val_bc, sigma=0.005, from_point_id="B", to_point_id="C"))
    net.add_observation(DistanceObservation(id="d_DC", obs_type=None, value=val_dc, sigma=0.005, from_point_id="D", to_point_id="C"))
    return net


def test_adjustment_populates_reliability_fields_when_enabled():
    net = _simple_distance_network(with_outlier=False)
    opts = AdjustmentOptions(compute_reliability=True, compute_covariances=True, max_iterations=20)
    res = adjust_network_2d(net, opts)
    assert res.success and res.converged
    assert res.chi_square_test is not None
    assert res.chi_square_test.p_value is not None
    assert all((ri.redundancy_number is None) or (0.0 <= ri.redundancy_number <= 1.0) for ri in res.residual_details)
    assert all((ri.mdb is None) or math.isfinite(ri.mdb) for ri in res.residual_details)


def test_adjustment_flags_large_outlier_via_legacy_threshold():
    net = _simple_distance_network(with_outlier=True)
    # keep default outlier_threshold=3.0
    opts = AdjustmentOptions(compute_reliability=True, compute_covariances=True, max_iterations=20)
    res = adjust_network_2d(net, opts)
    assert res.success and res.converged
    assert "d_BC" in res.flagged_observations
