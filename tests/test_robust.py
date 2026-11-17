"""
Tests for robust estimation (IRLS with Huber, Danish, and IGG-III weights).

These tests verify:
- Each weight function against its defining formula
- Continuity and monotonicity of the weight functions
- The generic IRLS loop
- That robust adjustment isolates a gross error that ordinary least squares smears
"""

import math

import numpy as np
import pytest

from survey_adjustment.core.models.point import Point
from survey_adjustment.core.models.observation import (
    ObservationType,
    HeightDifferenceObservation,
)
from survey_adjustment.core.models.network import Network
from survey_adjustment.core.models.options import AdjustmentOptions, RobustEstimator
from survey_adjustment.core.solver.least_squares_1d import adjust_leveling_1d
from survey_adjustment.core.solver.robust import (
    RobustMethod,
    huber_weight,
    danish_weight,
    igg3_weight,
    get_weight_function,
    compute_robust_weights,
    irls_solve,
)


class TestWeightFunctions:

    @pytest.mark.parametrize("w", [0.0, 0.5, 1.5, -1.5])
    def test_huber_unit_weight_inside(self, w):
        assert huber_weight(w, c=1.5) == 1.0

    @pytest.mark.parametrize("w", [2.0, 3.0, -6.0])
    def test_huber_outside(self, w):
        assert huber_weight(w, c=1.5) == pytest.approx(1.5 / abs(w))

    @pytest.mark.parametrize("w", [0.0, 2.0, -2.0])
    def test_danish_unit_weight_inside(self, w):
        assert danish_weight(w, c=2.0) == 1.0

    @pytest.mark.parametrize("w", [2.5, 4.0, -6.0])
    def test_danish_outside(self, w):
        assert danish_weight(w, c=2.0) == pytest.approx(math.exp(-((abs(w) - 2.0) / 2.0) ** 2))

    def test_igg3_regions(self):
        assert igg3_weight(1.0, k0=1.5, k1=3.0) == 1.0
        assert igg3_weight(2.0, k0=1.5, k1=3.0) == pytest.approx((1.5 / 2.0) * ((3.0 - 2.0) / 1.5) ** 2)
        assert igg3_weight(3.0, k0=1.5, k1=3.0) == pytest.approx(0.0, abs=1e-9)
        assert igg3_weight(10.0, k0=1.5, k1=3.0) == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("func", [
        lambda w: huber_weight(w, 1.5),
        lambda w: danish_weight(w, 2.0),
        lambda w: igg3_weight(w, 1.5, 3.0),
    ], ids=["huber", "danish", "igg3"])
    def test_symmetric_bounded_non_increasing(self, func):
        ws = np.linspace(0.0, 8.0, 801)
        u = np.array([func(w) for w in ws])
        assert np.all((u >= 0.0) & (u <= 1.0))
        assert np.all(np.diff(u) <= 1e-12)
        assert all(func(w) == func(-w) for w in ws)

    @pytest.mark.parametrize("func, knot", [
        (lambda w: huber_weight(w, 1.5), 1.5),
        (lambda w: danish_weight(w, 2.0), 2.0),
        (lambda w: igg3_weight(w, 1.5, 3.0), 1.5),
    ], ids=["huber", "danish", "igg3"])
    def test_continuous_at_threshold(self, func, knot):
        assert func(knot + 1e-9) == pytest.approx(func(knot), abs=1e-6)

    def test_get_weight_function(self):
        assert get_weight_function(RobustMethod.NONE) is None
        assert get_weight_function(RobustMethod.HUBER, huber_c=2.0)(4.0) == pytest.approx(0.5)

    def test_compute_robust_weights_ignores_non_finite(self):
        weights = compute_robust_weights(np.array([0.5, np.nan, 3.0]), lambda w: huber_weight(w, 1.5))
        np.testing.assert_allclose(weights, [1.0, 1.0, 0.5])


class TestIRLSLoop:

    def test_none_returns_unit_weights(self):
        result = irls_solve(lambda w: (np.zeros(4), np.zeros(4), 1.0), 4, RobustMethod.NONE)
        assert result.converged
        assert result.iterations == 0
        np.testing.assert_array_equal(result.final_weights, np.ones(4))

    def test_weighted_mean_downweights_outlier(self):
        """Huber location estimate of five values, one of them an outlier."""
        values = np.array([10.0, 10.1, 9.9, 10.05, 14.0])
        sigma = 0.1

        def solve(weights):
            mean = np.sum(weights * values) / np.sum(weights)
            residuals = values - mean
            return residuals / sigma, residuals, 1.0

        result = irls_solve(solve, len(values), RobustMethod.HUBER)
        assert result.converged
        assert len(result.iteration_log) == result.iterations
        assert np.argmin(result.final_weights) == len(values) - 1

        robust_mean = np.sum(result.final_weights * values) / np.sum(result.final_weights)
        assert abs(robust_mean - 10.0) < abs(values.mean() - 10.0)


class TestRobustLeveling:
    """A 5 cm blunder in a redundant leveling network."""

    TRUTH = {"A": 100.0, "B": 101.2, "C": 102.5, "D": 101.8}
    LINES = {"L1": ("A", "B"), "L2": ("B", "C"), "L3": ("C", "D"), "L4": ("D", "A"),
             "L5": ("A", "C"), "L6": ("B", "D"), "L7": ("A", "D")}
    NOISE = {"L1": 0.0003, "L2": -0.0002, "L3": 0.0004, "L4": -0.0003,
             "L5": 0.0001, "L6": -0.0004, "L7": 0.0002}
    BLUNDER = ("L3", 0.050)

    def _network(self):
        network = Network(name="blunder")
        network.add_point(Point(id="A", name="A", easting=0.0, northing=0.0,
                                height=self.TRUTH["A"], fixed_height=True))
        for pid in "BCD":
            network.add_point(Point(id=pid, name=pid, easting=0.0, northing=0.0,
                                    height=self.TRUTH[pid]))
        for obs_id, (f, t) in self.LINES.items():
            value = self.TRUTH[t] - self.TRUTH[f] + self.NOISE[obs_id]
            if obs_id == self.BLUNDER[0]:
                value += self.BLUNDER[1]
            network.add_observation(HeightDifferenceObservation(
                id=obs_id, obs_type=ObservationType.HEIGHT_DIFF, value=value, sigma=0.001,
                from_point_id=f, to_point_id=t))
        return network

    def _max_height_error(self, result):
        return max(abs(result.adjusted_points[p].height - h) for p, h in self.TRUTH.items())

    def test_least_squares_is_corrupted_by_blunder(self):
        result = adjust_leveling_1d(self._network())
        assert self._max_height_error(result) > 0.005

    @pytest.mark.parametrize("estimator", [RobustEstimator.HUBER, RobustEstimator.DANISH,
                                           RobustEstimator.IGG3])
    def test_robust_estimation_isolates_blunder(self, estimator):
        ls = adjust_leveling_1d(self._network())
        robust = adjust_leveling_1d(self._network(), AdjustmentOptions(robust_estimator=estimator))
        assert robust.success

        # Closer to the truth than ordinary least squares ...
        assert self._max_height_error(robust) < self._max_height_error(ls)

        # ... because the blundered observation got the smallest weight.
        weights = {r.obs_id: r.weight_factor for r in robust.residual_details}
        blunder_id = self.BLUNDER[0]
        assert weights[blunder_id] == min(weights.values())
        assert weights[blunder_id] < 0.5

    # Known defect: redescending estimators start from the smeared LS solution and
    # reject good observations. See issue #16; remove the markers once it is fixed.

    @pytest.mark.xfail(strict=True, reason="issue #16: robust estimation does not isolate the blunder")
    @pytest.mark.parametrize("estimator", [RobustEstimator.HUBER, RobustEstimator.DANISH,
                                           RobustEstimator.IGG3])
    def test_robust_recovers_heights_to_millimetre(self, estimator):
        robust = adjust_leveling_1d(self._network(), AdjustmentOptions(robust_estimator=estimator))
        weights = [r.weight_factor for r in robust.residual_details]
        assert robust.robust_converged
        assert sum(u < 0.01 for u in weights) == 1  # only the blunder is rejected
        assert self._max_height_error(robust) < 0.001

    @pytest.mark.xfail(strict=True, reason="issue #16: IRLS does not converge for redescending methods")
    @pytest.mark.parametrize("estimator", [RobustEstimator.DANISH, RobustEstimator.IGG3])
    def test_redescending_irls_converges_for_small_blunder(self, estimator):
        network = self._network()
        network.get_observation("L3").value -= self.BLUNDER[1] - 0.005  # 5 mm blunder
        robust = adjust_leveling_1d(network, AdjustmentOptions(robust_estimator=estimator))
        assert robust.robust_converged
