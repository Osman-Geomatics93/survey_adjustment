"""Tests for Phase 7A: Robust Estimation (IRLS)."""

import math
import pytest

from survey_adjustment.core.models.network import Network
from survey_adjustment.core.models.point import Point
from survey_adjustment.core.models.observation import (
    DistanceObservation,
    DirectionObservation,
    HeightDifferenceObservation,
    GnssBaselineObservation,
)
from survey_adjustment.core.models.options import AdjustmentOptions, RobustEstimator
from survey_adjustment.core.solver.robust import (
    RobustMethod,
    huber_weight,
    danish_weight,
    igg3_weight,
    compute_robust_weights,
    get_weight_function,
)
from survey_adjustment.core.solver import adjust_network_2d
from survey_adjustment.core.solver.least_squares_1d import adjust_leveling_1d
from survey_adjustment.core.solver.least_squares_3d import adjust_gnss_3d
from survey_adjustment.core.solver.least_squares_mixed import adjust_network_mixed


# -----------------------------------------------------------------------------
# Weight function tests
# -----------------------------------------------------------------------------

class TestWeightFunctions:
    """Test robust weight functions."""

    def test_huber_weight_within_threshold(self):
        """Huber returns 1.0 for |w| <= c."""
        c = 1.5
        for w in [0.0, 0.5, 1.0, 1.5, -1.0, -1.5]:
            assert huber_weight(w, c) == 1.0

    def test_huber_weight_above_threshold(self):
        """Huber returns c/|w| for |w| > c."""
        c = 1.5
        assert huber_weight(2.0, c) == pytest.approx(1.5 / 2.0)
        assert huber_weight(3.0, c) == pytest.approx(1.5 / 3.0)
        assert huber_weight(-2.5, c) == pytest.approx(1.5 / 2.5)

    def test_danish_weight_within_threshold(self):
        """Danish returns 1.0 for |w| <= c."""
        c = 2.0
        for w in [0.0, 1.0, 2.0, -1.5, -2.0]:
            assert danish_weight(w, c) == 1.0

    def test_danish_weight_above_threshold(self):
        """Danish applies exponential decay for |w| > c."""
        c = 2.0
        # For w=3, weight = exp(-((3-2)/2)^2) = exp(-0.25)
        expected = math.exp(-((3.0 - 2.0) / 2.0) ** 2)
        assert danish_weight(3.0, c) == pytest.approx(expected)

        # For w=4, weight = exp(-((4-2)/2)^2) = exp(-1)
        expected = math.exp(-1.0)
        assert danish_weight(4.0, c) == pytest.approx(expected)

    def test_igg3_weight_region1(self):
        """IGG-III returns 1.0 for |w| <= k0."""
        k0, k1 = 1.5, 3.0
        for w in [0.0, 0.5, 1.0, 1.5, -1.0, -1.5]:
            assert igg3_weight(w, k0, k1) == 1.0

    def test_igg3_weight_region2(self):
        """IGG-III applies downweighting for k0 < |w| < k1."""
        k0, k1 = 1.5, 3.0
        w = 2.0
        # weight = (k0/|w|) * ((k1 - |w|) / (k1 - k0))^2
        expected = (1.5 / 2.0) * ((3.0 - 2.0) / (3.0 - 1.5)) ** 2
        assert igg3_weight(w, k0, k1) == pytest.approx(expected)

    def test_igg3_weight_region3(self):
        """IGG-III returns near-zero for |w| >= k1."""
        k0, k1 = 1.5, 3.0
        assert igg3_weight(3.0, k0, k1) < 1e-8
        assert igg3_weight(5.0, k0, k1) < 1e-8
        assert igg3_weight(-4.0, k0, k1) < 1e-8

    def test_compute_robust_weights_array(self):
        """compute_robust_weights returns array of correct weights."""
        import numpy as np
        std_res = np.array([0.5, 1.0, 2.0, 3.5])
        weight_func = get_weight_function(RobustMethod.HUBER, huber_c=1.5)

        weights = compute_robust_weights(std_res, weight_func)

        assert len(weights) == 4
        assert weights[0] == 1.0  # 0.5 <= 1.5
        assert weights[1] == 1.0  # 1.0 <= 1.5
        assert weights[2] == pytest.approx(1.5 / 2.0)  # 2.0 > 1.5
        assert weights[3] == pytest.approx(1.5 / 3.5)  # 3.5 > 1.5


# -----------------------------------------------------------------------------
# 2D solver robust estimation tests
# -----------------------------------------------------------------------------

class TestRobust2D:
    """Test robust estimation in 2D solver."""

    def _build_clean_network(self) -> Network:
        """Build a small 2D network with no outliers."""
        net = Network(name="test_robust_2d")

        # Fixed control points
        net.add_point(Point("A", "A", 0.0, 0.0, fixed_easting=True, fixed_northing=True))
        net.add_point(Point("B", "B", 100.0, 0.0, fixed_easting=True, fixed_northing=True))

        # Unknown point with approximate coordinates
        net.add_point(Point("C", "C", 50.0, 50.0))

        # True position of C
        C_true = (50.0, 50.0)

        # Distance observations (consistent with true position)
        d_AC = math.hypot(C_true[0], C_true[1])
        d_BC = math.hypot(C_true[0] - 100.0, C_true[1])

        net.add_observation(DistanceObservation(
            id="d_AC", obs_type=None, value=d_AC, sigma=0.005,
            from_point_id="A", to_point_id="C"
        ))
        net.add_observation(DistanceObservation(
            id="d_BC", obs_type=None, value=d_BC, sigma=0.005,
            from_point_id="B", to_point_id="C"
        ))

        return net

    def _build_network_with_outlier(self) -> Network:
        """Build network with one gross outlier observation."""
        net = Network(name="test_robust_outlier")

        # Fixed control points
        net.add_point(Point("A", "A", 0.0, 0.0, fixed_easting=True, fixed_northing=True))
        net.add_point(Point("B", "B", 100.0, 0.0, fixed_easting=True, fixed_northing=True))

        # Unknown point
        net.add_point(Point("C", "C", 50.0, 50.0))

        C_true = (50.0, 50.0)
        d_AC = math.hypot(C_true[0], C_true[1])
        d_BC = math.hypot(C_true[0] - 100.0, C_true[1])

        # Good observations
        net.add_observation(DistanceObservation(
            id="d_AC", obs_type=None, value=d_AC, sigma=0.005,
            from_point_id="A", to_point_id="C"
        ))
        net.add_observation(DistanceObservation(
            id="d_BC", obs_type=None, value=d_BC, sigma=0.005,
            from_point_id="B", to_point_id="C"
        ))

        # Add one gross outlier (distance off by 1 meter - huge for sigma=0.005)
        net.add_observation(DistanceObservation(
            id="d_BC_outlier", obs_type=None, value=d_BC + 1.0, sigma=0.005,
            from_point_id="B", to_point_id="C"
        ))

        return net

    def test_standard_ls_equals_robust_none(self):
        """Standard LS and robust_estimator=None should give identical results."""
        net = self._build_clean_network()

        # Standard LS
        opts_std = AdjustmentOptions(
            max_iterations=20,
            convergence_threshold=1e-10,
            robust_estimator=None,
        )
        result_std = adjust_network_2d(net, opts_std)

        # "Robust" with no method (should be identical)
        opts_robust = AdjustmentOptions(
            max_iterations=20,
            convergence_threshold=1e-10,
            robust_estimator=None,
        )
        result_robust = adjust_network_2d(net, opts_robust)

        assert result_std.success
        assert result_robust.success

        # Adjusted coordinates should be identical
        assert result_std.adjusted_points["C"].easting == pytest.approx(
            result_robust.adjusted_points["C"].easting, abs=1e-10
        )
        assert result_std.adjusted_points["C"].northing == pytest.approx(
            result_robust.adjusted_points["C"].northing, abs=1e-10
        )

        # Variance factor should be identical
        assert result_std.variance_factor == pytest.approx(
            result_robust.variance_factor, rel=1e-10
        )

    def test_huber_robust_converges(self):
        """Huber robust estimation converges on clean network."""
        net = self._build_clean_network()

        opts = AdjustmentOptions(
            max_iterations=20,
            convergence_threshold=1e-10,
            robust_estimator=RobustEstimator.HUBER,
        )
        result = adjust_network_2d(net, opts)

        assert result.success
        assert result.converged
        assert result.robust_method == "huber"
        assert result.robust_converged

    def test_danish_robust_converges(self):
        """Danish robust estimation converges on clean network."""
        net = self._build_clean_network()

        opts = AdjustmentOptions(
            max_iterations=20,
            convergence_threshold=1e-10,
            robust_estimator=RobustEstimator.DANISH,
        )
        result = adjust_network_2d(net, opts)

        assert result.success
        assert result.converged
        assert result.robust_method == "danish"

    def test_igg3_robust_converges(self):
        """IGG-III robust estimation converges on clean network."""
        net = self._build_clean_network()

        opts = AdjustmentOptions(
            max_iterations=20,
            convergence_threshold=1e-10,
            robust_estimator=RobustEstimator.IGG3,
        )
        result = adjust_network_2d(net, opts)

        assert result.success
        assert result.converged
        assert result.robust_method == "igg3"

    def test_robust_downweights_outlier(self):
        """Robust estimation downweights gross outlier observation."""
        net = self._build_network_with_outlier()

        opts = AdjustmentOptions(
            max_iterations=50,
            convergence_threshold=1e-10,
            robust_estimator=RobustEstimator.HUBER,
            robust_max_iterations=30,
        )
        result = adjust_network_2d(net, opts)

        assert result.success

        # Find weight factor for outlier observation
        outlier_detail = None
        for detail in result.residual_details:
            if detail.obs_id == "d_BC_outlier":
                outlier_detail = detail
                break

        assert outlier_detail is not None
        assert outlier_detail.weight_factor is not None
        # Outlier should be downweighted (weight < 1)
        assert outlier_detail.weight_factor < 0.5, f"Expected outlier to be downweighted, got {outlier_detail.weight_factor}"

    def test_weight_factors_in_result(self):
        """Robust estimation includes weight_factor in residual details."""
        net = self._build_clean_network()

        opts = AdjustmentOptions(
            max_iterations=20,
            convergence_threshold=1e-10,
            robust_estimator=RobustEstimator.HUBER,
        )
        result = adjust_network_2d(net, opts)

        assert result.success

        # All residual details should have weight_factor
        for detail in result.residual_details:
            assert detail.weight_factor is not None
            # For clean network, all weights should be close to 1.0
            assert 0.0 < detail.weight_factor <= 1.0


# -----------------------------------------------------------------------------
# 1D solver robust estimation tests
# -----------------------------------------------------------------------------

class TestRobust1D:
    """Test robust estimation in 1D leveling solver."""

    def _build_leveling_network(self, include_outlier: bool = False) -> Network:
        """Build a simple leveling network."""
        net = Network(name="test_leveling_robust")

        # Fixed benchmark
        net.add_point(Point("BM1", "BM1", 0.0, 0.0, height=100.0, fixed_height=True))

        # Unknown points
        net.add_point(Point("P1", "P1", 100.0, 0.0, height=101.0))
        net.add_point(Point("P2", "P2", 200.0, 0.0, height=102.0))

        # Height differences (consistent values)
        net.add_observation(HeightDifferenceObservation(
            id="dh_BM1_P1", obs_type=None, value=1.002, sigma=0.001,
            from_point_id="BM1", to_point_id="P1"
        ))
        net.add_observation(HeightDifferenceObservation(
            id="dh_P1_P2", obs_type=None, value=0.998, sigma=0.001,
            from_point_id="P1", to_point_id="P2"
        ))
        net.add_observation(HeightDifferenceObservation(
            id="dh_BM1_P2", obs_type=None, value=2.001, sigma=0.002,
            from_point_id="BM1", to_point_id="P2"
        ))

        if include_outlier:
            # Add gross outlier (50mm error for sigma=1mm)
            net.add_observation(HeightDifferenceObservation(
                id="dh_outlier", obs_type=None, value=1.052, sigma=0.001,
                from_point_id="BM1", to_point_id="P1"
            ))

        return net

    def test_1d_huber_converges(self):
        """Huber robust estimation converges for leveling."""
        net = self._build_leveling_network()

        opts = AdjustmentOptions(
            robust_estimator=RobustEstimator.HUBER,
            compute_reliability=True,
        )
        result = adjust_leveling_1d(net, opts)

        assert result.success
        assert result.robust_method == "huber"
        assert result.robust_converged

    def test_1d_robust_downweights_outlier(self):
        """1D robust estimation downweights outlier."""
        net = self._build_leveling_network(include_outlier=True)

        opts = AdjustmentOptions(
            robust_estimator=RobustEstimator.DANISH,
            robust_max_iterations=30,
        )
        result = adjust_leveling_1d(net, opts)

        assert result.success

        # Find outlier weight
        outlier_detail = None
        for detail in result.residual_details:
            if detail.obs_id == "dh_outlier":
                outlier_detail = detail
                break

        assert outlier_detail is not None
        assert outlier_detail.weight_factor is not None
        # Outlier should be heavily downweighted
        assert outlier_detail.weight_factor < 0.3


# -----------------------------------------------------------------------------
# 3D GNSS solver robust estimation tests
# -----------------------------------------------------------------------------

class TestRobust3DGNSS:
    """Test robust estimation in 3D GNSS solver."""

    def _build_gnss_network(self) -> Network:
        """Build a simple GNSS baseline network."""
        net = Network(name="test_gnss_robust")

        # Fixed reference
        net.add_point(Point(
            "REF", "REF", 0.0, 0.0, height=0.0,
            fixed_easting=True, fixed_northing=True, fixed_height=True
        ))

        # Unknown point
        net.add_point(Point("P1", "P1", 100.0, 100.0, height=50.0))

        # GNSS baselines (dE, dN, dH)
        # Note: obs_type, value, sigma are placeholders overwritten by __post_init__
        from survey_adjustment.core.models.observation import ObservationType
        net.add_observation(GnssBaselineObservation(
            id="bl_REF_P1",
            obs_type=ObservationType.GNSS_BASELINE,
            value=0.0, sigma=1.0,
            from_point_id="REF", to_point_id="P1",
            dE=100.001, dN=99.998, dH=50.002,
            cov_EE=0.0001, cov_NN=0.0001, cov_HH=0.0004,
            cov_EN=0.0, cov_EH=0.0, cov_NH=0.0
        ))

        return net

    def test_3d_huber_converges(self):
        """Huber robust estimation converges for GNSS."""
        net = self._build_gnss_network()

        opts = AdjustmentOptions(
            robust_estimator=RobustEstimator.HUBER,
        )
        result = adjust_gnss_3d(net, opts)

        assert result.success
        assert result.robust_method == "huber"


# -----------------------------------------------------------------------------
# Mixed solver robust estimation tests
# -----------------------------------------------------------------------------

class TestRobustMixed:
    """Test robust estimation in mixed solver."""

    def _build_mixed_network(self) -> Network:
        """Build a mixed network with classical + leveling (overdetermined)."""
        net = Network(name="test_mixed_robust")

        # Fixed control
        net.add_point(Point(
            "A", "A", 0.0, 0.0, height=100.0,
            fixed_easting=True, fixed_northing=True, fixed_height=True
        ))
        net.add_point(Point(
            "B", "B", 100.0, 0.0, height=100.5,
            fixed_easting=True, fixed_northing=True, fixed_height=True
        ))

        # Unknown point
        net.add_point(Point("C", "C", 50.0, 50.0, height=101.0))

        # Classical distances (2 observations for 2 horizontal unknowns + 1 redundancy)
        d_AC = math.hypot(50.0, 50.0)
        d_BC = math.hypot(50.0, 50.0)  # Distance from B(100,0) to C(50,50)
        net.add_observation(DistanceObservation(
            id="d_AC", obs_type=None, value=d_AC + 0.001, sigma=0.005,
            from_point_id="A", to_point_id="C"
        ))
        net.add_observation(DistanceObservation(
            id="d_BC", obs_type=None, value=d_BC, sigma=0.005,
            from_point_id="B", to_point_id="C"
        ))

        # Leveling (2 observations for 1 height unknown + 1 redundancy)
        net.add_observation(HeightDifferenceObservation(
            id="dh_A_C", obs_type=None, value=1.001, sigma=0.002,
            from_point_id="A", to_point_id="C"
        ))
        net.add_observation(HeightDifferenceObservation(
            id="dh_B_C", obs_type=None, value=0.501, sigma=0.002,
            from_point_id="B", to_point_id="C"
        ))

        return net

    def test_mixed_huber_converges(self):
        """Huber robust estimation converges for mixed network."""
        net = self._build_mixed_network()

        opts = AdjustmentOptions(
            max_iterations=30,
            convergence_threshold=1e-10,
            robust_estimator=RobustEstimator.HUBER,
        )
        result = adjust_network_mixed(net, opts)

        assert result.success
        assert result.robust_method == "huber"


# -----------------------------------------------------------------------------
# IRLS convergence tests
# -----------------------------------------------------------------------------

class TestIRLSConvergence:
    """Test IRLS convergence behavior."""

    def test_irls_iteration_count(self):
        """IRLS performs multiple iterations when needed."""
        net = Network(name="test_irls")

        net.add_point(Point("A", "A", 0.0, 0.0, fixed_easting=True, fixed_northing=True))
        net.add_point(Point("B", "B", 100.0, 0.0, fixed_easting=True, fixed_northing=True))
        net.add_point(Point("C", "C", 50.0, 50.0))

        # Add observations with one moderate outlier
        d_AC = math.hypot(50.0, 50.0)
        d_BC = math.hypot(50.0, 50.0)

        net.add_observation(DistanceObservation(
            id="d_AC", obs_type=None, value=d_AC, sigma=0.005,
            from_point_id="A", to_point_id="C"
        ))
        net.add_observation(DistanceObservation(
            id="d_BC", obs_type=None, value=d_BC, sigma=0.005,
            from_point_id="B", to_point_id="C"
        ))
        net.add_observation(DistanceObservation(
            id="d_AC_2", obs_type=None, value=d_AC + 0.05, sigma=0.005,  # Moderate outlier
            from_point_id="A", to_point_id="C"
        ))

        opts = AdjustmentOptions(
            max_iterations=30,
            convergence_threshold=1e-10,
            robust_estimator=RobustEstimator.HUBER,
            robust_max_iterations=50,
        )
        result = adjust_network_2d(net, opts)

        assert result.success
        # IRLS should have done some iterations
        assert result.robust_iterations >= 1
