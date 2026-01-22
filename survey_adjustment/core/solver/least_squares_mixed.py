"""survey_adjustment.core.solver.least_squares_mixed

Unified least-squares adjustment combining:
- Classical observations: distances, directions, angles (2D: E, N)
- GNSS baselines: (3D: E, N, H with full covariance)
- Leveling observations: height differences (1D: H only)

This solver handles mixed observation types in a single adjustment:
- Unknowns: E, N, H of free points + orientation ω for direction stations
- Classical observations affect only E, N (height not involved)
- GNSS baselines affect E, N, H with 3x3 covariance blocks
- Leveling observations affect only H
- Proper weighting: scalar for classical/leveling, block-inverse for GNSS

The adjustment is iterative (Gauss-Newton) due to non-linear direction/angle
observations. GNSS baselines and leveling are linear but included in the same solve.

Supports robust estimation via IRLS (Iteratively Reweighted Least Squares)
with Huber, Danish, and IGG-III weight functions.

This module intentionally contains **no QGIS imports** so it can be unit-
tested and re-used in non-QGIS contexts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore

from ..models.network import Network
from ..models.options import AdjustmentOptions, RobustEstimator
from ..models.point import Point
from .robust import (
    RobustMethod,
    get_weight_function,
    compute_robust_weights,
)
from ..models.observation import (
    Observation,
    DistanceObservation,
    DirectionObservation,
    AngleObservation,
    HeightDifferenceObservation,
    GnssBaselineObservation,
)
from ..results.adjustment_result import (
    AdjustmentResult,
    ResidualInfo,
    ErrorEllipse,
)
from ..statistics import (
    chi_square_global_test,
    standardized_residuals,
    local_outlier_threshold,
    normal_ppf,
    chi2_ppf,
)
from ..statistics.reliability import (
    redundancy_numbers,
    mdb_values,
    external_reliability,
)
from .geometry import (
    wrap_pi,
    wrap_2pi,
    azimuth,
    distance_2d,
    distance_partials,
    azimuth_partials,
    angle_at_point,
    angle_partials,
)


@dataclass(frozen=True)
class MixedParameterIndex:
    """Mapping of unknown parameters to indices in the solver vector.

    For mixed adjustment, coordinates include E, N, H (3D) when GNSS is present.
    """
    coord_index: Dict[Tuple[str, str], int]    # (point_id, 'E'|'N'|'H') -> idx
    orientation_index: Dict[str, int]          # direction set_id -> idx
    num_params: int
    coord_order: List[Tuple[str, str]]
    orientation_order: List[str]


@dataclass
class _State:
    """Mutable state during iterative adjustment."""
    points: Dict[str, Tuple[float, float, float]]  # id -> (E, N, H)
    orientations: Dict[str, float]                  # set_id -> omega (rad)


def _build_mixed_parameter_index(
    network: Network,
    gnss_obs: List[GnssBaselineObservation],
    leveling_obs: List[HeightDifferenceObservation],
    has_classical: bool,
) -> MixedParameterIndex:
    """Build parameter index for mixed adjustment.

    For classical-only: E, N for non-fixed horizontal components
    For GNSS or mixed: E, N, H for non-fixed components
    For leveling: H for points in leveling observations
    Plus orientation unknowns for direction sets.

    IMPORTANT: Height (H) unknowns are only added for points that actually
    appear in GNSS baseline or leveling observations, since classical 2D
    observations don't constrain height.
    """
    coord_index: Dict[Tuple[str, str], int] = {}
    orientation_index: Dict[str, int] = {}
    coord_order: List[Tuple[str, str]] = []
    orientation_order: List[str] = []

    # Collect point IDs that appear in GNSS baselines
    gnss_point_ids: Set[str] = set()
    for obs in gnss_obs:
        gnss_point_ids.add(obs.from_point_id)
        gnss_point_ids.add(obs.to_point_id)

    # Collect point IDs that appear in leveling observations
    leveling_point_ids: Set[str] = set()
    for obs in leveling_obs:
        leveling_point_ids.add(obs.from_point_id)
        leveling_point_ids.add(obs.to_point_id)

    # Points that need height unknowns
    height_point_ids = gnss_point_ids | leveling_point_ids

    has_gnss = len(gnss_obs) > 0
    has_leveling = len(leveling_obs) > 0
    has_height_obs = has_gnss or has_leveling

    idx = 0
    for pid in sorted(network.points.keys()):
        p = network.points[pid]

        # E and N for all observations
        if not p.fixed_easting:
            coord_index[(pid, 'E')] = idx
            coord_order.append((pid, 'E'))
            idx += 1
        if not p.fixed_northing:
            coord_index[(pid, 'N')] = idx
            coord_order.append((pid, 'N'))
            idx += 1

        # H if this point is in GNSS or leveling observations
        if has_height_obs and pid in height_point_ids and not p.fixed_height:
            coord_index[(pid, 'H')] = idx
            coord_order.append((pid, 'H'))
            idx += 1

    # Direction set orientations
    if has_classical:
        set_ids: Set[str] = set()
        for obs in network.get_enabled_observations():
            if isinstance(obs, DirectionObservation):
                set_ids.add(obs.set_id)
        for set_id in sorted(set_ids):
            orientation_index[set_id] = idx
            orientation_order.append(set_id)
            idx += 1

    return MixedParameterIndex(
        coord_index=coord_index,
        orientation_index=orientation_index,
        num_params=idx,
        coord_order=coord_order,
        orientation_order=orientation_order,
    )


def _network_span(points: Dict[str, Point]) -> float:
    """Approximate network span for orientation convergence scaling."""
    if len(points) < 2:
        return 1.0
    coords = [(p.easting, p.northing) for p in points.values()]
    max_d = 0.0
    for i in range(len(coords)):
        e1, n1 = coords[i]
        for j in range(i + 1, len(coords)):
            e2, n2 = coords[j]
            max_d = max(max_d, distance_2d(e1, n1, e2, n2))
    return max(max_d, 1.0)


def _invert_3x3(matrix: np.ndarray) -> np.ndarray:
    """Invert a 3x3 matrix."""
    return np.linalg.inv(matrix)


def adjust_network_mixed(
    network: Network,
    options: AdjustmentOptions | None = None,
) -> AdjustmentResult:
    """Run a mixed least-squares adjustment combining classical, GNSS, and leveling observations.

    This solver handles:
    - Distance observations (2D)
    - Direction observations with orientation unknowns (2D)
    - Angle observations (2D)
    - GNSS baseline observations (3D with full covariance)
    - Height difference observations (1D leveling)

    Args:
        network: Input network with points and mixed observations
        options: Adjustment options (defaults if None)

    Returns:
        AdjustmentResult with adjusted coordinates, residuals, covariance, etc.
    """
    if np is None:
        return AdjustmentResult.failure("NumPy is required for the least-squares solver")

    options = options or AdjustmentOptions.default()

    # Validate network for mixed adjustment
    errors = network.validate_mixed()
    if errors:
        return AdjustmentResult.failure("; ".join(errors))

    # Categorize observations
    enabled_obs: List[Observation] = list(network.get_enabled_observations())

    classical_obs: List[Observation] = []
    gnss_obs: List[GnssBaselineObservation] = []
    leveling_obs: List[HeightDifferenceObservation] = []

    for obs in enabled_obs:
        if isinstance(obs, GnssBaselineObservation):
            gnss_obs.append(obs)
        elif isinstance(obs, HeightDifferenceObservation):
            leveling_obs.append(obs)
        elif isinstance(obs, (DistanceObservation, DirectionObservation, AngleObservation)):
            classical_obs.append(obs)
        else:
            return AdjustmentResult.failure(f"Unsupported observation type: {type(obs)}")

    has_classical = len(classical_obs) > 0
    has_gnss = len(gnss_obs) > 0
    has_leveling = len(leveling_obs) > 0

    if not has_classical and not has_gnss and not has_leveling:
        return AdjustmentResult.failure("No observations to adjust")

    # Count total observation equations
    # Classical: 1 equation each
    # GNSS: 3 equations each (dE, dN, dH)
    # Leveling: 1 equation each
    m_classical = len(classical_obs)
    m_gnss = 3 * len(gnss_obs)
    m_leveling = len(leveling_obs)
    m = m_classical + m_gnss + m_leveling

    # Build parameter index
    index = _build_mixed_parameter_index(network, gnss_obs, leveling_obs, has_classical)
    n = index.num_params

    if n == 0:
        return AdjustmentResult.failure("All coordinates are fixed - nothing to adjust")

    # Check for height values if GNSS or leveling present
    if has_gnss or has_leveling:
        # Collect point IDs that need heights
        height_point_ids: Set[str] = set()
        for obs in gnss_obs:
            height_point_ids.add(obs.from_point_id)
            height_point_ids.add(obs.to_point_id)
        for obs in leveling_obs:
            height_point_ids.add(obs.from_point_id)
            height_point_ids.add(obs.to_point_id)

        for pid in height_point_ids:
            p = network.points[pid]
            if p.height is None:
                return AdjustmentResult.failure(f"Point '{pid}' has no height value (required for GNSS/leveling)")

    # Robust estimation setup
    robust_method = RobustMethod.NONE
    if options.robust_estimator is not None:
        if options.robust_estimator == RobustEstimator.HUBER:
            robust_method = RobustMethod.HUBER
        elif options.robust_estimator == RobustEstimator.DANISH:
            robust_method = RobustMethod.DANISH
        elif options.robust_estimator == RobustEstimator.IGG3:
            robust_method = RobustMethod.IGG3

    weight_func = get_weight_function(
        robust_method,
        huber_c=options.huber_c,
        danish_c=options.danish_c,
        igg3_k0=options.igg3_k0,
        igg3_k1=options.igg3_k1,
    )

    # Initialize robust weights for each observation type
    robust_weights_classical = np.ones(m_classical) if m_classical > 0 else np.array([])
    robust_weights_gnss = np.ones(len(gnss_obs)) if len(gnss_obs) > 0 else np.array([])  # One per baseline
    robust_weights_leveling = np.ones(m_leveling) if m_leveling > 0 else np.array([])
    robust_converged = True
    robust_iterations = 0
    robust_message: Optional[str] = None

    # Build initial state
    state = _State(
        points={
            pid: (p.easting, p.northing, p.height if p.height is not None else 0.0)
            for pid, p in network.points.items()
        },
        orientations={set_id: 0.0 for set_id in index.orientation_order},
    )

    span = _network_span(network.points)
    orient_tol = max(1e-14, options.convergence_threshold / span)

    converged = False

    # Pre-compute GNSS weight blocks (constant across iterations)
    gnss_weight_blocks: List[np.ndarray] = []
    gnss_cov_blocks: List[np.ndarray] = []

    for obs in gnss_obs:
        C = np.array(obs.covariance_matrix, dtype=float)
        gnss_cov_blocks.append(C)
        try:
            P = _invert_3x3(C)
        except np.linalg.LinAlgError:
            return AdjustmentResult.failure(
                f"Baseline {obs.id}: covariance matrix is singular"
            )
        gnss_weight_blocks.append(P)

    # Outer IRLS loop (only if robust method is enabled)
    max_irls = options.robust_max_iterations if robust_method != RobustMethod.NONE else 1

    for irls_iter in range(1, max_irls + 1):
        # Reset state for each IRLS iteration
        if irls_iter > 1:
            state = _State(
                points={
                    pid: (p.easting, p.northing, p.height if p.height is not None else 0.0)
                    for pid, p in network.points.items()
                },
                orientations={set_id: 0.0 for set_id in index.orientation_order},
            )

        converged = False

        # Iterative adjustment (Gauss-Newton)
        for it in range(1, options.max_iterations + 1):
            # Build design matrix and misclosure
            A, w, sigmas_classical, computed_classical, sigmas_leveling, computed_leveling = _linearize_mixed(
                classical_obs, gnss_obs, leveling_obs, state, index, m, n, m_classical, m_gnss, m_leveling
            )

            # Build normal equations with mixed weighting (including robust weights)
            # For classical: diagonal weights (1/sigma^2) * robust_weight
            # For GNSS: block weights (P = C^-1) * robust_weight
            # For leveling: diagonal weights (1/sigma^2) * robust_weight

            N = np.zeros((n, n), dtype=float)
            u = np.zeros(n, dtype=float)

            # Classical contribution (rows 0 to m_classical-1)
            if m_classical > 0:
                A_c = A[:m_classical, :]
                w_c = w[:m_classical]
                base_weights_c = 1.0 / (sigmas_classical ** 2)
                weights_c = base_weights_c * robust_weights_classical

                Aw_c = A_c * weights_c[:, None]
                N += A_c.T @ Aw_c
                u += A_c.T @ (weights_c * w_c)

            # GNSS contribution (rows m_classical to m_classical + m_gnss - 1)
            if m_gnss > 0:
                for i, P_block in enumerate(gnss_weight_blocks):
                    row_base = m_classical + 3 * i
                    A_g = A[row_base:row_base+3, :]
                    l_g = w[row_base:row_base+3]

                    # Apply scalar robust weight to entire 3x3 block
                    P_weighted = P_block * robust_weights_gnss[i]

                    # N += A_g^T @ P @ A_g
                    AtP = A_g.T @ P_weighted
                    N += AtP @ A_g
                    u += AtP @ l_g

            # Leveling contribution (rows m_classical + m_gnss to m - 1)
            if m_leveling > 0:
                row_start = m_classical + m_gnss
                A_l = A[row_start:row_start + m_leveling, :]
                w_l = w[row_start:row_start + m_leveling]
                base_weights_l = 1.0 / (sigmas_leveling ** 2)
                weights_l = base_weights_l * robust_weights_leveling

                Aw_l = A_l * weights_l[:, None]
                N += A_l.T @ Aw_l
                u += A_l.T @ (weights_l * w_l)

            # Solve normal equations
            try:
                dx = np.linalg.solve(N, u)
            except np.linalg.LinAlgError:
                return AdjustmentResult.failure(
                    "Normal matrix is singular (datum definition / geometry issue)"
                )

            # Apply corrections
            _apply_corrections_mixed(state, index, dx)

            # Convergence check
            coord_max = 0.0
            for (pid, comp), idxp in index.coord_index.items():
                coord_max = max(coord_max, abs(dx[idxp]))
            orient_max = 0.0
            for set_id, idxo in index.orientation_index.items():
                orient_max = max(orient_max, abs(dx[idxo]))

            if coord_max <= options.convergence_threshold and orient_max <= orient_tol:
                converged = True
                break

        # After G-N convergence, update robust weights if enabled
        if robust_method != RobustMethod.NONE and irls_iter < max_irls:
            # Recompute for standardized residuals
            A_tmp, w_tmp, sigmas_c_tmp, _, sigmas_l_tmp, _ = _linearize_mixed(
                classical_obs, gnss_obs, leveling_obs, state, index, m, n, m_classical, m_gnss, m_leveling
            )
            residuals_tmp = w_tmp

            # Build full Pdiag with robust weights
            Pdiag_tmp = np.zeros(m, dtype=float)
            if m_classical > 0:
                Pdiag_tmp[:m_classical] = (1.0 / (sigmas_c_tmp ** 2)) * robust_weights_classical
            for i, P_block in enumerate(gnss_weight_blocks):
                row_base = m_classical + 3 * i
                Pdiag_tmp[row_base + 0] = P_block[0, 0] * robust_weights_gnss[i]
                Pdiag_tmp[row_base + 1] = P_block[1, 1] * robust_weights_gnss[i]
                Pdiag_tmp[row_base + 2] = P_block[2, 2] * robust_weights_gnss[i]
            if m_leveling > 0:
                row_start = m_classical + m_gnss
                Pdiag_tmp[row_start:row_start + m_leveling] = (1.0 / (sigmas_l_tmp ** 2)) * robust_weights_leveling

            vTPv_tmp = float((residuals_tmp * residuals_tmp * Pdiag_tmp).sum())
            # Add GNSS cross-terms
            for i, P_block in enumerate(gnss_weight_blocks):
                row_base = m_classical + 3 * i
                v_g = residuals_tmp[row_base:row_base+3]
                vTPv_tmp += float(v_g @ P_block @ v_g) * robust_weights_gnss[i] - \
                           float((v_g * v_g * np.diag(P_block)).sum()) * robust_weights_gnss[i]

            # Use a-priori sigma0 for IRLS to avoid masking effect where outliers
            # inflate the a-posteriori sigma0 and prevent their own detection
            sigma0_tmp = math.sqrt(max(options.a_priori_variance, 1e-30))

            # Compute standardized residuals
            try:
                # Rebuild N with robust weights
                N_tmp = np.zeros((n, n), dtype=float)
                if m_classical > 0:
                    A_c = A_tmp[:m_classical, :]
                    weights_c = (1.0 / (sigmas_c_tmp ** 2)) * robust_weights_classical
                    N_tmp += (A_c.T * weights_c) @ A_c
                for i, P_block in enumerate(gnss_weight_blocks):
                    row_base = m_classical + 3 * i
                    A_g = A_tmp[row_base:row_base+3, :]
                    N_tmp += A_g.T @ (P_block * robust_weights_gnss[i]) @ A_g
                if m_leveling > 0:
                    row_start = m_classical + m_gnss
                    A_l = A_tmp[row_start:row_start + m_leveling, :]
                    weights_l = (1.0 / (sigmas_l_tmp ** 2)) * robust_weights_leveling
                    N_tmp += (A_l.T * weights_l) @ A_l

                Qxx_tmp = np.linalg.solve(N_tmp, np.eye(n))
                B_tmp = A_tmp @ Qxx_tmp
                diag_AQAt = (B_tmp * A_tmp).sum(axis=1)

                # Build Qll diagonal
                Qll_diag = np.zeros(m, dtype=float)
                if m_classical > 0:
                    Qll_diag[:m_classical] = sigmas_c_tmp ** 2
                for i, C in enumerate(gnss_cov_blocks):
                    row_base = m_classical + 3 * i
                    Qll_diag[row_base + 0] = C[0, 0]
                    Qll_diag[row_base + 1] = C[1, 1]
                    Qll_diag[row_base + 2] = C[2, 2]
                if m_leveling > 0:
                    row_start = m_classical + m_gnss
                    Qll_diag[row_start:row_start + m_leveling] = sigmas_l_tmp ** 2

                qvv_diag_tmp = Qll_diag - diag_AQAt
                qvv_diag_tmp = np.maximum(qvv_diag_tmp, 1e-30)
                std_res_tmp = residuals_tmp / (sigma0_tmp * np.sqrt(qvv_diag_tmp))
            except np.linalg.LinAlgError:
                # Fallback
                std_res_tmp = np.zeros(m)
                if m_classical > 0:
                    std_res_tmp[:m_classical] = residuals_tmp[:m_classical] / (sigmas_c_tmp * sigma0_tmp)
                for i, C in enumerate(gnss_cov_blocks):
                    row_base = m_classical + 3 * i
                    std_res_tmp[row_base + 0] = residuals_tmp[row_base + 0] / (math.sqrt(C[0, 0]) * sigma0_tmp)
                    std_res_tmp[row_base + 1] = residuals_tmp[row_base + 1] / (math.sqrt(C[1, 1]) * sigma0_tmp)
                    std_res_tmp[row_base + 2] = residuals_tmp[row_base + 2] / (math.sqrt(C[2, 2]) * sigma0_tmp)
                if m_leveling > 0:
                    row_start = m_classical + m_gnss
                    std_res_tmp[row_start:row_start + m_leveling] = residuals_tmp[row_start:row_start + m_leveling] / (sigmas_l_tmp * sigma0_tmp)

            # Update robust weights
            max_weight_change = 0.0

            # Classical
            if m_classical > 0:
                new_weights_c = compute_robust_weights(std_res_tmp[:m_classical], weight_func)
                max_weight_change = max(max_weight_change, np.max(np.abs(new_weights_c - robust_weights_classical)))
                robust_weights_classical = new_weights_c

            # GNSS (use max standardized residual per baseline)
            if len(gnss_obs) > 0:
                baseline_std_res = np.zeros(len(gnss_obs))
                for i in range(len(gnss_obs)):
                    row_base = m_classical + 3 * i
                    baseline_std_res[i] = max(
                        abs(std_res_tmp[row_base + 0]),
                        abs(std_res_tmp[row_base + 1]),
                        abs(std_res_tmp[row_base + 2]),
                    )
                new_weights_gnss = compute_robust_weights(baseline_std_res, weight_func)
                max_weight_change = max(max_weight_change, np.max(np.abs(new_weights_gnss - robust_weights_gnss)))
                robust_weights_gnss = new_weights_gnss

            # Leveling
            if m_leveling > 0:
                row_start = m_classical + m_gnss
                new_weights_l = compute_robust_weights(std_res_tmp[row_start:row_start + m_leveling], weight_func)
                max_weight_change = max(max_weight_change, np.max(np.abs(new_weights_l - robust_weights_leveling)))
                robust_weights_leveling = new_weights_l

            robust_iterations = irls_iter

            if max_weight_change < options.robust_tol:
                robust_converged = True
                robust_message = f"IRLS converged after {irls_iter} iterations"
                break
        else:
            robust_iterations = irls_iter if robust_method != RobustMethod.NONE else 0
            break

    if robust_method != RobustMethod.NONE and robust_iterations >= max_irls:
        robust_converged = False
        robust_message = f"IRLS did not converge after {max_irls} iterations"

    # Final computation for residuals and statistics
    A_fin, w_fin, sigmas_fin, computed_fin, sigmas_lev_fin, computed_lev_fin = _linearize_mixed(
        classical_obs, gnss_obs, leveling_obs, state, index, m, n, m_classical, m_gnss, m_leveling
    )
    residuals = w_fin

    # Compute vTPv (weighted sum of squared residuals) with final robust weights
    vTPv = 0.0

    # Classical contribution
    if m_classical > 0:
        v_c = residuals[:m_classical]
        base_weights_c = 1.0 / (sigmas_fin ** 2)
        weights_c = base_weights_c * robust_weights_classical
        vTPv += float((v_c * v_c * weights_c).sum())

    # GNSS contribution
    for i, P_block in enumerate(gnss_weight_blocks):
        row_base = m_classical + 3 * i
        v_g = residuals[row_base:row_base+3]
        vTPv += float(v_g @ (P_block * robust_weights_gnss[i]) @ v_g)

    # Leveling contribution
    if m_leveling > 0:
        row_start = m_classical + m_gnss
        v_l = residuals[row_start:row_start + m_leveling]
        base_weights_l = 1.0 / (sigmas_lev_fin ** 2)
        weights_l = base_weights_l * robust_weights_leveling
        vTPv += float((v_l * v_l * weights_l).sum())

    dof = m - n

    if dof > 0:
        sigma0_sq = vTPv / dof
        variance_factor = sigma0_sq / options.a_priori_variance
    else:
        variance_factor = 1.0
        sigma0_sq = options.a_priori_variance

    sigma0_sq_post = variance_factor * options.a_priori_variance

    # Compute Qxx (cofactor matrix of unknowns)
    Qxx: Optional[np.ndarray] = None
    cov_matrix: Optional[np.ndarray] = None

    if options.compute_covariances:
        # Rebuild N for Qxx (with robust weights)
        N_fin = np.zeros((n, n), dtype=float)

        if m_classical > 0:
            A_c = A_fin[:m_classical, :]
            base_weights_c = 1.0 / (sigmas_fin ** 2)
            weights_c = base_weights_c * robust_weights_classical
            N_fin += (A_c.T * weights_c) @ A_c

        for i, P_block in enumerate(gnss_weight_blocks):
            row_base = m_classical + 3 * i
            A_g = A_fin[row_base:row_base+3, :]
            N_fin += A_g.T @ (P_block * robust_weights_gnss[i]) @ A_g

        if m_leveling > 0:
            row_start = m_classical + m_gnss
            A_l = A_fin[row_start:row_start + m_leveling, :]
            base_weights_l = 1.0 / (sigmas_lev_fin ** 2)
            weights_l = base_weights_l * robust_weights_leveling
            N_fin += (A_l.T * weights_l) @ A_l

        try:
            Qxx = np.linalg.solve(N_fin, np.eye(n))
            cov_matrix = sigma0_sq_post * Qxx
        except np.linalg.LinAlgError:
            Qxx = None
            cov_matrix = None

    # Build adjusted points with posterior sigmas
    adjusted_points: Dict[str, Point] = {}
    point_covs: Dict[str, np.ndarray] = {}
    error_ellipses: Dict[str, ErrorEllipse] = {}

    for pid, p in network.points.items():
        e, nn, h = state.points[pid]

        sigma_e = None
        sigma_n = None
        sigma_h = None

        if cov_matrix is not None:
            if (pid, 'E') in index.coord_index:
                ie = index.coord_index[(pid, 'E')]
                sigma_e = math.sqrt(max(cov_matrix[ie, ie], 0.0))
            elif p.fixed_easting:
                sigma_e = 0.0

            if (pid, 'N') in index.coord_index:
                in_ = index.coord_index[(pid, 'N')]
                sigma_n = math.sqrt(max(cov_matrix[in_, in_], 0.0))
            elif p.fixed_northing:
                sigma_n = 0.0

            if (pid, 'H') in index.coord_index:
                ih = index.coord_index[(pid, 'H')]
                sigma_h = math.sqrt(max(cov_matrix[ih, ih], 0.0))
            elif p.fixed_height:
                sigma_h = 0.0

            # Build point covariance (2x2 for EN, or 3x3 if height included)
            has_height_obs = has_gnss or has_leveling
            if has_height_obs:
                cov_3x3 = np.zeros((3, 3), dtype=float)
                indices = []
                for comp, key in enumerate(['E', 'N', 'H']):
                    if (pid, key) in index.coord_index:
                        indices.append((comp, index.coord_index[(pid, key)]))
                    else:
                        indices.append((comp, None))

                for ci, idx_i in indices:
                    for cj, idx_j in indices:
                        if idx_i is not None and idx_j is not None:
                            cov_3x3[ci, cj] = cov_matrix[idx_i, idx_j]

                point_covs[pid] = cov_3x3
            else:
                cov_2x2 = np.zeros((2, 2), dtype=float)
                if (pid, 'E') in index.coord_index:
                    ie = index.coord_index[(pid, 'E')]
                    cov_2x2[0, 0] = cov_matrix[ie, ie]
                if (pid, 'N') in index.coord_index:
                    in_ = index.coord_index[(pid, 'N')]
                    cov_2x2[1, 1] = cov_matrix[in_, in_]
                if (pid, 'E') in index.coord_index and (pid, 'N') in index.coord_index:
                    ie = index.coord_index[(pid, 'E')]
                    in_ = index.coord_index[(pid, 'N')]
                    cov_2x2[0, 1] = cov_matrix[ie, in_]
                    cov_2x2[1, 0] = cov_matrix[in_, ie]
                point_covs[pid] = cov_2x2

        # Use adjusted height if GNSS or leveling present, otherwise original
        has_height_obs = has_gnss or has_leveling
        point_h = h if has_height_obs else p.height

        adjusted_points[pid] = Point(
            id=p.id,
            name=p.name,
            easting=e,
            northing=nn,
            height=point_h,
            fixed_easting=p.fixed_easting,
            fixed_northing=p.fixed_northing,
            fixed_height=p.fixed_height,
            sigma_easting=sigma_e,
            sigma_northing=sigma_n,
            sigma_height=sigma_h,
        )

        # Compute error ellipse (2D for horizontal)
        if options.compute_error_ellipses and cov_matrix is not None:
            if (pid, 'E') in index.coord_index and (pid, 'N') in index.coord_index:
                ie = index.coord_index[(pid, 'E')]
                in_ = index.coord_index[(pid, 'N')]
                cov_2x2 = np.array([
                    [cov_matrix[ie, ie], cov_matrix[ie, in_]],
                    [cov_matrix[in_, ie], cov_matrix[in_, in_]],
                ])
                ellipse = _compute_error_ellipse(pid, cov_2x2, options.confidence_level)
                error_ellipses[pid] = ellipse

    # Compute standardized residuals and reliability measures
    sigma0_hat = max(math.sqrt(sigma0_sq_post), 1e-12) if dof > 0 else 1.0
    sigma0_for_w = max(math.sqrt(options.a_priori_variance), 1e-12)

    # Build diagonal of Qvv for each observation
    qvv_diag: Optional[np.ndarray] = None

    if dof > 0 and Qxx is not None:
        # diag(Qvv) = diag(Qll) - diag(A Qxx A^T)
        # Qll is the cofactor matrix of observations

        # For classical: Qll_ii = sigma_i^2
        # For GNSS: Qll is block diagonal with covariance blocks
        # For leveling: Qll_ii = sigma_i^2

        B = A_fin @ Qxx
        diag_AQAt = (B * A_fin).sum(axis=1)

        # Build Qll diagonal
        Qll_diag = np.zeros(m, dtype=float)

        # Classical
        if m_classical > 0:
            Qll_diag[:m_classical] = sigmas_fin ** 2

        # GNSS (diagonal of covariance blocks)
        for i, C in enumerate(gnss_cov_blocks):
            row_base = m_classical + 3 * i
            Qll_diag[row_base + 0] = C[0, 0]
            Qll_diag[row_base + 1] = C[1, 1]
            Qll_diag[row_base + 2] = C[2, 2]

        # Leveling
        if m_leveling > 0:
            row_start = m_classical + m_gnss
            Qll_diag[row_start:row_start + m_leveling] = sigmas_lev_fin ** 2

        qvv_diag = Qll_diag - diag_AQAt
        qvv_diag = np.maximum(qvv_diag, 1e-30)

    # Local test threshold
    k_alpha = local_outlier_threshold(options.alpha_local)
    k_beta = normal_ppf(options.mdb_power)

    # Compute standardized residuals
    std_residuals: Dict[str, float] = {}
    residual_details: List[ResidualInfo] = []
    flagged: List[str] = []

    # Build P diagonal for reliability computations (with robust weights)
    Pdiag = np.zeros(m, dtype=float)
    if m_classical > 0:
        base_Pdiag_c = 1.0 / (sigmas_fin ** 2)
        Pdiag[:m_classical] = base_Pdiag_c * robust_weights_classical
    for i, P_block in enumerate(gnss_weight_blocks):
        row_base = m_classical + 3 * i
        Pdiag[row_base + 0] = P_block[0, 0] * robust_weights_gnss[i]
        Pdiag[row_base + 1] = P_block[1, 1] * robust_weights_gnss[i]
        Pdiag[row_base + 2] = P_block[2, 2] * robust_weights_gnss[i]
    if m_leveling > 0:
        row_start = m_classical + m_gnss
        base_Pdiag_l = 1.0 / (sigmas_lev_fin ** 2)
        Pdiag[row_start:row_start + m_leveling] = base_Pdiag_l * robust_weights_leveling

    # Reliability measures
    r_vals = None
    mdb_vals = None
    ext_rel = None

    if options.compute_reliability and Qxx is not None and qvv_diag is not None and dof > 0:
        r_vals = redundancy_numbers(qvv_diag, Pdiag)

        # Build sigma array for MDB (use observation sigmas for all types)
        sigmas_all = np.zeros(m, dtype=float)
        if m_classical > 0:
            sigmas_all[:m_classical] = sigmas_fin
        for i, C in enumerate(gnss_cov_blocks):
            row_base = m_classical + 3 * i
            sigmas_all[row_base + 0] = math.sqrt(C[0, 0])
            sigmas_all[row_base + 1] = math.sqrt(C[1, 1])
            sigmas_all[row_base + 2] = math.sqrt(C[2, 2])
        if m_leveling > 0:
            row_start = m_classical + m_gnss
            sigmas_all[row_start:row_start + m_leveling] = sigmas_lev_fin

        mdb_vals = mdb_values(k_alpha, k_beta, sigma0_hat, sigmas_all, r_vals)

        coord_param_indices = [idx for (pid, comp), idx in index.coord_index.items()]
        ext_rel = external_reliability(Qxx, A_fin, Pdiag, mdb_vals, coord_param_indices)

    # Process classical residuals
    for j, obs in enumerate(classical_obs):
        res = residuals[j]
        comp_val = computed_fin[j]

        # Compute standardized residual with safeguard against division by zero
        qvv_var = qvv_diag[j] * sigma0_sq_post if qvv_diag is not None else 0.0
        if qvv_var > 1e-30:
            std = res / math.sqrt(qvv_var)
        elif sigmas_fin[j] * sigma0_hat > 1e-30:
            std = res / (sigmas_fin[j] * sigma0_hat)
        else:
            std = 0.0  # No redundancy - cannot compute meaningful standardized residual

        obs_type = obs.obs_type.value
        is_candidate = abs(float(std)) > float(k_alpha)
        is_flagged = abs(float(std)) > float(options.outlier_threshold)

        info = ResidualInfo(
            obs_id=obs.id,
            obs_type=obs_type,
            observed=float(obs.value),
            computed=float(comp_val),
            residual=float(res),
            standardized_residual=float(std),
            redundancy_number=float(r_vals[j]) if r_vals is not None else None,
            mdb=float(mdb_vals[j]) if mdb_vals is not None else None,
            external_reliability=float(ext_rel[j]) if ext_rel is not None else None,
            is_outlier_candidate=is_candidate,
            flagged=is_flagged,
            weight_factor=float(robust_weights_classical[j]) if robust_method != RobustMethod.NONE else None,
        )

        if isinstance(obs, (DistanceObservation, DirectionObservation)):
            info.from_point = obs.from_point_id
            info.to_point = obs.to_point_id
        elif isinstance(obs, AngleObservation):
            info.at_point = obs.at_point_id
            info.from_point = obs.from_point_id
            info.to_point = obs.to_point_id

        std_residuals[obs.id] = float(std)
        residual_details.append(info)
        if is_flagged:
            flagged.append(obs.id)

    # Process GNSS baseline residuals
    for i, obs in enumerate(gnss_obs):
        row_base = m_classical + 3 * i

        vE = residuals[row_base + 0]
        vN = residuals[row_base + 1]
        vH = residuals[row_base + 2]

        # Standardized residuals for each component with safeguard against division by zero
        C = gnss_cov_blocks[i]

        qvv_E = qvv_diag[row_base + 0] * sigma0_sq_post if qvv_diag is not None else 0.0
        if qvv_E > 1e-30:
            wE = vE / math.sqrt(qvv_E)
        elif C[0, 0] * sigma0_sq_post > 1e-30:
            wE = vE / (math.sqrt(C[0, 0]) * sigma0_hat)
        else:
            wE = 0.0

        qvv_N = qvv_diag[row_base + 1] * sigma0_sq_post if qvv_diag is not None else 0.0
        if qvv_N > 1e-30:
            wN = vN / math.sqrt(qvv_N)
        elif C[1, 1] * sigma0_sq_post > 1e-30:
            wN = vN / (math.sqrt(C[1, 1]) * sigma0_hat)
        else:
            wN = 0.0

        qvv_H = qvv_diag[row_base + 2] * sigma0_sq_post if qvv_diag is not None else 0.0
        if qvv_H > 1e-30:
            wH = vH / math.sqrt(qvv_H)
        elif C[2, 2] * sigma0_sq_post > 1e-30:
            wH = vH / (math.sqrt(C[2, 2]) * sigma0_hat)
        else:
            wH = 0.0

        w_max = max(abs(wE), abs(wN), abs(wH))

        # Average redundancy
        r_avg = None
        if r_vals is not None:
            r_avg = (r_vals[row_base] + r_vals[row_base + 1] + r_vals[row_base + 2]) / 3.0

        is_candidate = bool(w_max > k_alpha)
        is_flagged = bool(w_max > options.outlier_threshold)

        # Computed baseline length
        from_coords = state.points[obs.from_point_id]
        to_coords = state.points[obs.to_point_id]
        computed_dE = to_coords[0] - from_coords[0]
        computed_dN = to_coords[1] - from_coords[1]
        computed_dH = to_coords[2] - from_coords[2]
        computed_length = math.sqrt(computed_dE**2 + computed_dN**2 + computed_dH**2)

        info = ResidualInfo(
            obs_id=obs.id,
            obs_type="gnss_baseline",
            observed=obs.baseline_length,
            computed=computed_length,
            residual=math.sqrt(vE**2 + vN**2 + vH**2),  # 3D residual magnitude
            standardized_residual=w_max,
            redundancy_number=r_avg,
            mdb=None,  # MDB complex for correlated obs
            external_reliability=None,
            is_outlier_candidate=is_candidate,
            flagged=is_flagged,
            from_point=obs.from_point_id,
            to_point=obs.to_point_id,
            weight_factor=float(robust_weights_gnss[i]) if robust_method != RobustMethod.NONE else None,
        )

        # Store component residuals as attributes
        info._vE = vE
        info._vN = vN
        info._vH = vH
        info._wE = wE
        info._wN = wN
        info._wH = wH

        std_residuals[obs.id] = w_max
        residual_details.append(info)
        if is_flagged:
            flagged.append(obs.id)

    # Process leveling residuals
    for j, obs in enumerate(leveling_obs):
        row_idx = m_classical + m_gnss + j
        res = residuals[row_idx]
        comp_val = computed_lev_fin[j]

        # Compute standardized residual with safeguard against division by zero
        qvv_var = qvv_diag[row_idx] * sigma0_sq_post if qvv_diag is not None else 0.0
        if qvv_var > 1e-30:
            std = res / math.sqrt(qvv_var)
        elif sigmas_lev_fin[j] * sigma0_hat > 1e-30:
            std = res / (sigmas_lev_fin[j] * sigma0_hat)
        else:
            std = 0.0  # No redundancy - cannot compute meaningful standardized residual

        is_candidate = abs(float(std)) > float(k_alpha)
        is_flagged = abs(float(std)) > float(options.outlier_threshold)

        info = ResidualInfo(
            obs_id=obs.id,
            obs_type="height_diff",
            observed=float(obs.value),
            computed=float(comp_val),
            residual=float(res),
            standardized_residual=float(std),
            redundancy_number=float(r_vals[row_idx]) if r_vals is not None else None,
            mdb=float(mdb_vals[row_idx]) if mdb_vals is not None else None,
            external_reliability=float(ext_rel[row_idx]) if ext_rel is not None else None,
            is_outlier_candidate=is_candidate,
            flagged=is_flagged,
            from_point=obs.from_point_id,
            to_point=obs.to_point_id,
            weight_factor=float(robust_weights_leveling[j]) if robust_method != RobustMethod.NONE else None,
        )

        std_residuals[obs.id] = float(std)
        residual_details.append(info)
        if is_flagged:
            flagged.append(obs.id)

    # Chi-square global test
    chi_test = None
    if dof > 0:
        chi_test = chi_square_global_test(
            vTPv=vTPv,
            dof=dof,
            alpha=options.alpha,
            a_priori_variance=options.a_priori_variance,
        )

    # Build residuals dict
    residuals_dict: Dict[str, float] = {}
    for j, obs in enumerate(classical_obs):
        residuals_dict[obs.id] = float(residuals[j])
    for i, obs in enumerate(gnss_obs):
        row_base = m_classical + 3 * i
        # Store 3D magnitude for GNSS
        vE = residuals[row_base + 0]
        vN = residuals[row_base + 1]
        vH = residuals[row_base + 2]
        residuals_dict[obs.id] = math.sqrt(vE**2 + vN**2 + vH**2)
    for j, obs in enumerate(leveling_obs):
        row_idx = m_classical + m_gnss + j
        residuals_dict[obs.id] = float(residuals[row_idx])

    messages = []
    if dof == 0:
        messages.append("No redundancy (dof=0): variance factor set to 1.0")

    # Build message about observation types
    obs_parts = []
    if has_classical:
        obs_parts.append(f"{len(classical_obs)} classical")
    if has_gnss:
        obs_parts.append(f"{len(gnss_obs)} GNSS baselines")
    if has_leveling:
        obs_parts.append(f"{len(leveling_obs)} leveling")

    if len(obs_parts) > 1:
        messages.append(f"Mixed adjustment: {' + '.join(obs_parts)}")

    return AdjustmentResult(
        success=True,
        iterations=min(options.max_iterations, it if 'it' in locals() else 0),
        converged=converged,
        adjusted_points=adjusted_points,
        residuals=residuals_dict,
        standardized_residuals=std_residuals,
        residual_details=residual_details,
        degrees_of_freedom=dof,
        variance_factor=float(variance_factor),
        chi_square_test=chi_test,
        covariance_matrix=cov_matrix,
        point_covariances=point_covs,
        error_ellipses=error_ellipses,
        flagged_observations=flagged,
        messages=messages,
        network_name=network.name,
        # Robust estimation fields (Phase 7A)
        robust_method=robust_method.value if robust_method != RobustMethod.NONE else None,
        robust_iterations=robust_iterations,
        robust_converged=robust_converged,
        robust_message=robust_message,
    )


def _linearize_mixed(
    classical_obs: List[Observation],
    gnss_obs: List[GnssBaselineObservation],
    leveling_obs: List[HeightDifferenceObservation],
    state: _State,
    index: MixedParameterIndex,
    m: int,
    n: int,
    m_classical: int,
    m_gnss: int,
    m_leveling: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float], np.ndarray, List[float]]:
    """Build design matrix A and misclosure vector w for mixed observations.

    Returns:
        A: Design matrix (m x n)
        w: Misclosure vector (m,)
        sigmas_classical: Standard deviations for classical obs
        computed_classical: Computed values for classical obs
        sigmas_leveling: Standard deviations for leveling obs
        computed_leveling: Computed values for leveling obs
    """
    A = np.zeros((m, n), dtype=float)
    w = np.zeros(m, dtype=float)

    sigmas_classical = np.zeros(m_classical, dtype=float)
    computed_classical: List[float] = []
    sigmas_leveling = np.zeros(m_leveling, dtype=float)
    computed_leveling: List[float] = []

    # Classical observations (rows 0 to m_classical-1)
    for i, obs in enumerate(classical_obs):
        sigmas_classical[i] = float(obs.sigma)

        if isinstance(obs, DistanceObservation):
            e1, n1, _ = state.points[obs.from_point_id]
            e2, n2, _ = state.points[obs.to_point_id]
            comp = distance_2d(e1, n1, e2, n2)
            computed_classical.append(comp)
            w[i] = float(obs.value) - comp

            dE1, dN1, dE2, dN2 = distance_partials(e1, n1, e2, n2)
            _set_A_coord(A, i, index, obs.from_point_id, dE1, dN1)
            _set_A_coord(A, i, index, obs.to_point_id, dE2, dN2)

        elif isinstance(obs, DirectionObservation):
            e1, n1, _ = state.points[obs.from_point_id]
            e2, n2, _ = state.points[obs.to_point_id]
            az = azimuth(e1, n1, e2, n2)
            omega = state.orientations.get(obs.set_id, 0.0)
            comp = wrap_2pi(az + omega)
            computed_classical.append(comp)
            w[i] = wrap_pi(float(obs.value) - (az + omega))

            dE1, dN1, dE2, dN2 = azimuth_partials(e1, n1, e2, n2)
            _set_A_coord(A, i, index, obs.from_point_id, dE1, dN1)
            _set_A_coord(A, i, index, obs.to_point_id, dE2, dN2)

            if obs.set_id in index.orientation_index:
                A[i, index.orientation_index[obs.set_id]] = 1.0

        elif isinstance(obs, AngleObservation):
            e_at, n_at, _ = state.points[obs.at_point_id]
            e_from, n_from, _ = state.points[obs.from_point_id]
            e_to, n_to, _ = state.points[obs.to_point_id]
            comp = angle_at_point(e_at, n_at, e_from, n_from, e_to, n_to)
            computed_classical.append(comp)
            w[i] = wrap_pi(float(obs.value) - comp)

            dE_from, dN_from, dE_at, dN_at, dE_to, dN_to = angle_partials(
                e_at, n_at, e_from, n_from, e_to, n_to
            )
            _set_A_coord(A, i, index, obs.from_point_id, dE_from, dN_from)
            _set_A_coord(A, i, index, obs.at_point_id, dE_at, dN_at)
            _set_A_coord(A, i, index, obs.to_point_id, dE_to, dN_to)

    # GNSS observations (rows m_classical to m_classical + m_gnss - 1)
    for i, obs in enumerate(gnss_obs):
        row_base = m_classical + 3 * i

        from_pid = obs.from_point_id
        to_pid = obs.to_point_id

        from_E, from_N, from_H = state.points[from_pid]
        to_E, to_N, to_H = state.points[to_pid]

        # Design matrix: dE = E_to - E_from, etc.
        # E component (row_base + 0)
        if (to_pid, 'E') in index.coord_index:
            A[row_base + 0, index.coord_index[(to_pid, 'E')]] = 1.0
        if (from_pid, 'E') in index.coord_index:
            A[row_base + 0, index.coord_index[(from_pid, 'E')]] = -1.0

        # N component (row_base + 1)
        if (to_pid, 'N') in index.coord_index:
            A[row_base + 1, index.coord_index[(to_pid, 'N')]] = 1.0
        if (from_pid, 'N') in index.coord_index:
            A[row_base + 1, index.coord_index[(from_pid, 'N')]] = -1.0

        # H component (row_base + 2)
        if (to_pid, 'H') in index.coord_index:
            A[row_base + 2, index.coord_index[(to_pid, 'H')]] = 1.0
        if (from_pid, 'H') in index.coord_index:
            A[row_base + 2, index.coord_index[(from_pid, 'H')]] = -1.0

        # Misclosure: l = observed - computed
        computed_dE = to_E - from_E
        computed_dN = to_N - from_N
        computed_dH = to_H - from_H

        w[row_base + 0] = obs.dE - computed_dE
        w[row_base + 1] = obs.dN - computed_dN
        w[row_base + 2] = obs.dH - computed_dH

    # Leveling observations (rows m_classical + m_gnss to m - 1)
    for i, obs in enumerate(leveling_obs):
        row_idx = m_classical + m_gnss + i
        sigmas_leveling[i] = float(obs.sigma)

        from_pid = obs.from_point_id
        to_pid = obs.to_point_id

        _, _, from_H = state.points[from_pid]
        _, _, to_H = state.points[to_pid]

        # Computed height difference
        comp = to_H - from_H
        computed_leveling.append(comp)

        # Misclosure: l = observed - computed
        w[row_idx] = float(obs.value) - comp

        # Design matrix: dH = H_to - H_from
        # Partial w.r.t. H_from = -1, H_to = +1
        if (to_pid, 'H') in index.coord_index:
            A[row_idx, index.coord_index[(to_pid, 'H')]] = 1.0
        if (from_pid, 'H') in index.coord_index:
            A[row_idx, index.coord_index[(from_pid, 'H')]] = -1.0

    return A, w, sigmas_classical, computed_classical, sigmas_leveling, computed_leveling


def _set_A_coord(A: np.ndarray, row: int, index: MixedParameterIndex,
                 point_id: str, dE: float, dN: float) -> None:
    """Helper to set E, N coordinate partials into A if adjustable."""
    if (point_id, 'E') in index.coord_index:
        A[row, index.coord_index[(point_id, 'E')]] += float(dE)
    if (point_id, 'N') in index.coord_index:
        A[row, index.coord_index[(point_id, 'N')]] += float(dN)


def _apply_corrections_mixed(state: _State, index: MixedParameterIndex,
                              dx: np.ndarray) -> None:
    """Apply corrections vector dx to the state."""
    for (pid, comp), j in index.coord_index.items():
        e, n, h = state.points[pid]
        if comp == 'E':
            e += float(dx[j])
        elif comp == 'N':
            n += float(dx[j])
        elif comp == 'H':
            h += float(dx[j])
        state.points[pid] = (e, n, h)

    for set_id, j in index.orientation_index.items():
        state.orientations[set_id] = state.orientations.get(set_id, 0.0) + float(dx[j])


def _compute_error_ellipse(point_id: str, cov2: np.ndarray,
                           confidence: float) -> ErrorEllipse:
    """Compute error ellipse parameters from 2x2 covariance matrix."""
    vals, vecs = np.linalg.eigh(cov2)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    k2 = chi2_ppf(confidence, 2)
    scale = math.sqrt(max(k2, 0.0))

    semi_major = math.sqrt(max(vals[0], 0.0)) * scale
    semi_minor = math.sqrt(max(vals[1], 0.0)) * scale

    vE, vN = float(vecs[0, 0]), float(vecs[1, 0])
    orientation = wrap_2pi(math.atan2(vE, vN))

    return ErrorEllipse(
        point_id=point_id,
        semi_major=float(semi_major),
        semi_minor=float(semi_minor),
        orientation=float(orientation),
        confidence_level=float(confidence),
    )
