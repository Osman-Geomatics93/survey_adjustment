"""survey_adjustment.core.solver.least_squares_2d

2D least-squares adjustment (Gauss-Newton / linearized weighted LS).

Supported observation types:
  - DistanceObservation
  - DirectionObservation (with orientation unknown per set_id)
  - AngleObservation

Supports robust estimation via IRLS (Iteratively Reweighted Least Squares)
with Huber, Danish, and IGG-III weight functions.

This module intentionally contains **no QGIS imports** so it can be unit-
tested and re-used in non-QGIS contexts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore

from ..models.network import Network
from ..models.options import AdjustmentOptions, RobustEstimator
from ..models.point import Point
from ..models.observation import (
    Observation,
    DistanceObservation,
    DirectionObservation,
    AngleObservation,
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
from .indexing import build_parameter_index, validate_network_for_adjustment
from .robust import (
    RobustMethod,
    get_weight_function,
    compute_robust_weights,
    describe_method,
)
from ..validation import (
    analyze_constraint_health,
    apply_minimal_constraints,
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


@dataclass
class _State:
    points: Dict[str, Tuple[float, float]]          # id -> (E, N)
    orientations: Dict[str, float]                  # set_id -> omega (rad)


def adjust_network_2d(network: Network, options: AdjustmentOptions | None = None) -> AdjustmentResult:
    """Run a 2D least-squares adjustment.

    Args:
        network: Input network (points + observations)
        options: Adjustment options (defaults if None)

    Returns:
        AdjustmentResult with adjusted coordinates, residuals, covariance, etc.
    """
    if np is None:
        return AdjustmentResult.failure("NumPy is required for the least-squares solver")

    options = options or AdjustmentOptions.default()

    # Analyze constraint health (Phase 7B)
    constraint_health = analyze_constraint_health(network, adjustment_type="2d")
    applied_constraints = []

    # Apply auto-datum if enabled and network is not solvable
    if options.auto_datum and not constraint_health.is_solvable:
        applied_constraints = apply_minimal_constraints(network, adjustment_type="2d")
        # Re-analyze after applying constraints
        constraint_health = analyze_constraint_health(network, adjustment_type="2d")
        constraint_health.applied_constraints = applied_constraints

    index = build_parameter_index(network, use_direction_orientations=True)
    errors = validate_network_for_adjustment(network, index)
    if errors:
        # Include constraint health in failure result
        result = AdjustmentResult.failure("; ".join(errors))
        result.datum_summary = constraint_health.to_dict()
        result.applied_auto_constraints = [c.to_dict() for c in applied_constraints]
        return result

    enabled_obs: List[Observation] = list(network.get_enabled_observations())
    m = len(enabled_obs)
    n = index.num_params

    # Build initial state
    state = _State(
        points={pid: (p.easting, p.northing) for pid, p in network.points.items()},
        orientations={set_id: 0.0 for set_id in index.orientation_order},
    )

    span = _network_span(network.points)
    orient_tol = max(1e-14, options.convergence_threshold / span)

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

    # Initialize robust weights to 1.0
    robust_weights = np.ones(m)
    robust_converged = True
    robust_iterations = 0
    robust_message: Optional[str] = None

    # Outer IRLS loop (only if robust method is enabled)
    max_irls = options.robust_max_iterations if robust_method != RobustMethod.NONE else 1

    for irls_iter in range(1, max_irls + 1):
        # Reset state for each IRLS iteration (re-solve from initial approximation)
        if irls_iter > 1:
            state = _State(
                points={pid: (p.easting, p.northing) for pid, p in network.points.items()},
                orientations={set_id: 0.0 for set_id in index.orientation_order},
            )

        converged = False
        last_dx = None

        # Inner Gauss-Newton iteration loop
        for it in range(1, options.max_iterations + 1):
            A, w, sigmas, computed = _linearize(enabled_obs, state, index)

            # Base weights from sigmas, modified by robust weights
            base_weights = 1.0 / (sigmas ** 2)
            weights = base_weights * robust_weights

            Aw = A * weights[:, None]
            N = A.T @ Aw
            u = A.T @ (weights * w)

            try:
                dx = np.linalg.solve(N, u)
            except np.linalg.LinAlgError:
                return AdjustmentResult.failure("Normal matrix is singular (datum definition / geometry issue)")

            last_dx = dx
            _apply_corrections(state, index, dx)

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

        # After G-N convergence, compute standardized residuals for IRLS update
        if robust_method != RobustMethod.NONE and irls_iter < max_irls:
            A_tmp, w_tmp, sigmas_tmp, _ = _linearize(enabled_obs, state, index)
            residuals_tmp = w_tmp
            Pdiag_tmp = 1.0 / (sigmas_tmp ** 2) * robust_weights
            vTPv_tmp = float((residuals_tmp * residuals_tmp * Pdiag_tmp).sum())
            dof_tmp = m - n

            # Use a-priori sigma0 for IRLS to avoid masking effect where outliers
            # inflate the a-posteriori sigma0 and prevent their own detection
            sigma0_tmp = math.sqrt(max(options.a_priori_variance, 1e-30))

            # Compute qvv for standardized residuals
            try:
                N_tmp = (A_tmp.T * Pdiag_tmp) @ A_tmp
                Qxx_tmp = np.linalg.solve(N_tmp, np.eye(n))
                B_tmp = A_tmp @ Qxx_tmp
                diag_AQAt = (B_tmp * A_tmp).sum(axis=1)
                qvv_diag_tmp = (1.0 / Pdiag_tmp) - diag_AQAt
                qvv_diag_tmp = np.maximum(qvv_diag_tmp, 1e-30)
                std_res_tmp = residuals_tmp / (sigma0_tmp * np.sqrt(qvv_diag_tmp))
            except np.linalg.LinAlgError:
                std_res_tmp = residuals_tmp / (sigmas_tmp * sigma0_tmp)

            # Compute new robust weights
            new_robust_weights = compute_robust_weights(std_res_tmp, weight_func)

            # Check IRLS convergence
            weight_change = np.abs(new_robust_weights - robust_weights)
            max_weight_change = np.max(weight_change)

            robust_weights = new_robust_weights
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

    # Final recomputation for residuals and statistics
    A_fin, w_fin, sigmas_fin, computed_fin = _linearize(enabled_obs, state, index)
    residuals = w_fin  # l - f(x_hat) (wrapped for angular obs)

    # Apply final robust weights to weight matrix
    base_Pdiag = 1.0 / (sigmas_fin ** 2)
    Pdiag = base_Pdiag * robust_weights
    vTPv = float((residuals * residuals * Pdiag).sum())
    dof = m - n

    if dof > 0:
        sigma0_sq = vTPv / dof
        variance_factor = sigma0_sq / options.a_priori_variance
    else:
        variance_factor = 1.0

    # Covariance matrix
    cov_matrix = None
    point_covs: Dict[str, np.ndarray] = {}
    sigma0_sq_post = variance_factor * options.a_priori_variance

    if options.compute_covariances:
        try:
            # Solve for inverse (more stable than np.linalg.inv)
            Qxx = np.linalg.solve((A_fin.T * Pdiag) @ A_fin, np.eye(n))
            cov_matrix = sigma0_sq_post * Qxx
        except np.linalg.LinAlgError:
            cov_matrix = None

    # Build adjusted points with posterior sigmas
    adjusted_points: Dict[str, Point] = {}
    error_ellipses: Dict[str, ErrorEllipse] = {}

    for pid, p in network.points.items():
        e, nn = state.points[pid]
        sigma_e = None
        sigma_n = None
        cov2 = np.zeros((2, 2), dtype=float)

        if cov_matrix is not None:
            if (pid, 'E') in index.coord_index:
                ie = index.coord_index[(pid, 'E')]
                cov2[0, 0] = float(cov_matrix[ie, ie])
            if (pid, 'N') in index.coord_index:
                in_ = index.coord_index[(pid, 'N')]
                cov2[1, 1] = float(cov_matrix[in_, in_])
            if (pid, 'E') in index.coord_index and (pid, 'N') in index.coord_index:
                ie = index.coord_index[(pid, 'E')]
                in_ = index.coord_index[(pid, 'N')]
                cov2[0, 1] = float(cov_matrix[ie, in_])
                cov2[1, 0] = float(cov_matrix[in_, ie])

            point_covs[pid] = cov2
            sigma_e = math.sqrt(max(cov2[0, 0], 0.0)) if not p.fixed_easting else 0.0
            sigma_n = math.sqrt(max(cov2[1, 1], 0.0)) if not p.fixed_northing else 0.0

        adjusted_points[pid] = Point(
            id=p.id,
            name=p.name,
            easting=e,
            northing=nn,
            fixed_easting=p.fixed_easting,
            fixed_northing=p.fixed_northing,
            sigma_easting=sigma_e,
            sigma_northing=sigma_n,
        )

        if options.compute_error_ellipses and cov_matrix is not None and (pid, 'E') in index.coord_index and (pid, 'N') in index.coord_index:
            ellipse = _compute_error_ellipse(pid, cov2, options.confidence_level)
            error_ellipses[pid] = ellipse


    # Standardized residuals, local test, and reliability measures
    std_residuals: Dict[str, float] = {}
    residual_details: List[ResidualInfo] = []
    flagged: List[str] = []

    sigma0_hat = max(math.sqrt(sigma0_sq_post), 1e-12) if dof > 0 else 1.0
    # For local outlier tests, use a priori sigma0 (data snooping convention)
    sigma0_for_w = max(math.sqrt(options.a_priori_variance), 1e-12)

    qvv_diag = None
    Qxx = None
    if dof > 0:
        # Need Qxx (cofactor of unknowns) for Qvv and reliability.
        try:
            N_fin = (A_fin.T * Pdiag) @ A_fin
            Qxx = np.linalg.solve(N_fin, np.eye(n))
        except np.linalg.LinAlgError:
            Qxx = None

    if Qxx is not None:
        # diag(A Qxx A^T) efficiently
        B = A_fin @ Qxx
        diag_AQAt = (B * A_fin).sum(axis=1)
        qvv_diag = (1.0 / Pdiag) - diag_AQAt
        qvv_diag = np.maximum(qvv_diag, 1e-30)
        std_vals = standardized_residuals(residuals, sigma0_for_w, qvv_diag)
    else:
        # Fallback: normalize by observation sigma only
        std_vals = residuals / (sigmas_fin * max(sigma0_hat, 1e-12))

    # Local test threshold
    k_alpha = local_outlier_threshold(options.alpha_local)
    k_beta = normal_ppf(options.mdb_power)

    r_vals = None
    mdb = None
    ext_rel = None
    if options.compute_reliability and Qxx is not None and qvv_diag is not None and dof > 0:
        r_vals = redundancy_numbers(qvv_diag, Pdiag)
        mdb = mdb_values(k_alpha, k_beta, sigma0_hat, sigmas_fin, r_vals)
        coord_param_indices = list(index.coord_index.values())
        ext_rel = external_reliability(Qxx, A_fin, Pdiag, mdb, coord_param_indices)

    for j, (obs, comp_val, res, std) in enumerate(zip(enabled_obs, computed_fin, residuals, std_vals)):
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
            mdb=float(mdb[j]) if mdb is not None else None,
            external_reliability=float(ext_rel[j]) if ext_rel is not None else None,
            is_outlier_candidate=is_candidate,
            flagged=is_flagged,
            weight_factor=float(robust_weights[j]) if robust_method != RobustMethod.NONE else None,
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

    # Chi-square global test
    chi_test = None
    if dof > 0:
        chi_test = chi_square_global_test(
            vTPv=vTPv,
            dof=dof,
            alpha=options.alpha,
            a_priori_variance=options.a_priori_variance,
        )

    result = AdjustmentResult(
        success=True,
        iterations=min(options.max_iterations, it if 'it' in locals() else 0),
        converged=converged,
        adjusted_points=adjusted_points,
        residuals={o.id: float(r) for o, r in zip(enabled_obs, residuals)},
        standardized_residuals=std_residuals,
        residual_details=residual_details,
        degrees_of_freedom=dof,
        variance_factor=float(variance_factor),
        chi_square_test=chi_test,
        covariance_matrix=cov_matrix,
        point_covariances=point_covs,
        error_ellipses=error_ellipses,
        flagged_observations=flagged,
        messages=[] if dof > 0 else ["No redundancy (dof=0): variance factor set to 1.0"],
        network_name=network.name,
        # Robust estimation fields (Phase 7A)
        robust_method=robust_method.value if robust_method != RobustMethod.NONE else None,
        robust_iterations=robust_iterations,
        robust_converged=robust_converged,
        robust_message=robust_message,
        # Constraint health (Phase 7B)
        datum_summary=constraint_health.to_dict(),
        applied_auto_constraints=[c.to_dict() for c in applied_constraints],
    )

    return result


def _linearize(
    observations: List[Observation],
    state: _State,
    index,
) -> Tuple['np.ndarray', 'np.ndarray', 'np.ndarray', List[float]]:
    """Build design matrix A and misclosure vector w for the current state."""
    m = len(observations)
    n = index.num_params
    A = np.zeros((m, n), dtype=float)
    w = np.zeros(m, dtype=float)
    sig = np.zeros(m, dtype=float)
    computed: List[float] = []

    for i, obs in enumerate(observations):
        sig[i] = float(obs.sigma)
        if isinstance(obs, DistanceObservation):
            e1, n1 = state.points[obs.from_point_id]
            e2, n2 = state.points[obs.to_point_id]
            comp = distance_2d(e1, n1, e2, n2)
            computed.append(comp)
            w[i] = float(obs.value) - comp
            dE1, dN1, dE2, dN2 = distance_partials(e1, n1, e2, n2)
            _set_A_coord(A, i, index, obs.from_point_id, dE1, dN1)
            _set_A_coord(A, i, index, obs.to_point_id, dE2, dN2)

        elif isinstance(obs, DirectionObservation):
            e1, n1 = state.points[obs.from_point_id]
            e2, n2 = state.points[obs.to_point_id]
            az = azimuth(e1, n1, e2, n2)
            omega = state.orientations.get(obs.set_id, 0.0)
            comp = wrap_2pi(az + omega)
            computed.append(comp)
            w[i] = wrap_pi(float(obs.value) - (az + omega))
            dE1, dN1, dE2, dN2 = azimuth_partials(e1, n1, e2, n2)
            _set_A_coord(A, i, index, obs.from_point_id, dE1, dN1)
            _set_A_coord(A, i, index, obs.to_point_id, dE2, dN2)
            if obs.set_id in index.orientation_index:
                A[i, index.orientation_index[obs.set_id]] = 1.0

        elif isinstance(obs, AngleObservation):
            e_at, n_at = state.points[obs.at_point_id]
            e_from, n_from = state.points[obs.from_point_id]
            e_to, n_to = state.points[obs.to_point_id]
            comp = angle_at_point(e_at, n_at, e_from, n_from, e_to, n_to)
            computed.append(comp)
            w[i] = wrap_pi(float(obs.value) - comp)

            dE_from, dN_from, dE_at, dN_at, dE_to, dN_to = angle_partials(
                e_at, n_at, e_from, n_from, e_to, n_to
            )
            _set_A_coord(A, i, index, obs.from_point_id, dE_from, dN_from)
            _set_A_coord(A, i, index, obs.at_point_id, dE_at, dN_at)
            _set_A_coord(A, i, index, obs.to_point_id, dE_to, dN_to)

        else:
            raise ValueError(f"Unsupported observation type: {type(obs)}")

    return A, w, sig, computed


def _set_A_coord(A, row: int, index, point_id: str, dE: float, dN: float) -> None:
    """Helper to set coordinate partials into A if the component is adjustable."""
    if (point_id, 'E') in index.coord_index:
        A[row, index.coord_index[(point_id, 'E')]] += float(dE)
    if (point_id, 'N') in index.coord_index:
        A[row, index.coord_index[(point_id, 'N')]] += float(dN)


def _apply_corrections(state: _State, index, dx: 'np.ndarray') -> None:
    """Apply corrections vector dx to the state."""
    for (pid, comp), j in index.coord_index.items():
        e, n = state.points[pid]
        if comp == 'E':
            e += float(dx[j])
        else:
            n += float(dx[j])
        state.points[pid] = (e, n)

    for set_id, j in index.orientation_index.items():
        state.orientations[set_id] = state.orientations.get(set_id, 0.0) + float(dx[j])


def _compute_error_ellipse(point_id: str, cov2: 'np.ndarray', confidence: float) -> ErrorEllipse:
    """Compute error ellipse parameters from 2x2 covariance matrix."""
    # Eigen-decomposition
    vals, vecs = np.linalg.eigh(cov2)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # Scale factor for the requested confidence for 2D
    k2 = chi2_ppf(confidence, 2)
    scale = math.sqrt(max(k2, 0.0))

    semi_major = math.sqrt(max(vals[0], 0.0)) * scale
    semi_minor = math.sqrt(max(vals[1], 0.0)) * scale

    # Orientation of semi-major axis (vecs[:,0]) from North (Y) clockwise
    vE, vN = float(vecs[0, 0]), float(vecs[1, 0])
    orientation = wrap_2pi(math.atan2(vE, vN))

    return ErrorEllipse(
        point_id=point_id,
        semi_major=float(semi_major),
        semi_minor=float(semi_minor),
        orientation=float(orientation),
        confidence_level=float(confidence),
    )
