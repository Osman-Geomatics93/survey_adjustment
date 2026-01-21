"""survey_adjustment.core.solver.least_squares_2d

2D least-squares adjustment (Gauss-Newton / linearized weighted LS).

Supported observation types:
  - DistanceObservation
  - DirectionObservation (with orientation unknown per set_id)
  - AngleObservation

This module intentionally contains **no QGIS imports** so it can be unit-
tested and re-used in non-QGIS contexts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore

from ..models.network import Network
from ..models.options import AdjustmentOptions
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
    ChiSquareTestResult,
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


# ---------------------------
# Statistical helpers
# ---------------------------

def _norm_ppf(p: float) -> float:
    """Approximate inverse CDF of the standard normal distribution.

    Uses Peter John Acklam's rational approximation.
    Accuracy is more than sufficient for confidence bounds used here.
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p must be in (0,1)")

    # Coefficients from Acklam's approximation
    a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ]
    b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ]

    plow = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )

    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )


def _chi2_ppf(p: float, dof: int) -> float:
    """Approximate inverse CDF for chi-square(dof) using Wilson–Hilferty."""
    if dof <= 0:
        raise ValueError("dof must be positive")
    z = _norm_ppf(p)
    k = float(dof)
    t = 1 - 2.0 / (9.0 * k) + z * math.sqrt(2.0 / (9.0 * k))
    return k * (t ** 3)


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

    index = build_parameter_index(network, use_direction_orientations=True)
    errors = validate_network_for_adjustment(network, index)
    if errors:
        return AdjustmentResult.failure("; ".join(errors))

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

    converged = False
    last_dx = None

    for it in range(1, options.max_iterations + 1):
        A, w, sigmas, computed = _linearize(enabled_obs, state, index)

        weights = 1.0 / (sigmas ** 2)
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

    # Final recomputation for residuals and statistics
    A_fin, w_fin, sigmas_fin, computed_fin = _linearize(enabled_obs, state, index)
    residuals = w_fin  # l - f(x_hat) (wrapped for angular obs)

    Pdiag = 1.0 / (sigmas_fin ** 2)
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

    # Standardized residuals & flagging
    std_residuals = {}
    residual_details: List[ResidualInfo] = []
    flagged: List[str] = []

    if cov_matrix is not None and dof > 0:
        # Qvv = P^{-1} - A Qxx A^T. Only need diagonal for standardization.
        Qxx = cov_matrix / sigma0_sq_post
        qvv_diag = []
        for i in range(m):
            ai = A_fin[i, :]
            q = (sigmas_fin[i] ** 2) - float(ai @ Qxx @ ai.T)
            qvv_diag.append(max(q, 1e-20))
        qvv_diag = np.array(qvv_diag)
        denom = np.sqrt(sigma0_sq_post * qvv_diag)
        std_vals = residuals / denom
    else:
        # Fallback: normalize by sigma only
        std_vals = residuals / (sigmas_fin * max(math.sqrt(sigma0_sq_post), 1e-12))

    for obs, comp_val, res, std in zip(enabled_obs, computed_fin, residuals, std_vals):
        obs_type = obs.obs_type.value
        info = ResidualInfo(
            obs_id=obs.id,
            obs_type=obs_type,
            observed=float(obs.value),
            computed=float(comp_val),
            residual=float(res),
            standardized_residual=float(std),
            flagged=abs(float(std)) > options.outlier_threshold,
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
        if info.flagged:
            flagged.append(obs.id)

    # Chi-square global test
    chi_test = None
    if dof > 0:
        alpha = options.alpha
        lower = _chi2_ppf(alpha / 2.0, dof)
        upper = _chi2_ppf(1 - alpha / 2.0, dof)
        test_stat = vTPv / options.a_priori_variance
        chi_test = ChiSquareTestResult(
            test_statistic=test_stat,
            critical_lower=lower,
            critical_upper=upper,
            confidence_level=options.confidence_level,
            passed=(lower <= test_stat <= upper),
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
    k2 = _chi2_ppf(confidence, 2)
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
