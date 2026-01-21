"""survey_adjustment.core.statistics.reliability

Reliability measures for least-squares adjustment.

Implemented (Phase 2):
- Redundancy numbers (internal reliability)
- Minimal Detectable Bias (MDB) using an (alpha, power) approximation
- External reliability: coordinate impact of an undetected bias of size MDB

Definitions used:
- P is diagonal with p_i = 1/sigma_i^2
- Qvv is residual cofactor matrix (m x m). We typically use only diag(Qvv).
- Redundancy number: r_i = qvv_ii * p_i (expected in [0,1])

MDB:
A common geodetic approximation (Baarda data snooping) uses a noncentrality
parameter lambda0 that depends on the significance level alpha and test power
(1-beta). A practical 1D approximation is:
    sqrt(lambda0) ~= k_alpha + k_beta
where k_alpha = N^{-1}(1-alpha/2) and k_beta = N^{-1}(power).
Then:
    MDB_i = (k_alpha + k_beta) * sigma0_hat * sigma_i / sqrt(r_i)

External reliability (coordinate impact):
For an observation i with bias b_i, the effect on unknowns is approximated by
    delta_x = Qxx A^T P e_i * b_i
We report a compact metric: max absolute shift among coordinate unknowns.
"""

from __future__ import annotations

import math
from typing import Iterable, List

import numpy as np


def redundancy_numbers(qvv_diag: np.ndarray, p_diag: np.ndarray) -> np.ndarray:
    """Compute redundancy numbers r_i = qvv_ii * p_i.

    Args:
        qvv_diag: diag(Qvv)
        p_diag: diag(P)

    Returns:
        r vector (length m)
    """
    r = qvv_diag * p_diag
    return np.clip(r, 0.0, 1.0)


def mdb_values(
    k_alpha: float,
    k_beta: float,
    sigma0_hat: float,
    sigma_obs: np.ndarray,
    redundancy: np.ndarray,
) -> np.ndarray:
    """Compute Minimal Detectable Bias (MDB) per observation.

    MDB_i = (k_alpha + k_beta) * sigma0_hat * sigma_i / sqrt(r_i)

    Args:
        k_alpha: critical value for significance (two-sided)
        k_beta: critical value for desired power
        sigma0_hat: a-posteriori sigma0
        sigma_obs: observation standard deviations (sigma_i)
        redundancy: redundancy numbers r_i

    Returns:
        MDB vector (length m); inf when redundancy is 0
    """
    k = float(k_alpha) + float(k_beta)
    r = np.maximum(redundancy, 0.0)
    out = np.full_like(sigma_obs, float('inf'), dtype=float)
    mask = r > 0.0
    out[mask] = k * float(sigma0_hat) * sigma_obs[mask] / np.sqrt(r[mask])
    return out


def external_reliability(
    Qxx: np.ndarray,
    A: np.ndarray,
    p_diag: np.ndarray,
    mdb: np.ndarray,
    coordinate_param_indices: Iterable[int],
) -> np.ndarray:
    """Compute an external reliability metric per observation.

    For each observation i, compute the unknown correction vector induced by a
    bias b_i = MDB_i:
        dx_i = Qxx * (p_i * A_i^T) * MDB_i

    We return a compact scalar metric per observation:
        max(|dx_j|) over coordinate unknown indices.

    Args:
        Qxx: cofactor matrix of unknowns (n x n)
        A: design matrix (m x n)
        p_diag: weights (length m)
        mdb: MDB vector (length m)
        coordinate_param_indices: indices of coordinate unknown components

    Returns:
        external reliability scalar per observation (length m)
    """
    coord_idx: List[int] = list(coordinate_param_indices)
    if not coord_idx:
        return np.zeros(A.shape[0], dtype=float)

    m = A.shape[0]
    out = np.zeros(m, dtype=float)

    # Precompute for speed: nothing fancy, m is usually moderate.
    for i in range(m):
        if not math.isfinite(float(mdb[i])) or float(mdb[i]) <= 0.0:
            out[i] = 0.0
            continue
        gi = Qxx @ (float(p_diag[i]) * A[i, :])
        dx = gi * float(mdb[i])
        out[i] = float(np.max(np.abs(dx[coord_idx])))

    return out
