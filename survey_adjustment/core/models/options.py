"""
Adjustment options for survey network adjustment.

This module defines configuration options for the least-squares adjustment,
including iteration control, statistical parameters, and optional features.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum


class RobustEstimator(Enum):
    """
    Robust estimation methods for outlier handling.

    Supported methods:
    - NONE: Standard least squares (no robust estimation)
    - HUBER: Huber's M-estimator (soft downweighting)
    - DANISH: Danish method (aggressive downweighting)
    - IGG3: IGG-III method (three-part with hard rejection)
    """
    NONE = "none"
    HUBER = "huber"
    DANISH = "danish"
    IGG3 = "igg3"


@dataclass
class AdjustmentOptions:
    """
    Configuration options for least-squares adjustment.

    Attributes:
        max_iterations: Maximum number of iterations for convergence (default: 10)
        convergence_threshold: Convergence criterion in meters (default: 1e-8)
        confidence_level: Confidence level for statistical tests (default: 0.95)
        a_priori_variance: A priori variance of unit weight (default: 1.0)
        compute_covariances: Whether to compute full covariance matrix (default: True)
        robust_estimator: Robust estimation method, None for standard LS
        compute_error_ellipses: Whether to compute error ellipses (default: True)
        outlier_threshold: Standardized residual threshold for outlier detection (default: 3.0)
        angle_units_degrees: If True, expect angle input in degrees (default: True)
        sigma_units_arcseconds: If True, expect angular sigma in arc-seconds (default: False)

        Robust estimation parameters (Phase 7A):
        robust_max_iterations: Maximum IRLS iterations (default: 20)
        robust_tol: Convergence tolerance for IRLS weight change (default: 1e-3)
        huber_c: Huber tuning constant (default: 1.5)
        danish_c: Danish tuning constant (default: 2.0)
        igg3_k0: IGG-III lower threshold (default: 1.5)
        igg3_k1: IGG-III upper threshold (default: 3.0)
    """

    max_iterations: int = 10
    convergence_threshold: float = 1e-8  # meters
    confidence_level: float = 0.95
    a_priori_variance: float = 1.0
    compute_covariances: bool = True
    robust_estimator: Optional[RobustEstimator] = None
    compute_error_ellipses: bool = True
    # Phase 2 statistical settings
    alpha_local: float = 0.01  # local test significance (two-sided)
    mdb_power: float = 0.80    # desired test power (1-beta)
    compute_reliability: bool = True

    outlier_threshold: float = 3.0
    angle_units_degrees: bool = True
    sigma_units_arcseconds: bool = False

    # Robust estimation parameters (Phase 7A)
    robust_max_iterations: int = 20
    robust_tol: float = 1e-3
    huber_c: float = 1.5
    danish_c: float = 2.0
    igg3_k0: float = 1.5
    igg3_k1: float = 3.0

    # Auto-datum (Phase 7B) - applies minimal constraints automatically
    auto_datum: bool = False

    def __post_init__(self):
        """Validate options after initialization."""
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")

        if self.convergence_threshold <= 0:
            raise ValueError("convergence_threshold must be positive")

        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")

        if self.a_priori_variance <= 0:
            raise ValueError("a_priori_variance must be positive")

        if not 0 < self.alpha_local < 1:
            raise ValueError("alpha_local must be between 0 and 1")

        if not 0 < self.mdb_power < 1:
            raise ValueError("mdb_power must be between 0 and 1")

        if self.outlier_threshold <= 0:
            raise ValueError("outlier_threshold must be positive")

        # Convert string to enum if needed
        if isinstance(self.robust_estimator, str):
            if self.robust_estimator.lower() == "none" or self.robust_estimator == "":
                self.robust_estimator = None
            else:
                self.robust_estimator = RobustEstimator(self.robust_estimator.lower())

    @property
    def alpha(self) -> float:
        """
        Significance level (complement of confidence level).

        Returns:
            Alpha value for statistical tests
        """
        return 1.0 - self.confidence_level

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize options to dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            "max_iterations": self.max_iterations,
            "convergence_threshold": self.convergence_threshold,
            "confidence_level": self.confidence_level,
            "a_priori_variance": self.a_priori_variance,
            "compute_covariances": self.compute_covariances,
            "robust_estimator": self.robust_estimator.value if self.robust_estimator else None,
            "compute_error_ellipses": self.compute_error_ellipses,
            "alpha_local": self.alpha_local,
            "mdb_power": self.mdb_power,
            "compute_reliability": self.compute_reliability,
            "outlier_threshold": self.outlier_threshold,
            "angle_units_degrees": self.angle_units_degrees,
            "sigma_units_arcseconds": self.sigma_units_arcseconds,
            # Robust estimation parameters
            "robust_max_iterations": self.robust_max_iterations,
            "robust_tol": self.robust_tol,
            "huber_c": self.huber_c,
            "danish_c": self.danish_c,
            "igg3_k0": self.igg3_k0,
            "igg3_k1": self.igg3_k1,
            # Auto-datum (Phase 7B)
            "auto_datum": self.auto_datum,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AdjustmentOptions':
        """
        Create AdjustmentOptions from a dictionary.

        Args:
            data: Dictionary with option values

        Returns:
            New AdjustmentOptions instance
        """
        robust = data.get("robust_estimator")
        if robust and isinstance(robust, str) and robust.lower() != "none":
            robust = RobustEstimator(robust.lower())
        else:
            robust = None

        return cls(
            max_iterations=data.get("max_iterations", 10),
            convergence_threshold=data.get("convergence_threshold", 1e-8),
            confidence_level=data.get("confidence_level", 0.95),
            a_priori_variance=data.get("a_priori_variance", 1.0),
            compute_covariances=data.get("compute_covariances", True),
            robust_estimator=robust,
            compute_error_ellipses=data.get("compute_error_ellipses", True),
            alpha_local=data.get("alpha_local", 0.01),
            mdb_power=data.get("mdb_power", 0.80),
            compute_reliability=data.get("compute_reliability", True),
            outlier_threshold=data.get("outlier_threshold", 3.0),
            angle_units_degrees=data.get("angle_units_degrees", True),
            sigma_units_arcseconds=data.get("sigma_units_arcseconds", False),
            # Robust estimation parameters
            robust_max_iterations=data.get("robust_max_iterations", 20),
            robust_tol=data.get("robust_tol", 1e-3),
            huber_c=data.get("huber_c", 1.5),
            danish_c=data.get("danish_c", 2.0),
            igg3_k0=data.get("igg3_k0", 1.5),
            igg3_k1=data.get("igg3_k1", 3.0),
            # Auto-datum (Phase 7B)
            auto_datum=data.get("auto_datum", False),
        )

    @classmethod
    def default(cls) -> 'AdjustmentOptions':
        """
        Create options with default values.

        Returns:
            AdjustmentOptions with default settings
        """
        return cls()

    @classmethod
    def high_precision(cls) -> 'AdjustmentOptions':
        """
        Create options for high-precision adjustment.

        Returns:
            AdjustmentOptions configured for high precision work
        """
        return cls(
            max_iterations=20,
            convergence_threshold=1e-10,
            confidence_level=0.99,
            compute_covariances=True,
            compute_error_ellipses=True
        )

    def __repr__(self) -> str:
        return (
            f"AdjustmentOptions("
            f"max_iter={self.max_iterations}, "
            f"conv={self.convergence_threshold}, "
            f"conf={self.confidence_level})"
        )
