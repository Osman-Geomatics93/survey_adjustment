"""
Adjustment result classes for survey network adjustment.

This module defines the output data structures for the least-squares adjustment,
including adjusted coordinates, residuals, statistics, and error ellipses.
"""

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None  # type: ignore

from ..models.point import Point


def _iso_utc_now() -> str:
    """Return an ISO-8601 UTC timestamp ending with 'Z'."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_safe_value(value: Any) -> Any:
    """Convert non-JSON-safe floats (nan/inf) to None."""
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


@dataclass
class ErrorEllipse:
    """
    Error ellipse parameters for a point.

    The error ellipse represents the uncertainty region at a given
    confidence level. The ellipse is defined by its semi-axes and
    orientation.

    Attributes:
        point_id: ID of the point this ellipse belongs to
        semi_major: Semi-major axis length in meters
        semi_minor: Semi-minor axis length in meters
        orientation: Orientation of semi-major axis in radians (from north, clockwise)
        confidence_level: Confidence level (e.g., 0.95 for 95%)
    """

    point_id: str
    semi_major: float
    semi_minor: float
    orientation: float  # radians from north, clockwise
    confidence_level: float

    @property
    def orientation_degrees(self) -> float:
        """Return orientation in degrees."""
        return math.degrees(self.orientation)

    @property
    def eccentricity(self) -> float:
        """Calculate eccentricity of the ellipse."""
        if self.semi_major == 0:
            return 0.0
        return math.sqrt(1 - (self.semi_minor / self.semi_major) ** 2)

    @property
    def area(self) -> float:
        """Calculate area of the ellipse in square meters."""
        return math.pi * self.semi_major * self.semi_minor

    def to_dict(self) -> Dict[str, Any]:
        """Serialize error ellipse to dictionary."""
        return {
            "point_id": self.point_id,
            "semi_major_m": self.semi_major,
            "semi_minor_m": self.semi_minor,
            "orientation_rad": self.orientation,
            "orientation_deg": self.orientation_degrees,
            "confidence_level": self.confidence_level
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ErrorEllipse':
        """Create ErrorEllipse from dictionary."""
        # Handle both radian and degree input
        orientation = data.get("orientation_rad", data.get("orientation", 0))
        if "orientation_deg" in data and "orientation_rad" not in data:
            orientation = math.radians(data["orientation_deg"])

        return cls(
            point_id=data["point_id"],
            semi_major=data.get("semi_major_m", data.get("semi_major", 0)),
            semi_minor=data.get("semi_minor_m", data.get("semi_minor", 0)),
            orientation=orientation,
            confidence_level=data.get("confidence_level", 0.95)
        )


@dataclass
class ChiSquareTestResult:
    """
    Result of chi-square global test for variance factor.

    The chi-square test checks if the a posteriori variance factor
    is statistically consistent with the a priori variance.

    Attributes:
        test_statistic: Computed test statistic (dof * variance_factor)
        critical_lower: Lower critical value at given confidence
        critical_upper: Upper critical value at given confidence
        confidence_level: Confidence level of the test
        passed: True if test_statistic is within critical bounds
    """

    test_statistic: float
    critical_lower: float
    critical_upper: float
    confidence_level: float
    passed: bool
    p_value: float | None = None
    degrees_of_freedom: int | None = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize chi-square test result to dictionary."""
        return {
            "test_name": "chi_square",
            "test_statistic": _json_safe_value(self.test_statistic),
            "critical_lower": _json_safe_value(self.critical_lower),
            "critical_upper": _json_safe_value(self.critical_upper),
            "confidence_level": self.confidence_level,
            "passed": self.passed,
            "p_value": _json_safe_value(self.p_value),
            "degrees_of_freedom": self.degrees_of_freedom
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChiSquareTestResult':
        """Create ChiSquareTestResult from dictionary."""
        return cls(
            test_statistic=data["test_statistic"],
            critical_lower=data["critical_lower"],
            critical_upper=data["critical_upper"],
            confidence_level=data["confidence_level"],
            passed=data["passed"],
            p_value=data.get("p_value"),
            degrees_of_freedom=data.get("degrees_of_freedom")
        )


@dataclass
class ResidualInfo:
    """
    Detailed information about a single residual.

    Attributes:
        obs_id: ID of the observation
        obs_type: Type of observation
        observed: Observed value
        computed: Computed value from adjusted coordinates
        residual: Residual (observed - computed)
        standardized_residual: Residual divided by its standard deviation
        flagged: True if observation is flagged as potential outlier
        weight_factor: Robust weight factor (1.0 = full weight, <1.0 = downweighted)
    """

    obs_id: str
    obs_type: str
    observed: float
    computed: float
    residual: float
    standardized_residual: float
    redundancy_number: float | None = None
    mdb: float | None = None
    external_reliability: float | None = None
    is_outlier_candidate: bool = False
    flagged: bool = False
    from_point: Optional[str] = None
    to_point: Optional[str] = None
    at_point: Optional[str] = None
    # Robust estimation (Phase 7A)
    weight_factor: float | None = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize residual info to dictionary."""
        data = {
            "obs_id": self.obs_id,
            "obs_type": self.obs_type,
            "observed": _json_safe_value(self.observed),
            "computed": _json_safe_value(self.computed),
            "residual": _json_safe_value(self.residual),
            "standardized_residual": _json_safe_value(self.standardized_residual),
            "redundancy_number": _json_safe_value(self.redundancy_number),
            "mdb": _json_safe_value(self.mdb),
            "external_reliability": _json_safe_value(self.external_reliability),
            "is_outlier_candidate": self.is_outlier_candidate,
            "flagged": self.flagged,
            "weight_factor": _json_safe_value(self.weight_factor),
        }
        if self.from_point:
            data["from_point"] = self.from_point
        if self.to_point:
            data["to_point"] = self.to_point
        if self.at_point:
            data["at_point"] = self.at_point
        return data


@dataclass
class AdjustmentResult:
    """
    Complete results from a least-squares adjustment.

    This class contains all output from the adjustment computation,
    including adjusted coordinates, residuals, statistics, and
    quality measures.

    Attributes:
        success: True if adjustment completed without errors
        iterations: Number of iterations performed
        converged: True if solution converged within threshold

        adjusted_points: Dictionary of adjusted Point objects
        residuals: Dictionary mapping obs_id to residual value
        standardized_residuals: Dictionary mapping obs_id to standardized residual
        residual_details: List of detailed residual information

        degrees_of_freedom: Number of redundant observations
        variance_factor: A posteriori variance factor (sigma0^2)
        chi_square_test: Result of global chi-square test

        covariance_matrix: Full covariance matrix (numpy array if available)
        point_covariances: Dictionary mapping point_id to 2x2 covariance matrix

        error_ellipses: Dictionary mapping point_id to ErrorEllipse

        flagged_observations: List of observation IDs flagged as outliers

        messages: List of warnings or informational messages
        error_message: Error description if success is False
    """

    success: bool = True
    iterations: int = 0
    converged: bool = False

    # Adjusted coordinates
    adjusted_points: Dict[str, Point] = field(default_factory=dict)

    # Residuals
    residuals: Dict[str, float] = field(default_factory=dict)
    standardized_residuals: Dict[str, float] = field(default_factory=dict)
    residual_details: List[ResidualInfo] = field(default_factory=list)

    # Statistics
    degrees_of_freedom: int = 0
    variance_factor: float = 1.0
    chi_square_test: Optional[ChiSquareTestResult] = None

    # Covariances (numpy arrays stored as lists for serialization)
    covariance_matrix: Optional[Any] = None  # np.ndarray or list
    point_covariances: Dict[str, Any] = field(default_factory=dict)  # point_id -> 2x2 matrix

    # Error ellipses
    error_ellipses: Dict[str, ErrorEllipse] = field(default_factory=dict)

    # Outlier detection
    flagged_observations: List[str] = field(default_factory=list)

    # Messages
    messages: List[str] = field(default_factory=list)
    error_message: Optional[str] = None

    # Metadata
    timestamp: Optional[str] = None
    plugin_version: str = "1.0.0"
    network_name: str = ""

    # Robust estimation (Phase 7A)
    robust_method: Optional[str] = None  # "none", "huber", "danish", "igg3"
    robust_iterations: int = 0
    robust_converged: bool = True
    robust_message: Optional[str] = None

    # Constraint health / datum summary (Phase 7B)
    datum_summary: Optional[Dict[str, Any]] = None  # ConstraintHealth.to_dict()
    applied_auto_constraints: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = _iso_utc_now()

    @property
    def a_posteriori_sigma0(self) -> float:
        """
        A posteriori standard deviation of unit weight.

        Returns:
            Square root of variance factor
        """
        return math.sqrt(self.variance_factor)

    @property
    def redundancy(self) -> int:
        """Alias for degrees_of_freedom."""
        return self.degrees_of_freedom

    def get_point_sigma(self, point_id: str) -> tuple:
        """
        Get standard deviations for a point.

        Args:
            point_id: ID of the point

        Returns:
            Tuple of (sigma_easting, sigma_northing)

        Raises:
            KeyError: If point not found
        """
        if point_id in self.adjusted_points:
            point = self.adjusted_points[point_id]
            return (point.sigma_easting, point.sigma_northing)
        raise KeyError(f"Point '{point_id}' not in results")

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize adjustment result to dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        # Convert numpy arrays to lists if needed
        cov_matrix = None
        if self.covariance_matrix is not None:
            if HAS_NUMPY and isinstance(self.covariance_matrix, np.ndarray):
                cov_matrix = self.covariance_matrix.tolist()
            else:
                cov_matrix = self.covariance_matrix

        point_covs = {}
        for pid, cov in self.point_covariances.items():
            if HAS_NUMPY and isinstance(cov, np.ndarray):
                point_covs[pid] = cov.tolist()
            else:
                point_covs[pid] = cov

        # Build observations summary
        obs_by_type: Dict[str, int] = {}
        for detail in self.residual_details:
            obs_type = detail.obs_type
            obs_by_type[obs_type] = obs_by_type.get(obs_type, 0) + 1

        return {
            "metadata": {
                "plugin_version": self.plugin_version,
                "timestamp": self.timestamp,
                "network_name": self.network_name
            },
            "input_summary": {
                "num_points": len(self.adjusted_points),
                "num_fixed_points": len([p for p in self.adjusted_points.values() if p.is_fixed]),
                "num_observations": len(self.residuals),
                "observations_by_type": obs_by_type
            },
            "adjustment": {
                "success": self.success,
                "iterations": self.iterations,
                "converged": self.converged,
                "degrees_of_freedom": self.degrees_of_freedom,
                "variance_factor": _json_safe_value(self.variance_factor),
                "a_posteriori_sigma0": self.a_posteriori_sigma0,
                "error_message": self.error_message,
                "messages": self.messages,
                # Robust estimation
                "robust_method": self.robust_method,
                "robust_iterations": self.robust_iterations,
                "robust_converged": self.robust_converged,
                "robust_message": self.robust_message,
            },
            "global_test": self.chi_square_test.to_dict() if self.chi_square_test else None,
            "datum_summary": self.datum_summary,
            "applied_auto_constraints": self.applied_auto_constraints if self.applied_auto_constraints else None,
            "adjusted_points": [
                {
                    "id": p.id,
                    "name": p.name,
                    "easting": p.easting,
                    "northing": p.northing,
                    "height": p.height,
                    "fixed_easting": p.fixed_easting,
                    "fixed_northing": p.fixed_northing,
                    "fixed_height": p.fixed_height,
                    "sigma_easting": p.sigma_easting,
                    "sigma_northing": p.sigma_northing,
                    "sigma_height": p.sigma_height,
                }
                for p in self.adjusted_points.values()
            ],
            "error_ellipses": [
                ellipse.to_dict() for ellipse in self.error_ellipses.values()
            ],
            "residuals": [
                detail.to_dict() for detail in self.residual_details
            ],
            "flagged_observations": self.flagged_observations,
            "covariance_matrix": cov_matrix,
            "point_covariances": point_covs
        }

    def to_json(self, indent: int = 2) -> str:
        """
        Serialize adjustment result to JSON string.

        Args:
            indent: Number of spaces for indentation

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AdjustmentResult':
        """
        Create AdjustmentResult from dictionary.

        Args:
            data: Dictionary with result data

        Returns:
            New AdjustmentResult instance
        """
        # Extract nested data
        metadata = data.get("metadata", {})
        adjustment = data.get("adjustment", {})
        global_test = data.get("global_test")

        # Build adjusted points
        adjusted_points = {}
        for point_data in data.get("adjusted_points", []):
            point = Point.from_dict(point_data)
            adjusted_points[point.id] = point

        # Build residuals dict
        residuals = {}
        std_residuals = {}
        residual_details = []
        for res_data in data.get("residuals", []):
            obs_id = res_data["obs_id"]
            residuals[obs_id] = res_data["residual"]
            std_residuals[obs_id] = res_data["standardized_residual"]
            residual_details.append(ResidualInfo(
                obs_id=obs_id,
                obs_type=res_data["obs_type"],
                observed=res_data["observed"],
                computed=res_data["computed"],
                residual=res_data["residual"],
                standardized_residual=res_data["standardized_residual"],
                redundancy_number=res_data.get("redundancy_number"),
                mdb=res_data.get("mdb"),
                external_reliability=res_data.get("external_reliability"),
                is_outlier_candidate=res_data.get("is_outlier_candidate", False),
                flagged=res_data.get("flagged", False),
                from_point=res_data.get("from_point"),
                to_point=res_data.get("to_point"),
                at_point=res_data.get("at_point"),
                weight_factor=res_data.get("weight_factor"),
            ))

        # Build error ellipses
        error_ellipses = {}
        for ellipse_data in data.get("error_ellipses", []):
            ellipse = ErrorEllipse.from_dict(ellipse_data)
            error_ellipses[ellipse.point_id] = ellipse

        # Build chi-square test
        chi_test = None
        if global_test:
            chi_test = ChiSquareTestResult.from_dict(global_test)

        return cls(
            success=adjustment.get("success", True),
            iterations=adjustment.get("iterations", 0),
            converged=adjustment.get("converged", False),
            adjusted_points=adjusted_points,
            residuals=residuals,
            standardized_residuals=std_residuals,
            residual_details=residual_details,
            degrees_of_freedom=adjustment.get("degrees_of_freedom", 0),
            variance_factor=adjustment.get("variance_factor", 1.0),
            chi_square_test=chi_test,
            covariance_matrix=data.get("covariance_matrix"),
            point_covariances=data.get("point_covariances", {}),
            error_ellipses=error_ellipses,
            flagged_observations=data.get("flagged_observations", []),
            messages=adjustment.get("messages", []),
            error_message=adjustment.get("error_message"),
            timestamp=metadata.get("timestamp"),
            plugin_version=metadata.get("plugin_version", "1.0.0"),
            network_name=metadata.get("network_name", ""),
            # Robust estimation
            robust_method=adjustment.get("robust_method"),
            robust_iterations=adjustment.get("robust_iterations", 0),
            robust_converged=adjustment.get("robust_converged", True),
            robust_message=adjustment.get("robust_message"),
            # Constraint health (Phase 7B)
            datum_summary=data.get("datum_summary"),
            applied_auto_constraints=data.get("applied_auto_constraints", []),
        )

    @classmethod
    def failure(cls, error_message: str) -> 'AdjustmentResult':
        """
        Create a failed adjustment result.

        Args:
            error_message: Description of the failure

        Returns:
            AdjustmentResult with success=False
        """
        return cls(
            success=False,
            converged=False,
            error_message=error_message
        )

    def __repr__(self) -> str:
        status = "success" if self.success else "failed"
        conv = "converged" if self.converged else "not converged"
        return (
            f"AdjustmentResult({status}, {conv}, "
            f"iter={self.iterations}, dof={self.degrees_of_freedom})"
        )
