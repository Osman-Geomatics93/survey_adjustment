"""
Observation classes for survey network adjustment.

Conventions:
- Angles: Radians internally, converted from degrees at I/O boundary
- Azimuth: North = 0, clockwise positive (standard surveying convention)
- Distance: Meters (ground/grid as specified)
- Standard deviation: Meters for distances, radians for angles
- Observation IDs: Auto-generated UUID or user-provided string

Observation Types:
- Distance: Measured slope/horizontal distance between two points
- Direction: Measured horizontal direction from one point to another (requires orientation unknown)
- Angle: Measured horizontal angle at a point between two other points
- HeightDiff: Leveling observation (Phase 5)
- GNSSBaseline: GNSS baseline vector (Phase 5)
"""

import math
import uuid
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List


class ObservationType(Enum):
    """Enumeration of supported observation types."""
    DISTANCE = "distance"
    DIRECTION = "direction"
    ANGLE = "angle"
    HEIGHT_DIFF = "height_diff"      # Phase 5
    GNSS_BASELINE = "gnss_baseline"  # Phase 5


def _generate_obs_id() -> str:
    """Generate a unique observation ID."""
    return str(uuid.uuid4())[:8].upper()


@dataclass
class Observation(ABC):
    """
    Base class for all observation types.

    Attributes:
        id: Unique identifier for the observation
        obs_type: Type of observation (distance, direction, angle, etc.)
        value: Observed value (meters for distances, radians for angles)
        sigma: Standard deviation (meters for distances, radians for angles)
        enabled: If False, observation is excluded from adjustment (for outlier handling)
    """

    id: str
    obs_type: ObservationType
    value: float
    sigma: float
    enabled: bool = True

    def __post_init__(self):
        """Validate observation data after initialization."""
        if not self.id:
            object.__setattr__(self, 'id', _generate_obs_id())

        if self.sigma <= 0:
            raise ValueError(f"Standard deviation must be positive, got {self.sigma}")

    @property
    def weight(self) -> float:
        """
        Calculate weight from standard deviation.

        Weight = 1 / sigma^2 (inverse variance weighting)
        """
        return 1.0 / (self.sigma ** 2)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize observation to dictionary (base fields only)."""
        return {
            "id": self.id,
            "obs_type": self.obs_type.value,
            "value": self.value,
            "sigma": self.sigma,
            "enabled": self.enabled
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Observation':
        """
        Factory method to create appropriate Observation subclass from dictionary.

        Args:
            data: Dictionary with observation attributes

        Returns:
            Appropriate Observation subclass instance
        """
        obs_type = data.get("obs_type", data.get("type", ""))

        if isinstance(obs_type, ObservationType):
            obs_type = obs_type.value

        if obs_type == ObservationType.DISTANCE.value or obs_type == "distance":
            return DistanceObservation.from_dict(data)
        elif obs_type == ObservationType.DIRECTION.value or obs_type == "direction":
            return DirectionObservation.from_dict(data)
        elif obs_type == ObservationType.ANGLE.value or obs_type == "angle":
            return AngleObservation.from_dict(data)
        elif obs_type == ObservationType.HEIGHT_DIFF.value or obs_type == "height_diff":
            return HeightDifferenceObservation.from_dict(data)
        elif obs_type == ObservationType.GNSS_BASELINE.value or obs_type == "gnss_baseline":
            return GnssBaselineObservation.from_dict(data)
        else:
            raise ValueError(f"Unknown observation type: {obs_type}")


@dataclass
class DistanceObservation(Observation):
    """
    Distance observation between two points.

    Attributes:
        from_point_id: ID of the instrument station (from point)
        to_point_id: ID of the target station (to point)
    """

    from_point_id: str = ""
    to_point_id: str = ""

    def __post_init__(self):
        """Set observation type and validate."""
        object.__setattr__(self, 'obs_type', ObservationType.DISTANCE)
        super().__post_init__()

        if not self.from_point_id:
            raise ValueError("from_point_id cannot be empty")
        if not self.to_point_id:
            raise ValueError("to_point_id cannot be empty")
        if self.from_point_id == self.to_point_id:
            raise ValueError("from_point_id and to_point_id cannot be the same")
        if self.value <= 0:
            raise ValueError(f"Distance must be positive, got {self.value}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize distance observation to dictionary."""
        data = super().to_dict()
        data.update({
            "from_point_id": self.from_point_id,
            "to_point_id": self.to_point_id
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DistanceObservation':
        """Create DistanceObservation from dictionary."""
        return cls(
            id=data.get("id", data.get("obs_id", "")),
            obs_type=ObservationType.DISTANCE,
            value=float(data.get("value", data.get("distance", 0))),
            sigma=float(data.get("sigma", data.get("sigma_distance", 0.001))),
            enabled=data.get("enabled", True),
            from_point_id=str(data.get("from_point_id", data.get("from_point", ""))),
            to_point_id=str(data.get("to_point_id", data.get("to_point", "")))
        )

    def __repr__(self) -> str:
        return f"DistanceObs({self.id}: {self.from_point_id}->{self.to_point_id}, {self.value:.3f}m)"


@dataclass
class DirectionObservation(Observation):
    """
    Direction observation from one point to another.

    Directions require an orientation unknown for each set of observations
    from the same instrument station. The set_id groups directions that share
    the same orientation unknown.

    Attributes:
        from_point_id: ID of the instrument station
        to_point_id: ID of the target station
        set_id: Identifier for grouping directions with same orientation unknown
    """

    from_point_id: str = ""
    to_point_id: str = ""
    set_id: str = ""

    def __post_init__(self):
        """Set observation type and validate."""
        object.__setattr__(self, 'obs_type', ObservationType.DIRECTION)
        super().__post_init__()

        if not self.from_point_id:
            raise ValueError("from_point_id cannot be empty")
        if not self.to_point_id:
            raise ValueError("to_point_id cannot be empty")
        if self.from_point_id == self.to_point_id:
            raise ValueError("from_point_id and to_point_id cannot be the same")
        if not self.set_id:
            # Default set_id based on from_point
            object.__setattr__(self, 'set_id', f"SET_{self.from_point_id}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize direction observation to dictionary."""
        data = super().to_dict()
        data.update({
            "from_point_id": self.from_point_id,
            "to_point_id": self.to_point_id,
            "set_id": self.set_id
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DirectionObservation':
        """
        Create DirectionObservation from dictionary.

        Note: If 'direction' is in degrees, it must be converted to radians
        before calling this method, or use from_dict_degrees().
        """
        return cls(
            id=data.get("id", data.get("obs_id", "")),
            obs_type=ObservationType.DIRECTION,
            value=float(data.get("value", data.get("direction", 0))),
            sigma=float(data.get("sigma", data.get("sigma_direction", 0.00015))),
            enabled=data.get("enabled", True),
            from_point_id=str(data.get("from_point_id", data.get("from_point", ""))),
            to_point_id=str(data.get("to_point_id", data.get("to_point", ""))),
            set_id=str(data.get("set_id", ""))
        )

    @classmethod
    def from_dict_degrees(cls, data: Dict[str, Any], sigma_in_arcseconds: bool = False) -> 'DirectionObservation':
        """
        Create DirectionObservation from dictionary with degrees input.

        Args:
            data: Dictionary with direction in degrees
            sigma_in_arcseconds: If True, sigma is in arc-seconds; convert to radians

        Returns:
            DirectionObservation with values in radians
        """
        direction_deg = float(data.get("direction", 0))
        direction_rad = math.radians(direction_deg)

        sigma = float(data.get("sigma", data.get("sigma_direction", 0.00015)))
        if sigma_in_arcseconds:
            # Convert arc-seconds to radians: arcsec * (pi/180) / 3600
            sigma = sigma * math.pi / (180.0 * 3600.0)

        return cls(
            id=data.get("id", data.get("obs_id", "")),
            obs_type=ObservationType.DIRECTION,
            value=direction_rad,
            sigma=sigma,
            enabled=data.get("enabled", True),
            from_point_id=str(data.get("from_point_id", data.get("from_point", ""))),
            to_point_id=str(data.get("to_point_id", data.get("to_point", ""))),
            set_id=str(data.get("set_id", ""))
        )

    @property
    def value_degrees(self) -> float:
        """Return direction value in degrees."""
        return math.degrees(self.value)

    def __repr__(self) -> str:
        return f"DirectionObs({self.id}: {self.from_point_id}->{self.to_point_id}, {self.value_degrees:.4f}deg)"


@dataclass
class AngleObservation(Observation):
    """
    Angle observation at a point between two other points.

    The angle is measured at at_point, from the direction to from_point
    to the direction to to_point, clockwise positive.

    Example: Angle ABC measured at B, from A to C, clockwise.

    Attributes:
        at_point_id: ID of the instrument station where angle is measured
        from_point_id: ID of the backsight point
        to_point_id: ID of the foresight point
    """

    at_point_id: str = ""
    from_point_id: str = ""
    to_point_id: str = ""

    def __post_init__(self):
        """Set observation type and validate."""
        object.__setattr__(self, 'obs_type', ObservationType.ANGLE)
        super().__post_init__()

        if not self.at_point_id:
            raise ValueError("at_point_id cannot be empty")
        if not self.from_point_id:
            raise ValueError("from_point_id cannot be empty")
        if not self.to_point_id:
            raise ValueError("to_point_id cannot be empty")

        # All three points must be different
        points = {self.at_point_id, self.from_point_id, self.to_point_id}
        if len(points) != 3:
            raise ValueError("at_point_id, from_point_id, and to_point_id must all be different")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize angle observation to dictionary."""
        data = super().to_dict()
        data.update({
            "at_point_id": self.at_point_id,
            "from_point_id": self.from_point_id,
            "to_point_id": self.to_point_id
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AngleObservation':
        """Create AngleObservation from dictionary (value in radians)."""
        return cls(
            id=data.get("id", data.get("obs_id", "")),
            obs_type=ObservationType.ANGLE,
            value=float(data.get("value", data.get("angle", 0))),
            sigma=float(data.get("sigma", data.get("sigma_angle", 0.00015))),
            enabled=data.get("enabled", True),
            at_point_id=str(data.get("at_point_id", data.get("at_point", ""))),
            from_point_id=str(data.get("from_point_id", data.get("from_point", ""))),
            to_point_id=str(data.get("to_point_id", data.get("to_point", "")))
        )

    @classmethod
    def from_dict_degrees(cls, data: Dict[str, Any], sigma_in_arcseconds: bool = False) -> 'AngleObservation':
        """
        Create AngleObservation from dictionary with degrees input.

        Args:
            data: Dictionary with angle in degrees
            sigma_in_arcseconds: If True, sigma is in arc-seconds; convert to radians

        Returns:
            AngleObservation with values in radians
        """
        angle_deg = float(data.get("angle", 0))
        angle_rad = math.radians(angle_deg)

        sigma = float(data.get("sigma", data.get("sigma_angle", 0.00015)))
        if sigma_in_arcseconds:
            sigma = sigma * math.pi / (180.0 * 3600.0)

        return cls(
            id=data.get("id", data.get("obs_id", "")),
            obs_type=ObservationType.ANGLE,
            value=angle_rad,
            sigma=sigma,
            enabled=data.get("enabled", True),
            at_point_id=str(data.get("at_point_id", data.get("at_point", ""))),
            from_point_id=str(data.get("from_point_id", data.get("from_point", ""))),
            to_point_id=str(data.get("to_point_id", data.get("to_point", "")))
        )

    @property
    def value_degrees(self) -> float:
        """Return angle value in degrees."""
        return math.degrees(self.value)

    def __repr__(self) -> str:
        return f"AngleObs({self.id}: {self.from_point_id}-{self.at_point_id}-{self.to_point_id}, {self.value_degrees:.4f}deg)"


@dataclass
class HeightDifferenceObservation(Observation):
    """
    Height difference (leveling) observation between two points.

    The observed value (dh) is the height difference from the 'from' point
    to the 'to' point: dh = H_to - H_from (positive if going uphill).

    Residual convention: v = dh_observed - (H_to - H_from)

    Attributes:
        from_point_id: ID of the 'from' point (backsight in leveling)
        to_point_id: ID of the 'to' point (foresight in leveling)
    """

    from_point_id: str = ""
    to_point_id: str = ""

    def __post_init__(self):
        """Set observation type and validate."""
        object.__setattr__(self, 'obs_type', ObservationType.HEIGHT_DIFF)
        super().__post_init__()

        if not self.from_point_id:
            raise ValueError("from_point_id cannot be empty")
        if not self.to_point_id:
            raise ValueError("to_point_id cannot be empty")
        if self.from_point_id == self.to_point_id:
            raise ValueError("from_point_id and to_point_id cannot be the same")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize height difference observation to dictionary."""
        data = super().to_dict()
        data.update({
            "from_point_id": self.from_point_id,
            "to_point_id": self.to_point_id,
            "dh": self.value,  # Alias for clarity
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HeightDifferenceObservation':
        """Create HeightDifferenceObservation from dictionary."""
        # Accept 'dh' or 'value' for the height difference
        dh = float(data.get("dh", data.get("value", data.get("height_diff", 0))))
        sigma = float(data.get("sigma", data.get("sigma_dh", 0.001)))

        return cls(
            id=data.get("id", data.get("obs_id", "")),
            obs_type=ObservationType.HEIGHT_DIFF,
            value=dh,
            sigma=sigma,
            enabled=data.get("enabled", True),
            from_point_id=str(data.get("from_point_id", data.get("from_id", data.get("from_point", "")))),
            to_point_id=str(data.get("to_point_id", data.get("to_id", data.get("to_point", "")))),
        )

    @property
    def dh(self) -> float:
        """Alias for value: the observed height difference in meters."""
        return self.value

    def __repr__(self) -> str:
        sign = "+" if self.value >= 0 else ""
        return f"HeightDiffObs({self.id}: {self.from_point_id}->{self.to_point_id}, {sign}{self.value:.4f}m)"


@dataclass
class GnssBaselineObservation(Observation):
    """
    GNSS baseline observation between two points (3D vector).

    A baseline represents the vector difference (dE, dN, dH) from the 'from' point
    to the 'to' point in a local coordinate system (Easting, Northing, Height).

    The observation includes a 3x3 covariance matrix for the baseline components.
    Covariance can be specified as:
    - Full covariance terms: cov_EE, cov_EN, cov_EH, cov_NN, cov_NH, cov_HH
    - Or sigmas + correlations: sigma_E, sigma_N, sigma_H, rho_EN, rho_EH, rho_NH

    Residual convention: v = observed - computed
    where computed = (E_to - E_from, N_to - N_from, H_to - H_from)

    Attributes:
        from_point_id: ID of the reference point (base station)
        to_point_id: ID of the target point (rover)
        dE: Observed easting difference (meters)
        dN: Observed northing difference (meters)
        dH: Observed height difference (meters)
        cov_EE, cov_EN, cov_EH, cov_NN, cov_NH, cov_HH: Covariance matrix terms (m²)
    """

    from_point_id: str = ""
    to_point_id: str = ""

    # Baseline components
    dE: float = 0.0
    dN: float = 0.0
    dH: float = 0.0

    # Covariance matrix (symmetric 3x3, stored as upper triangle)
    # [cov_EE  cov_EN  cov_EH]
    # [cov_EN  cov_NN  cov_NH]
    # [cov_EH  cov_NH  cov_HH]
    cov_EE: float = 0.0001  # Default 1cm sigma
    cov_EN: float = 0.0
    cov_EH: float = 0.0
    cov_NN: float = 0.0001
    cov_NH: float = 0.0
    cov_HH: float = 0.0001

    def __post_init__(self):
        """Set observation type and validate."""
        object.__setattr__(self, 'obs_type', ObservationType.GNSS_BASELINE)
        # Set placeholder value/sigma for base class compatibility
        # The actual values are in dE, dN, dH and covariance matrix
        object.__setattr__(self, 'value', 0.0)
        object.__setattr__(self, 'sigma', 1.0)

        # Call parent validation (but skip sigma check since we use covariance)
        if not self.id:
            object.__setattr__(self, 'id', _generate_obs_id())

        if not self.from_point_id:
            raise ValueError("from_point_id cannot be empty")
        if not self.to_point_id:
            raise ValueError("to_point_id cannot be empty")
        if self.from_point_id == self.to_point_id:
            raise ValueError("from_point_id and to_point_id cannot be the same")

        # Validate covariance (diagonal must be positive)
        if self.cov_EE <= 0 or self.cov_NN <= 0 or self.cov_HH <= 0:
            raise ValueError("Covariance diagonal elements must be positive")

    @property
    def covariance_matrix(self) -> List[List[float]]:
        """Return the 3x3 covariance matrix as a nested list."""
        return [
            [self.cov_EE, self.cov_EN, self.cov_EH],
            [self.cov_EN, self.cov_NN, self.cov_NH],
            [self.cov_EH, self.cov_NH, self.cov_HH],
        ]

    @property
    def sigma_E(self) -> float:
        """Standard deviation of dE component."""
        return math.sqrt(self.cov_EE)

    @property
    def sigma_N(self) -> float:
        """Standard deviation of dN component."""
        return math.sqrt(self.cov_NN)

    @property
    def sigma_H(self) -> float:
        """Standard deviation of dH component."""
        return math.sqrt(self.cov_HH)

    @property
    def baseline_length(self) -> float:
        """3D length of the baseline vector."""
        return math.sqrt(self.dE**2 + self.dN**2 + self.dH**2)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize GNSS baseline observation to dictionary."""
        return {
            "id": self.id,
            "obs_type": self.obs_type.value,
            "enabled": self.enabled,
            "from_point_id": self.from_point_id,
            "to_point_id": self.to_point_id,
            "dE": self.dE,
            "dN": self.dN,
            "dH": self.dH,
            "cov_EE": self.cov_EE,
            "cov_EN": self.cov_EN,
            "cov_EH": self.cov_EH,
            "cov_NN": self.cov_NN,
            "cov_NH": self.cov_NH,
            "cov_HH": self.cov_HH,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GnssBaselineObservation':
        """Create GnssBaselineObservation from dictionary."""
        # Handle both covariance formats
        if "cov_EE" in data:
            # Full covariance format
            cov_EE = float(data.get("cov_EE", 0.0001))
            cov_EN = float(data.get("cov_EN", 0.0))
            cov_EH = float(data.get("cov_EH", 0.0))
            cov_NN = float(data.get("cov_NN", 0.0001))
            cov_NH = float(data.get("cov_NH", 0.0))
            cov_HH = float(data.get("cov_HH", 0.0001))
        elif "sigma_E" in data:
            # Sigmas + correlations format
            sigma_E = float(data.get("sigma_E", 0.01))
            sigma_N = float(data.get("sigma_N", 0.01))
            sigma_H = float(data.get("sigma_H", 0.01))
            rho_EN = float(data.get("rho_EN", 0.0))
            rho_EH = float(data.get("rho_EH", 0.0))
            rho_NH = float(data.get("rho_NH", 0.0))

            # Convert to covariance: cov_ij = rho_ij * sigma_i * sigma_j
            cov_EE = sigma_E ** 2
            cov_NN = sigma_N ** 2
            cov_HH = sigma_H ** 2
            cov_EN = rho_EN * sigma_E * sigma_N
            cov_EH = rho_EH * sigma_E * sigma_H
            cov_NH = rho_NH * sigma_N * sigma_H
        else:
            # Default covariance (10mm sigma, uncorrelated)
            default_var = 0.0001  # (0.01m)^2
            cov_EE = cov_NN = cov_HH = default_var
            cov_EN = cov_EH = cov_NH = 0.0

        return cls(
            id=data.get("id", data.get("obs_id", "")),
            obs_type=ObservationType.GNSS_BASELINE,
            value=0.0,  # Placeholder
            sigma=1.0,  # Placeholder
            enabled=data.get("enabled", True),
            from_point_id=str(data.get("from_point_id", data.get("from_id", data.get("from", "")))),
            to_point_id=str(data.get("to_point_id", data.get("to_id", data.get("to", "")))),
            dE=float(data.get("dE", data.get("delta_E", 0.0))),
            dN=float(data.get("dN", data.get("delta_N", 0.0))),
            dH=float(data.get("dH", data.get("delta_H", 0.0))),
            cov_EE=cov_EE,
            cov_EN=cov_EN,
            cov_EH=cov_EH,
            cov_NN=cov_NN,
            cov_NH=cov_NH,
            cov_HH=cov_HH,
        )

    @classmethod
    def from_sigmas_correlations(
        cls,
        from_point_id: str,
        to_point_id: str,
        dE: float,
        dN: float,
        dH: float,
        sigma_E: float,
        sigma_N: float,
        sigma_H: float,
        rho_EN: float = 0.0,
        rho_EH: float = 0.0,
        rho_NH: float = 0.0,
        obs_id: str = "",
        enabled: bool = True,
    ) -> 'GnssBaselineObservation':
        """
        Create observation from sigmas and correlations.

        Args:
            from_point_id: Reference point ID
            to_point_id: Target point ID
            dE, dN, dH: Baseline components (meters)
            sigma_E, sigma_N, sigma_H: Standard deviations (meters)
            rho_EN, rho_EH, rho_NH: Correlation coefficients (-1 to 1)
            obs_id: Optional observation ID
            enabled: Whether observation is enabled
        """
        return cls(
            id=obs_id,
            obs_type=ObservationType.GNSS_BASELINE,
            value=0.0,
            sigma=1.0,
            enabled=enabled,
            from_point_id=from_point_id,
            to_point_id=to_point_id,
            dE=dE,
            dN=dN,
            dH=dH,
            cov_EE=sigma_E ** 2,
            cov_EN=rho_EN * sigma_E * sigma_N,
            cov_EH=rho_EH * sigma_E * sigma_H,
            cov_NN=sigma_N ** 2,
            cov_NH=rho_NH * sigma_N * sigma_H,
            cov_HH=sigma_H ** 2,
        )

    def __repr__(self) -> str:
        return (
            f"GnssBaselineObs({self.id}: {self.from_point_id}->{self.to_point_id}, "
            f"dE={self.dE:.3f}, dN={self.dN:.3f}, dH={self.dH:.3f}m)"
        )


# Utility functions for angle conversions
def degrees_to_radians(degrees: float) -> float:
    """Convert degrees to radians."""
    return math.radians(degrees)


def radians_to_degrees(radians: float) -> float:
    """Convert radians to degrees."""
    return math.degrees(radians)


def arcseconds_to_radians(arcseconds: float) -> float:
    """Convert arc-seconds to radians."""
    return arcseconds * math.pi / (180.0 * 3600.0)


def radians_to_arcseconds(radians: float) -> float:
    """Convert radians to arc-seconds."""
    return radians * (180.0 * 3600.0) / math.pi


def dms_to_degrees(degrees: int, minutes: int, seconds: float) -> float:
    """Convert degrees-minutes-seconds to decimal degrees."""
    sign = -1 if degrees < 0 else 1
    return sign * (abs(degrees) + minutes / 60.0 + seconds / 3600.0)


def degrees_to_dms(decimal_degrees: float) -> tuple:
    """Convert decimal degrees to degrees-minutes-seconds tuple."""
    sign = -1 if decimal_degrees < 0 else 1
    dd = abs(decimal_degrees)
    degrees = int(dd)
    minutes = int((dd - degrees) * 60)
    seconds = (dd - degrees - minutes / 60.0) * 3600
    return (sign * degrees, minutes, seconds)
