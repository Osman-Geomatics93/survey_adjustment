"""
Point class for survey network adjustment.

Conventions:
- Coordinates: Easting (X), Northing (Y) - right-handed system
- Units: Meters for coordinates and standard deviations
- Point IDs: String type to allow alphanumeric station names
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class Point:
    """
    Represents a survey point (station) in the network.

    A point can be:
    - Fixed in both coordinates (control point)
    - Fixed in one coordinate only
    - Free (adjusted) with optional a priori standard deviations

    Attributes:
        id: Unique identifier for the point (alphanumeric allowed)
        name: Human-readable name or description
        easting: X coordinate in meters (approximate for free points)
        northing: Y coordinate in meters (approximate for free points)
        fixed_easting: If True, easting is held fixed during adjustment
        fixed_northing: If True, northing is held fixed during adjustment
        sigma_easting: A priori standard deviation of easting (meters), None if unknown
        sigma_northing: A priori standard deviation of northing (meters), None if unknown
    """

    id: str
    name: str
    easting: float
    northing: float
    fixed_easting: bool = False
    fixed_northing: bool = False
    sigma_easting: Optional[float] = None
    sigma_northing: Optional[float] = None

    def __post_init__(self):
        """Validate point data after initialization."""
        if not self.id:
            raise ValueError("Point ID cannot be empty")
        if not isinstance(self.id, str):
            raise ValueError("Point ID must be a string")

        # Ensure coordinates are numeric
        self.easting = float(self.easting)
        self.northing = float(self.northing)

        # Validate standard deviations if provided
        if self.sigma_easting is not None and self.sigma_easting < 0:
            raise ValueError("sigma_easting cannot be negative")
        if self.sigma_northing is not None and self.sigma_northing < 0:
            raise ValueError("sigma_northing cannot be negative")

    @property
    def is_fixed(self) -> bool:
        """Check if point is completely fixed (both coordinates)."""
        return self.fixed_easting and self.fixed_northing

    @property
    def is_free(self) -> bool:
        """Check if point is completely free (both coordinates adjustable)."""
        return not self.fixed_easting and not self.fixed_northing

    @property
    def is_partially_fixed(self) -> bool:
        """Check if point has mixed fixed/free coordinates."""
        return self.fixed_easting != self.fixed_northing

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize point to dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            "id": self.id,
            "name": self.name,
            "easting": self.easting,
            "northing": self.northing,
            "fixed_easting": self.fixed_easting,
            "fixed_northing": self.fixed_northing,
            "sigma_easting": self.sigma_easting,
            "sigma_northing": self.sigma_northing
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Point':
        """
        Create a Point from a dictionary.

        Args:
            data: Dictionary with point attributes

        Returns:
            New Point instance

        Raises:
            KeyError: If required fields are missing
            ValueError: If data is invalid
        """
        return cls(
            id=str(data["id"]) if "id" in data else str(data["point_id"]),
            name=data.get("name", ""),
            easting=float(data["easting"]),
            northing=float(data["northing"]),
            fixed_easting=_parse_bool(data.get("fixed_easting", data.get("fixed_e", False))),
            fixed_northing=_parse_bool(data.get("fixed_northing", data.get("fixed_n", False))),
            sigma_easting=_parse_optional_float(data.get("sigma_easting", data.get("sigma_e"))),
            sigma_northing=_parse_optional_float(data.get("sigma_northing", data.get("sigma_n")))
        )

    def __repr__(self) -> str:
        """Return string representation of the point."""
        status = "fixed" if self.is_fixed else ("partial" if self.is_partially_fixed else "free")
        return f"Point({self.id}, E={self.easting:.3f}, N={self.northing:.3f}, {status})"


def _parse_bool(value: Any) -> bool:
    """Parse a value to boolean, handling string representations."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'y')
    return bool(value)


def _parse_optional_float(value: Any) -> Optional[float]:
    """Parse a value to optional float, handling empty strings and None."""
    if value is None or value == '' or value == 'None':
        return None
    return float(value)
