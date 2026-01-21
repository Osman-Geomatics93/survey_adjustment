"""
Survey Adjustment & Network Analysis - QGIS Plugin

A comprehensive least-squares adjustment plugin for survey networks,
supporting distance, direction, and angle observations.

Conventions:
- Angles: Radians internally, converted from/to degrees at I/O boundary
- Azimuth: North = 0, clockwise positive (standard surveying convention)
- Coordinates: Easting (X), Northing (Y) - right-handed system
- Distance: Meters (ground/grid as specified)
- Standard deviation: Meters for distances, radians for angles
- Point IDs: String type to allow alphanumeric station names
"""

__version__ = "1.0.0"
__author__ = "Survey Adjustment Plugin"

from .core.models import Point, Network, AdjustmentOptions
from .core.models import (
    Observation,
    ObservationType,
    DistanceObservation,
    DirectionObservation,
    AngleObservation
)
from .core.results import AdjustmentResult, ErrorEllipse, ChiSquareTestResult

__all__ = [
    # Version
    "__version__",

    # Models
    "Point",
    "Network",
    "AdjustmentOptions",

    # Observations
    "Observation",
    "ObservationType",
    "DistanceObservation",
    "DirectionObservation",
    "AngleObservation",

    # Results
    "AdjustmentResult",
    "ErrorEllipse",
    "ChiSquareTestResult",
]
