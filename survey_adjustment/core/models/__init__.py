"""
Data models for survey network adjustment.

This module provides the core data structures:
- Point: Survey station with coordinates and constraints
- Observation: Base class and subclasses for measurements
- Network: Container for points and observations
- AdjustmentOptions: Configuration for the adjustment
"""

from .point import Point
from .observation import (
    Observation,
    ObservationType,
    DistanceObservation,
    DirectionObservation,
    AngleObservation,
    degrees_to_radians,
    radians_to_degrees,
    arcseconds_to_radians,
    radians_to_arcseconds,
    dms_to_degrees,
    degrees_to_dms
)
from .network import Network
from .options import AdjustmentOptions, RobustEstimator

__all__ = [
    # Point
    "Point",

    # Observations
    "Observation",
    "ObservationType",
    "DistanceObservation",
    "DirectionObservation",
    "AngleObservation",

    # Network
    "Network",

    # Options
    "AdjustmentOptions",
    "RobustEstimator",

    # Utility functions
    "degrees_to_radians",
    "radians_to_degrees",
    "arcseconds_to_radians",
    "radians_to_arcseconds",
    "dms_to_degrees",
    "degrees_to_dms",
]
