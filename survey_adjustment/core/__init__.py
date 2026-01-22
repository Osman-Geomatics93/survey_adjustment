"""
Core module for survey adjustment.

This module contains pure Python implementations with no QGIS dependencies.
It can be used standalone for testing or integration with other applications.
"""

from .models import (
    Point,
    Observation,
    ObservationType,
    DistanceObservation,
    DirectionObservation,
    AngleObservation,
    HeightDifferenceObservation,
    Network,
    AdjustmentOptions
)

from .results import (
    AdjustmentResult,
    ErrorEllipse,
    ChiSquareTestResult,
    ResidualInfo
)

from .solver import adjust_network_2d, adjust_leveling_1d

from .geometry import (
    ellipse_polygon_points,
    distance_residual_vector,
    direction_residual_vector,
    angle_residual_vector,
)

__all__ = [
    # Models
    "Point",
    "Observation",
    "ObservationType",
    "DistanceObservation",
    "DirectionObservation",
    "AngleObservation",
    "HeightDifferenceObservation",
    "Network",
    "AdjustmentOptions",

    # Results
    "AdjustmentResult",
    "ErrorEllipse",
    "ChiSquareTestResult",
    "ResidualInfo",

    # Solvers
    "adjust_network_2d",
    "adjust_leveling_1d",

    # Geometry
    "ellipse_polygon_points",
    "distance_residual_vector",
    "direction_residual_vector",
    "angle_residual_vector",
]
