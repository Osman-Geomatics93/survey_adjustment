"""
Result classes for survey network adjustment.

This module provides the output data structures:
- AdjustmentResult: Complete results from adjustment
- ErrorEllipse: Confidence ellipse parameters
- ChiSquareTestResult: Global test results
- ResidualInfo: Detailed residual information
"""

from .adjustment_result import (
    AdjustmentResult,
    ErrorEllipse,
    ChiSquareTestResult,
    ResidualInfo
)

from .schemas import (
    POINT_SCHEMA,
    DISTANCE_OBSERVATION_SCHEMA,
    DIRECTION_OBSERVATION_SCHEMA,
    ANGLE_OBSERVATION_SCHEMA,
    NETWORK_SCHEMA,
    ADJUSTMENT_OPTIONS_SCHEMA,
    ERROR_ELLIPSE_SCHEMA,
    RESIDUAL_SCHEMA,
    ADJUSTMENT_RESULT_SCHEMA,
    validate_json,
    get_schema,
    export_schemas
)

__all__ = [
    # Result classes
    "AdjustmentResult",
    "ErrorEllipse",
    "ChiSquareTestResult",
    "ResidualInfo",

    # Schemas
    "POINT_SCHEMA",
    "DISTANCE_OBSERVATION_SCHEMA",
    "DIRECTION_OBSERVATION_SCHEMA",
    "ANGLE_OBSERVATION_SCHEMA",
    "NETWORK_SCHEMA",
    "ADJUSTMENT_OPTIONS_SCHEMA",
    "ERROR_ELLIPSE_SCHEMA",
    "RESIDUAL_SCHEMA",
    "ADJUSTMENT_RESULT_SCHEMA",

    # Validation functions
    "validate_json",
    "get_schema",
    "export_schemas",
]
