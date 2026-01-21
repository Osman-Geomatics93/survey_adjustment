"""
JSON Schema definitions for survey adjustment data.

This module defines JSON schemas for validating input/output data structures.
Schemas follow the JSON Schema Draft-07 specification.
"""

from typing import Dict, Any, List, Optional
import json


# ============================================================================
# Input Schemas
# ============================================================================

POINT_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Point",
    "description": "A survey point (station) in the network",
    "type": "object",
    "properties": {
        "id": {
            "type": "string",
            "description": "Unique identifier for the point",
            "minLength": 1
        },
        "point_id": {
            "type": "string",
            "description": "Alternative key for point ID (CSV compatibility)",
            "minLength": 1
        },
        "name": {
            "type": "string",
            "description": "Human-readable name or description"
        },
        "easting": {
            "type": "number",
            "description": "X coordinate in meters"
        },
        "northing": {
            "type": "number",
            "description": "Y coordinate in meters"
        },
        "fixed_easting": {
            "type": "boolean",
            "description": "If true, easting is held fixed",
            "default": False
        },
        "fixed_northing": {
            "type": "boolean",
            "description": "If true, northing is held fixed",
            "default": False
        },
        "fixed_e": {
            "type": ["boolean", "string"],
            "description": "Alternative key for fixed_easting (CSV compatibility)"
        },
        "fixed_n": {
            "type": ["boolean", "string"],
            "description": "Alternative key for fixed_northing (CSV compatibility)"
        },
        "sigma_easting": {
            "type": ["number", "null"],
            "description": "A priori standard deviation of easting in meters",
            "minimum": 0
        },
        "sigma_northing": {
            "type": ["number", "null"],
            "description": "A priori standard deviation of northing in meters",
            "minimum": 0
        },
        "sigma_e": {
            "type": ["number", "null"],
            "description": "Alternative key for sigma_easting (CSV compatibility)"
        },
        "sigma_n": {
            "type": ["number", "null"],
            "description": "Alternative key for sigma_northing (CSV compatibility)"
        }
    },
    "oneOf": [
        {"required": ["id", "easting", "northing"]},
        {"required": ["point_id", "easting", "northing"]}
    ]
}

DISTANCE_OBSERVATION_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Distance Observation",
    "description": "Distance measurement between two points",
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "obs_id": {"type": "string"},
        "obs_type": {"const": "distance"},
        "from_point_id": {"type": "string", "minLength": 1},
        "from_point": {"type": "string", "minLength": 1},
        "to_point_id": {"type": "string", "minLength": 1},
        "to_point": {"type": "string", "minLength": 1},
        "value": {"type": "number", "exclusiveMinimum": 0},
        "distance": {"type": "number", "exclusiveMinimum": 0},
        "sigma": {"type": "number", "exclusiveMinimum": 0},
        "sigma_distance": {"type": "number", "exclusiveMinimum": 0},
        "enabled": {"type": "boolean", "default": True}
    },
    "oneOf": [
        {"required": ["from_point_id", "to_point_id", "value", "sigma"]},
        {"required": ["from_point", "to_point", "distance", "sigma_distance"]}
    ]
}

DIRECTION_OBSERVATION_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Direction Observation",
    "description": "Direction measurement from one point to another",
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "obs_id": {"type": "string"},
        "obs_type": {"const": "direction"},
        "from_point_id": {"type": "string", "minLength": 1},
        "from_point": {"type": "string", "minLength": 1},
        "to_point_id": {"type": "string", "minLength": 1},
        "to_point": {"type": "string", "minLength": 1},
        "value": {"type": "number"},
        "direction": {"type": "number"},
        "sigma": {"type": "number", "exclusiveMinimum": 0},
        "sigma_direction": {"type": "number", "exclusiveMinimum": 0},
        "set_id": {"type": "string"},
        "enabled": {"type": "boolean", "default": True}
    },
    "oneOf": [
        {"required": ["from_point_id", "to_point_id", "value", "sigma"]},
        {"required": ["from_point", "to_point", "direction", "sigma_direction"]}
    ]
}

ANGLE_OBSERVATION_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Angle Observation",
    "description": "Angle measurement at a point between two other points",
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "obs_id": {"type": "string"},
        "obs_type": {"const": "angle"},
        "at_point_id": {"type": "string", "minLength": 1},
        "at_point": {"type": "string", "minLength": 1},
        "from_point_id": {"type": "string", "minLength": 1},
        "from_point": {"type": "string", "minLength": 1},
        "to_point_id": {"type": "string", "minLength": 1},
        "to_point": {"type": "string", "minLength": 1},
        "value": {"type": "number"},
        "angle": {"type": "number"},
        "sigma": {"type": "number", "exclusiveMinimum": 0},
        "sigma_angle": {"type": "number", "exclusiveMinimum": 0},
        "enabled": {"type": "boolean", "default": True}
    },
    "oneOf": [
        {"required": ["at_point_id", "from_point_id", "to_point_id", "value", "sigma"]},
        {"required": ["at_point", "from_point", "to_point", "angle", "sigma_angle"]}
    ]
}

NETWORK_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Survey Network",
    "description": "Complete survey network with points and observations",
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "points": {
            "oneOf": [
                {
                    "type": "object",
                    "additionalProperties": {"$ref": "#/definitions/point"}
                },
                {
                    "type": "array",
                    "items": {"$ref": "#/definitions/point"}
                }
            ]
        },
        "observations": {
            "type": "array",
            "items": {
                "oneOf": [
                    {"$ref": "#/definitions/distance_observation"},
                    {"$ref": "#/definitions/direction_observation"},
                    {"$ref": "#/definitions/angle_observation"}
                ]
            }
        }
    },
    "required": ["points", "observations"],
    "definitions": {
        "point": POINT_SCHEMA,
        "distance_observation": DISTANCE_OBSERVATION_SCHEMA,
        "direction_observation": DIRECTION_OBSERVATION_SCHEMA,
        "angle_observation": ANGLE_OBSERVATION_SCHEMA
    }
}

ADJUSTMENT_OPTIONS_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Adjustment Options",
    "description": "Configuration options for least-squares adjustment",
    "type": "object",
    "properties": {
        "max_iterations": {
            "type": "integer",
            "minimum": 1,
            "default": 10
        },
        "convergence_threshold": {
            "type": "number",
            "exclusiveMinimum": 0,
            "default": 1e-8
        },
        "confidence_level": {
            "type": "number",
            "exclusiveMinimum": 0,
            "exclusiveMaximum": 1,
            "default": 0.95
        },
        "a_priori_variance": {
            "type": "number",
            "exclusiveMinimum": 0,
            "default": 1.0
        },
        "compute_covariances": {
            "type": "boolean",
            "default": True
        },
        "robust_estimator": {
            "type": ["string", "null"],
            "enum": [None, "none", "huber", "danish", "hampel", "iggg3"]
        },
        "compute_error_ellipses": {
            "type": "boolean",
            "default": True
        },
        "outlier_threshold": {
            "type": "number",
            "exclusiveMinimum": 0,
            "default": 3.0
        },
        "alpha_local": {
            "type": "number",
            "exclusiveMinimum": 0,
            "exclusiveMaximum": 1,
            "default": 0.01
        },
        "mdb_power": {
            "type": "number",
            "exclusiveMinimum": 0,
            "exclusiveMaximum": 1,
            "default": 0.8
        },
        "compute_reliability": {
            "type": "boolean",
            "default": True
        },
        "angle_units_degrees": {
            "type": "boolean",
            "default": True
        },
        "sigma_units_arcseconds": {
            "type": "boolean",
            "default": False
        }
    }
}


# ============================================================================
# Output Schemas
# ============================================================================

ERROR_ELLIPSE_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Error Ellipse",
    "type": "object",
    "properties": {
        "point_id": {"type": "string"},
        "semi_major_m": {"type": "number", "minimum": 0},
        "semi_minor_m": {"type": "number", "minimum": 0},
        "orientation_deg": {"type": "number"},
        "confidence_level": {"type": "number", "minimum": 0, "maximum": 1}
    },
    "required": ["point_id", "semi_major_m", "semi_minor_m", "orientation_deg", "confidence_level"]
}

RESIDUAL_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Residual",
    "type": "object",
    "properties": {
        "obs_id": {"type": "string"},
        "obs_type": {"type": "string", "enum": ["distance", "direction", "angle"]},
        "from_point": {"type": "string"},
        "to_point": {"type": "string"},
        "at_point": {"type": "string"},
        "observed": {"type": "number"},
        "computed": {"type": "number"},
        "residual": {"type": "number"},
        "standardized_residual": {"type": "number"},
        "redundancy_number": {"type": ["number", "null"]},
        "mdb": {"type": ["number", "null"]},
        "external_reliability": {"type": ["number", "null"]},
        "is_outlier_candidate": {"type": "boolean"},
        "flagged": {"type": "boolean"}
    },
    "required": ["obs_id", "obs_type", "observed", "computed", "residual", "standardized_residual"]
}

ADJUSTMENT_RESULT_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Adjustment Result",
    "description": "Complete output from least-squares adjustment",
    "type": "object",
    "properties": {
        "metadata": {
            "type": "object",
            "properties": {
                "plugin_version": {"type": "string"},
                "timestamp": {"type": "string", "format": "date-time"},
                "network_name": {"type": "string"}
            }
        },
        "input_summary": {
            "type": "object",
            "properties": {
                "num_points": {"type": "integer", "minimum": 0},
                "num_fixed_points": {"type": "integer", "minimum": 0},
                "num_observations": {"type": "integer", "minimum": 0},
                "observations_by_type": {
                    "type": "object",
                    "additionalProperties": {"type": "integer"}
                }
            }
        },
        "adjustment": {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "iterations": {"type": "integer", "minimum": 0},
                "converged": {"type": "boolean"},
                "degrees_of_freedom": {"type": "integer"},
                "variance_factor": {"type": "number"},
                "a_posteriori_sigma0": {"type": "number"},
                "error_message": {"type": ["string", "null"]},
                "messages": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["success"]
        },
        "global_test": {
            "type": ["object", "null"],
            "properties": {
                "test_name": {"type": "string"},
                "test_statistic": {"type": "number"},
                "critical_lower": {"type": "number"},
                "critical_upper": {"type": "number"},
                "confidence_level": {"type": "number"},
                "passed": {"type": "boolean"},
                "p_value": {"type": ["number", "null"]},
                "degrees_of_freedom": {"type": ["integer", "null"]}
            }
        },
        "adjusted_points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "easting": {"type": "number"},
                    "northing": {"type": "number"},
                    "fixed_easting": {"type": "boolean"},
                    "fixed_northing": {"type": "boolean"},
                    "sigma_easting": {"type": ["number", "null"]},
                    "sigma_northing": {"type": ["number", "null"]}
                },
                "required": ["id", "easting", "northing"]
            }
        },
        "error_ellipses": {
            "type": "array",
            "items": {"$ref": "#/definitions/error_ellipse"}
        },
        "residuals": {
            "type": "array",
            "items": {"$ref": "#/definitions/residual"}
        },
        "flagged_observations": {
            "type": "array",
            "items": {"type": "string"}
        },
        "covariance_matrix": {
            "type": ["array", "null"],
            "items": {
                "type": "array",
                "items": {"type": "number"}
            }
        },
        "point_covariances": {
            "type": "object",
            "additionalProperties": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "number"}
                }
            }
        }
    },
    "required": ["metadata", "adjustment"],
    "definitions": {
        "error_ellipse": ERROR_ELLIPSE_SCHEMA,
        "residual": RESIDUAL_SCHEMA
    }
}


# ============================================================================
# Validation Functions
# ============================================================================

def validate_json(data: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    """
    Validate JSON data against a schema.

    This is a simple validator that doesn't require external libraries.
    For production use, consider using the 'jsonschema' library.

    Args:
        data: Dictionary to validate
        schema: JSON schema to validate against

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Check required fields
    required = schema.get("required", [])
    for field in required:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    # Check types
    properties = schema.get("properties", {})
    for field, value in data.items():
        if field in properties:
            prop_schema = properties[field]
            expected_type = prop_schema.get("type")
            if expected_type:
                if not _check_type(value, expected_type):
                    errors.append(f"Field '{field}' has wrong type: expected {expected_type}")

            # Check minimum/maximum
            if isinstance(value, (int, float)):
                if "minimum" in prop_schema and value < prop_schema["minimum"]:
                    errors.append(f"Field '{field}' is below minimum: {prop_schema['minimum']}")
                if "maximum" in prop_schema and value > prop_schema["maximum"]:
                    errors.append(f"Field '{field}' is above maximum: {prop_schema['maximum']}")
                if "exclusiveMinimum" in prop_schema and value <= prop_schema["exclusiveMinimum"]:
                    errors.append(f"Field '{field}' must be greater than {prop_schema['exclusiveMinimum']}")
                if "exclusiveMaximum" in prop_schema and value >= prop_schema["exclusiveMaximum"]:
                    errors.append(f"Field '{field}' must be less than {prop_schema['exclusiveMaximum']}")

    return errors


def _check_type(value: Any, expected_type: Any) -> bool:
    """Check if value matches expected JSON schema type(s)."""
    if isinstance(expected_type, list):
        return any(_check_type(value, t) for t in expected_type)

    type_map = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None)
    }

    if expected_type in type_map:
        return isinstance(value, type_map[expected_type])

    return True


def get_schema(schema_name: str) -> Optional[Dict[str, Any]]:
    """
    Get a schema by name.

    Args:
        schema_name: Name of the schema (e.g., 'point', 'network', 'result')

    Returns:
        Schema dictionary or None if not found
    """
    schemas = {
        "point": POINT_SCHEMA,
        "distance_observation": DISTANCE_OBSERVATION_SCHEMA,
        "direction_observation": DIRECTION_OBSERVATION_SCHEMA,
        "angle_observation": ANGLE_OBSERVATION_SCHEMA,
        "network": NETWORK_SCHEMA,
        "options": ADJUSTMENT_OPTIONS_SCHEMA,
        "error_ellipse": ERROR_ELLIPSE_SCHEMA,
        "residual": RESIDUAL_SCHEMA,
        "result": ADJUSTMENT_RESULT_SCHEMA
    }
    return schemas.get(schema_name)


def export_schemas(output_path: str) -> None:
    """
    Export all schemas to a JSON file.

    Args:
        output_path: Path to write the schemas file
    """
    all_schemas = {
        "point": POINT_SCHEMA,
        "distance_observation": DISTANCE_OBSERVATION_SCHEMA,
        "direction_observation": DIRECTION_OBSERVATION_SCHEMA,
        "angle_observation": ANGLE_OBSERVATION_SCHEMA,
        "network": NETWORK_SCHEMA,
        "options": ADJUSTMENT_OPTIONS_SCHEMA,
        "error_ellipse": ERROR_ELLIPSE_SCHEMA,
        "residual": RESIDUAL_SCHEMA,
        "result": ADJUSTMENT_RESULT_SCHEMA
    }

    with open(output_path, 'w') as f:
        json.dump(all_schemas, f, indent=2)
