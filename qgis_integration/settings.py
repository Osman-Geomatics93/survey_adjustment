"""Persistent plugin settings using QgsSettings.

This module provides a centralized settings manager that:
- Stores user preferences persistently in the QGIS profile
- Falls back to defaults when running outside QGIS (for testing)
- Provides defaults used by Processing algorithms
- Validates and clamps values to safe ranges

All QGIS imports are guarded so the module remains importable outside QGIS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union

try:
    from qgis.core import QgsSettings
except Exception:  # outside QGIS
    QgsSettings = None


# =============================================================================
# Type conversion helpers
# =============================================================================

def _to_bool(v: Any) -> bool:
    """
    Robust boolean conversion that handles all QgsSettings quirks.

    QgsSettings can return:
    - actual bool
    - int (0/1)
    - str ("true", "false", "0", "1", "yes", "no", etc.)
    - QVariant wrapping any of the above
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "t")
    # Fallback for QVariant or other types
    return bool(v)


def _to_int(v: Any) -> int:
    """Robust integer conversion."""
    try:
        return int(v)
    except (ValueError, TypeError):
        return 0


def _to_float(v: Any) -> float:
    """Robust float conversion."""
    try:
        return float(v)
    except (ValueError, TypeError):
        return 0.0


def _to_str(v: Any) -> str:
    """Robust string conversion."""
    if v is None:
        return ""
    return str(v)


# =============================================================================
# Validators / Clamps
# =============================================================================

def _clamp(min_val: float, max_val: float) -> Callable[[float], float]:
    """Return a clamping function for numeric values."""
    def clamp(v: float) -> float:
        return max(min_val, min(max_val, v))
    return clamp


def _clamp_positive() -> Callable[[float], float]:
    """Return a function that ensures value is positive (> 0)."""
    def clamp(v: float) -> float:
        return max(0.001, v)  # small epsilon to avoid zero
    return clamp


def _one_of(valid_values: Tuple[str, ...], fallback: str) -> Callable[[str], str]:
    """Return a function that validates string is one of allowed values."""
    def validate(v: str) -> str:
        v_lower = v.lower().strip()
        if v_lower in valid_values:
            return v_lower
        return fallback
    return validate


# Validator functions for each setting that needs validation
# Format: key -> (converter, validator_or_None)
VALIDATORS: Dict[str, Tuple[Callable, Optional[Callable]]] = {
    # Units - enum-like strings
    "angle_unit": (_to_str, _one_of(("degrees", "radians", "gradians"), "degrees")),
    "sigma_angle_unit": (_to_str, _one_of(("arcseconds", "radians", "degrees"), "arcseconds")),
    "sigma_leveling_unit": (_to_str, _one_of(("mm", "m"), "mm")),

    # Solver - numeric ranges
    "max_iterations": (_to_int, _clamp(1, 500)),
    "convergence_tol": (_to_float, _clamp(1e-12, 1.0)),
    "confidence_level": (_to_float, _clamp(0.50, 0.9999)),
    "outlier_threshold": (_to_float, _clamp(1.0, 10.0)),

    # Robust estimation
    "robust_method": (_to_str, _one_of(("none", "huber", "danish", "igg3"), "none")),
    "huber_c": (_to_float, _clamp_positive()),
    "danish_c": (_to_float, _clamp_positive()),
    "igg3_k0": (_to_float, _clamp_positive()),
    "igg3_k1": (_to_float, _clamp_positive()),

    # Output preferences
    "auto_open_html": (_to_bool, None),
    "include_covariance_json": (_to_bool, None),
    "ellipse_vertices": (_to_int, _clamp(16, 360)),
    "residual_vector_scale": (_to_float, _clamp(1.0, 1e9)),

    # Datum / reliability
    "auto_datum_default": (_to_bool, None),
    "compute_reliability": (_to_bool, None),
}


# =============================================================================
# Defaults dataclass
# =============================================================================

@dataclass(frozen=True)
class _Defaults:
    """Default values for all plugin settings."""

    # Units / display defaults
    angle_unit: str = "degrees"            # degrees, radians, gradians
    sigma_angle_unit: str = "arcseconds"   # arcseconds, radians, degrees
    sigma_leveling_unit: str = "mm"        # mm, m

    # Solver defaults
    max_iterations: int = 50
    convergence_tol: float = 1e-6
    confidence_level: float = 0.95
    outlier_threshold: float = 3.0

    # Robust estimation
    robust_method: str = "none"            # none, huber, danish, igg3
    huber_c: float = 1.5
    danish_c: float = 2.0
    igg3_k0: float = 1.5
    igg3_k1: float = 3.0

    # Output preferences
    auto_open_html: bool = True
    include_covariance_json: bool = False
    ellipse_vertices: int = 64
    residual_vector_scale: float = 1000.0

    # Datum UX
    auto_datum_default: bool = False

    # Reliability
    compute_reliability: bool = True


# =============================================================================
# Main settings class
# =============================================================================

class PluginSettings:
    """
    Persistent plugin settings stored in QGIS profile using QgsSettings.

    Safe to use outside QGIS - falls back to defaults when QgsSettings
    is not available (e.g., during unit testing).

    Features:
    - Type-safe: converts QgsSettings strings back to correct types
    - Validated: clamps numeric values to safe ranges
    - Defensive: handles corrupted/manual edits gracefully

    Usage:
        # Get a setting (always returns valid, clamped value)
        max_iter = PluginSettings.get("max_iterations")

        # Set a setting
        PluginSettings.set("max_iterations", 100)

        # Reset to defaults
        PluginSettings.reset()  # all settings
        PluginSettings.reset("max_iterations")  # single setting

        # Get all settings
        all_settings = PluginSettings.all()
    """

    PREFIX = "SurveyAdjustment/"
    DEFAULTS = _Defaults()

    @classmethod
    def _defaults_dict(cls) -> Dict[str, Any]:
        """Get defaults as a dictionary."""
        return cls.DEFAULTS.__dict__.copy()

    @classmethod
    def _convert_and_validate(cls, key: str, raw_value: Any, default: Any) -> Any:
        """
        Convert raw value to correct type and validate/clamp.

        Args:
            key: Setting key
            raw_value: Value from QgsSettings (may be wrong type)
            default: Default value (used for type inference if no validator)

        Returns:
            Converted and validated value
        """
        if key in VALIDATORS:
            converter, validator = VALIDATORS[key]
            try:
                value = converter(raw_value)
                if validator is not None:
                    value = validator(value)
                return value
            except Exception:
                # If anything goes wrong, return default
                return default

        # Fallback: simple type coercion based on default type
        if isinstance(default, bool):
            return _to_bool(raw_value)
        if isinstance(default, int):
            return _to_int(raw_value)
        if isinstance(default, float):
            return _to_float(raw_value)
        return raw_value

    @classmethod
    def get(cls, key: str) -> Any:
        """
        Get a setting value.

        The value is:
        1. Retrieved from QgsSettings (or default if outside QGIS)
        2. Converted to the correct type
        3. Validated/clamped to safe range

        Args:
            key: Setting key (must exist in _Defaults)

        Returns:
            The setting value, converted and validated

        Raises:
            KeyError: If the key is not a valid setting
        """
        defaults = cls._defaults_dict()
        if key not in defaults:
            raise KeyError(f"Unknown setting key: {key}")

        default = defaults[key]

        if QgsSettings is None:
            return default

        settings = QgsSettings()
        full_key = cls.PREFIX + key
        raw_value = settings.value(full_key, default)

        return cls._convert_and_validate(key, raw_value, default)

    @classmethod
    def set(cls, key: str, value: Any) -> None:
        """
        Set a setting value.

        The value is validated before storing.

        Args:
            key: Setting key (must exist in _Defaults)
            value: Value to store

        Raises:
            KeyError: If the key is not a valid setting
        """
        if QgsSettings is None:
            return

        defaults = cls._defaults_dict()
        if key not in defaults:
            raise KeyError(f"Unknown setting key: {key}")

        # Validate before storing
        validated = cls._convert_and_validate(key, value, defaults[key])
        QgsSettings().setValue(cls.PREFIX + key, validated)

    @classmethod
    def reset(cls, key: Union[str, None] = None) -> None:
        """
        Reset setting(s) to default values.

        Args:
            key: Setting key to reset, or None to reset all settings

        Raises:
            KeyError: If the key is not a valid setting
        """
        if QgsSettings is None:
            return

        defaults = cls._defaults_dict()
        s = QgsSettings()

        if key is None:
            # Reset all settings
            for k, v in defaults.items():
                s.setValue(cls.PREFIX + k, v)
            return

        if key not in defaults:
            raise KeyError(f"Unknown setting key: {key}")
        s.setValue(cls.PREFIX + key, defaults[key])

    @classmethod
    def all(cls) -> Dict[str, Any]:
        """Get all settings as a dictionary (all values validated)."""
        return {k: cls.get(k) for k in cls._defaults_dict()}

    @classmethod
    def keys(cls) -> list:
        """Get all setting keys."""
        return list(cls._defaults_dict().keys())

    @classmethod
    def is_default(cls, key: str) -> bool:
        """Check if a setting has its default value."""
        return cls.get(key) == cls._defaults_dict()[key]

    @classmethod
    def get_default(cls, key: str) -> Any:
        """Get the default value for a setting."""
        defaults = cls._defaults_dict()
        if key not in defaults:
            raise KeyError(f"Unknown setting key: {key}")
        return defaults[key]

    @classmethod
    def get_computation_snapshot(cls) -> Dict[str, Any]:
        """
        Get a snapshot of settings that affect computation/results.

        This is included in JSON output for debugging and support.
        Only includes settings that affect the adjustment computation
        or report formatting.

        Returns:
            Dictionary of computation-relevant settings
        """
        return {
            # Units (affect interpretation of input/output)
            "angle_unit": cls.get("angle_unit"),
            "sigma_angle_unit": cls.get("sigma_angle_unit"),
            "sigma_leveling_unit": cls.get("sigma_leveling_unit"),
            # Solver parameters
            "max_iterations": cls.get("max_iterations"),
            "convergence_tol": cls.get("convergence_tol"),
            "confidence_level": cls.get("confidence_level"),
            "outlier_threshold": cls.get("outlier_threshold"),
            # Robust estimation
            "robust_method": cls.get("robust_method"),
            "huber_c": cls.get("huber_c"),
            "danish_c": cls.get("danish_c"),
            "igg3_k0": cls.get("igg3_k0"),
            "igg3_k1": cls.get("igg3_k1"),
            # Datum
            "auto_datum_default": cls.get("auto_datum_default"),
            # Output
            "ellipse_vertices": cls.get("ellipse_vertices"),
            "residual_vector_scale": cls.get("residual_vector_scale"),
            "compute_reliability": cls.get("compute_reliability"),
        }
