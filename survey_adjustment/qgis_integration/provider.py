"""QgsProcessingProvider implementation.

This file should only be imported by QGIS at runtime. It uses QGIS classes, but
unit tests won't import it because the plugin entry is lazy.
"""

from __future__ import annotations

try:
    from qgis.core import QgsProcessingProvider
except Exception:  # pragma: no cover
    QgsProcessingProvider = object  # type: ignore


class SurveyAdjustmentProvider(QgsProcessingProvider):
    """Processing provider for Survey Adjustment algorithms."""

    def id(self) -> str:  # type: ignore[override]
        return "survey_adjustment"

    def name(self) -> str:  # type: ignore[override]
        return "Survey Adjustment"

    def longName(self) -> str:  # type: ignore[override]
        return "Survey Adjustment & Network Analysis"

    def loadAlgorithms(self) -> None:  # type: ignore[override]
        # Lazy import algorithms so provider can load even if optional deps missing.
        from .algorithms.validate_network import ValidateNetworkAlgorithm
        from .algorithms.adjust_network_2d import AdjustNetwork2DAlgorithm
        from .algorithms.adjust_leveling_1d import AdjustLeveling1DAlgorithm
        from .algorithms.adjust_network_3d_gnss import AdjustNetwork3DGnssAlgorithm

        self.addAlgorithm(ValidateNetworkAlgorithm())
        self.addAlgorithm(AdjustNetwork2DAlgorithm())
        self.addAlgorithm(AdjustLeveling1DAlgorithm())
        self.addAlgorithm(AdjustNetwork3DGnssAlgorithm())
