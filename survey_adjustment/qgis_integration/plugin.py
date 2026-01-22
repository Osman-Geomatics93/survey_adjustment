"""QGIS plugin lifecycle.

This module must be importable outside QGIS so unit tests can run. Therefore,
all QGIS imports are performed lazily inside methods.
"""

from __future__ import annotations


class SurveyAdjustmentPlugin:
    """QGIS plugin class (registers/unregisters a Processing provider)."""

    def __init__(self, iface):
        self.iface = iface
        self._provider = None

    def initGui(self) -> None:
        from qgis.core import QgsApplication
        from .provider import SurveyAdjustmentProvider

        self._provider = SurveyAdjustmentProvider()
        QgsApplication.processingRegistry().addProvider(self._provider)

    def unload(self) -> None:
        try:
            from qgis.core import QgsApplication

            if self._provider is not None:
                QgsApplication.processingRegistry().removeProvider(self._provider)
        except Exception:
            # QGIS might already be shutting down; ignore.
            pass
        finally:
            self._provider = None
