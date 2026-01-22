"""QGIS plugin lifecycle.

This module must be importable outside QGIS so unit tests can run. Therefore,
all QGIS imports are performed lazily inside methods.
"""

from __future__ import annotations

import os


class SurveyAdjustmentPlugin:
    """QGIS plugin class (registers/unregisters a Processing provider)."""

    def __init__(self, iface):
        self.iface = iface
        self._provider = None
        self._settings_action = None
        self._reset_action = None

    def initGui(self) -> None:
        from qgis.core import QgsApplication
        from qgis.PyQt.QtWidgets import QAction
        from qgis.PyQt.QtGui import QIcon
        from .provider import SurveyAdjustmentProvider

        # Register Processing provider
        self._provider = SurveyAdjustmentProvider()
        QgsApplication.processingRegistry().addProvider(self._provider)

        # Icon for menu items
        icon_path = os.path.join(os.path.dirname(__file__), "..", "icon.png")
        icon = QIcon(icon_path) if os.path.exists(icon_path) else QIcon()

        # Add Settings menu item
        self._settings_action = QAction(
            icon,
            "Survey Adjustment Settings...",
            self.iface.mainWindow()
        )
        self._settings_action.triggered.connect(self._show_settings)
        self.iface.addPluginToMenu("Survey Adjustment", self._settings_action)

        # Add Reset Settings menu item
        self._reset_action = QAction(
            icon,
            "Reset Settings to Defaults",
            self.iface.mainWindow()
        )
        self._reset_action.triggered.connect(self._reset_settings)
        self.iface.addPluginToMenu("Survey Adjustment", self._reset_action)

    def _show_settings(self) -> None:
        """Open the settings dialog."""
        from .gui.settings_dialog import SettingsDialog  # lazy import
        dlg = SettingsDialog(self.iface.mainWindow())
        dlg.exec_()

    def _reset_settings(self) -> None:
        """Reset all settings to defaults and show confirmation."""
        from qgis.core import Qgis
        from .settings import PluginSettings  # lazy import

        PluginSettings.reset()

        # Show message bar confirmation
        self.iface.messageBar().pushMessage(
            "Survey Adjustment",
            "Settings reset to defaults.",
            level=Qgis.Info,
            duration=5
        )

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

        # Remove menu items
        if self._settings_action is not None:
            try:
                self.iface.removePluginMenu("Survey Adjustment", self._settings_action)
            except Exception:
                pass
            finally:
                self._settings_action = None

        if self._reset_action is not None:
            try:
                self.iface.removePluginMenu("Survey Adjustment", self._reset_action)
            except Exception:
                pass
            finally:
                self._reset_action = None
