"""Settings dialog for Survey Adjustment plugin.

Provides a GUI for editing persistent plugin settings.
Settings are stored using QgsSettings and persist across QGIS sessions.
"""

from __future__ import annotations

from qgis.PyQt.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QPushButton,
    QTabWidget,
    QWidget,
    QGroupBox,
    QLabel,
)
from qgis.PyQt.QtCore import Qt

from ..settings import PluginSettings


class SettingsDialog(QDialog):
    """Plugin settings dialog with tabbed interface."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Survey Adjustment Settings")
        self.setMinimumWidth(450)
        self._build_ui()
        self._load()

    def _build_ui(self):
        """Build the dialog UI."""
        layout = QVBoxLayout(self)

        # Info label
        info = QLabel(
            "These settings provide default values for Processing algorithms.\n"
            "You can override them when running each algorithm."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: gray; margin-bottom: 10px;")
        layout.addWidget(info)

        # Tab widget
        tabs = QTabWidget()
        tabs.addTab(self._tab_units(), "Units")
        tabs.addTab(self._tab_solver(), "Solver")
        tabs.addTab(self._tab_robust(), "Robust")
        tabs.addTab(self._tab_outputs(), "Outputs")
        layout.addWidget(tabs)

        # Buttons
        btns = QHBoxLayout()

        self.btn_reset = QPushButton("Reset to Defaults")
        self.btn_reset.clicked.connect(self._reset)
        btns.addWidget(self.btn_reset)

        btns.addStretch()

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        btns.addWidget(self.btn_cancel)

        self.btn_ok = QPushButton("OK")
        self.btn_ok.setDefault(True)
        self.btn_ok.clicked.connect(self._save_and_close)
        btns.addWidget(self.btn_ok)

        layout.addLayout(btns)

    def _tab_units(self) -> QWidget:
        """Create the Units tab."""
        w = QWidget()
        f = QFormLayout(w)

        # Angle unit
        self.angle_unit = QComboBox()
        self.angle_unit.addItems(["degrees", "radians", "gradians"])
        self.angle_unit.setToolTip("Default unit for angle/direction input")
        f.addRow("Default angle unit:", self.angle_unit)

        # Angular sigma unit
        self.sigma_angle_unit = QComboBox()
        self.sigma_angle_unit.addItems(["arcseconds", "radians", "degrees"])
        self.sigma_angle_unit.setToolTip("Default unit for angular standard deviations")
        f.addRow("Angular sigma unit:", self.sigma_angle_unit)

        # Leveling sigma unit
        self.sigma_leveling_unit = QComboBox()
        self.sigma_leveling_unit.addItems(["mm", "m"])
        self.sigma_leveling_unit.setToolTip("Default unit for leveling standard deviations")
        f.addRow("Leveling sigma unit:", self.sigma_leveling_unit)

        return w

    def _tab_solver(self) -> QWidget:
        """Create the Solver tab."""
        w = QWidget()
        f = QFormLayout(w)

        # Max iterations
        self.max_iter = QSpinBox()
        self.max_iter.setRange(1, 500)
        self.max_iter.setToolTip("Maximum Gauss-Newton iterations before stopping")
        f.addRow("Max iterations:", self.max_iter)

        # Convergence tolerance
        self.tol = QDoubleSpinBox()
        self.tol.setDecimals(10)
        self.tol.setRange(1e-12, 1.0)
        self.tol.setToolTip("Convergence threshold for coordinate corrections")
        f.addRow("Convergence tolerance:", self.tol)

        # Confidence level
        self.conf = QDoubleSpinBox()
        self.conf.setDecimals(4)
        self.conf.setRange(0.50, 0.9999)
        self.conf.setSingleStep(0.05)
        self.conf.setToolTip("Confidence level for error ellipses (e.g., 0.95 = 95%)")
        f.addRow("Ellipse confidence level:", self.conf)

        # Outlier threshold
        self.outlier = QDoubleSpinBox()
        self.outlier.setDecimals(2)
        self.outlier.setRange(1.0, 10.0)
        self.outlier.setSingleStep(0.5)
        self.outlier.setToolTip("Standardized residual threshold for flagging outliers")
        f.addRow("Outlier threshold (sigma):", self.outlier)

        # Compute reliability
        self.compute_reliability = QCheckBox()
        self.compute_reliability.setToolTip("Compute redundancy numbers, MDB, and external reliability")
        f.addRow("Compute reliability measures:", self.compute_reliability)

        # Auto-datum
        self.auto_datum = QCheckBox()
        self.auto_datum.setToolTip("Automatically apply minimal datum constraints if needed")
        f.addRow("Auto-datum by default:", self.auto_datum)

        return w

    def _tab_robust(self) -> QWidget:
        """Create the Robust Estimation tab."""
        w = QWidget()
        f = QFormLayout(w)

        # Robust method
        self.robust_method = QComboBox()
        self.robust_method.addItems(["none", "huber", "danish", "igg3"])
        self.robust_method.setToolTip("Default robust estimation method (IRLS)")
        f.addRow("Default robust method:", self.robust_method)

        # Separator
        f.addRow(QLabel(""))
        f.addRow(QLabel("<b>Robust Parameters:</b>"))

        # Huber c
        self.huber_c = QDoubleSpinBox()
        self.huber_c.setDecimals(2)
        self.huber_c.setRange(0.5, 5.0)
        self.huber_c.setSingleStep(0.1)
        self.huber_c.setToolTip("Huber function tuning constant")
        f.addRow("Huber c:", self.huber_c)

        # Danish c
        self.danish_c = QDoubleSpinBox()
        self.danish_c.setDecimals(2)
        self.danish_c.setRange(0.5, 5.0)
        self.danish_c.setSingleStep(0.1)
        self.danish_c.setToolTip("Danish function tuning constant")
        f.addRow("Danish c:", self.danish_c)

        # IGG3 k0
        self.igg3_k0 = QDoubleSpinBox()
        self.igg3_k0.setDecimals(2)
        self.igg3_k0.setRange(0.5, 5.0)
        self.igg3_k0.setSingleStep(0.1)
        self.igg3_k0.setToolTip("IGG-III lower bound (start downweighting)")
        f.addRow("IGG-III k0:", self.igg3_k0)

        # IGG3 k1
        self.igg3_k1 = QDoubleSpinBox()
        self.igg3_k1.setDecimals(2)
        self.igg3_k1.setRange(1.0, 10.0)
        self.igg3_k1.setSingleStep(0.5)
        self.igg3_k1.setToolTip("IGG-III upper bound (zero weight)")
        f.addRow("IGG-III k1:", self.igg3_k1)

        return w

    def _tab_outputs(self) -> QWidget:
        """Create the Outputs tab."""
        w = QWidget()
        f = QFormLayout(w)

        # Auto-open HTML
        self.auto_open_html = QCheckBox()
        self.auto_open_html.setToolTip("Automatically open HTML report in browser when complete")
        f.addRow("Auto-open HTML report:", self.auto_open_html)

        # Include covariance in JSON
        self.include_cov_json = QCheckBox()
        self.include_cov_json.setToolTip("Include full covariance matrix in JSON output")
        f.addRow("Include covariance in JSON:", self.include_cov_json)

        # Ellipse vertices
        self.ellipse_vertices = QSpinBox()
        self.ellipse_vertices.setRange(16, 360)
        self.ellipse_vertices.setSingleStep(8)
        self.ellipse_vertices.setToolTip("Number of vertices for error ellipse polygons")
        f.addRow("Ellipse polygon vertices:", self.ellipse_vertices)

        # Residual vector scale
        self.resid_scale = QDoubleSpinBox()
        self.resid_scale.setDecimals(1)
        self.resid_scale.setRange(1.0, 1e9)
        self.resid_scale.setSingleStep(100.0)
        self.resid_scale.setToolTip("Scale factor for residual vector visualization")
        f.addRow("Residual vector scale:", self.resid_scale)

        return w

    def _load(self):
        """Load current settings into UI widgets."""
        # Units
        self.angle_unit.setCurrentText(PluginSettings.get("angle_unit"))
        self.sigma_angle_unit.setCurrentText(PluginSettings.get("sigma_angle_unit"))
        self.sigma_leveling_unit.setCurrentText(PluginSettings.get("sigma_leveling_unit"))

        # Solver
        self.max_iter.setValue(PluginSettings.get("max_iterations"))
        self.tol.setValue(PluginSettings.get("convergence_tol"))
        self.conf.setValue(PluginSettings.get("confidence_level"))
        self.outlier.setValue(PluginSettings.get("outlier_threshold"))
        self.compute_reliability.setChecked(PluginSettings.get("compute_reliability"))
        self.auto_datum.setChecked(PluginSettings.get("auto_datum_default"))

        # Robust
        self.robust_method.setCurrentText(PluginSettings.get("robust_method"))
        self.huber_c.setValue(PluginSettings.get("huber_c"))
        self.danish_c.setValue(PluginSettings.get("danish_c"))
        self.igg3_k0.setValue(PluginSettings.get("igg3_k0"))
        self.igg3_k1.setValue(PluginSettings.get("igg3_k1"))

        # Outputs
        self.auto_open_html.setChecked(PluginSettings.get("auto_open_html"))
        self.include_cov_json.setChecked(PluginSettings.get("include_covariance_json"))
        self.ellipse_vertices.setValue(PluginSettings.get("ellipse_vertices"))
        self.resid_scale.setValue(PluginSettings.get("residual_vector_scale"))

    def _save_and_close(self):
        """Save settings and close dialog."""
        # Units
        PluginSettings.set("angle_unit", self.angle_unit.currentText())
        PluginSettings.set("sigma_angle_unit", self.sigma_angle_unit.currentText())
        PluginSettings.set("sigma_leveling_unit", self.sigma_leveling_unit.currentText())

        # Solver
        PluginSettings.set("max_iterations", self.max_iter.value())
        PluginSettings.set("convergence_tol", self.tol.value())
        PluginSettings.set("confidence_level", self.conf.value())
        PluginSettings.set("outlier_threshold", self.outlier.value())
        PluginSettings.set("compute_reliability", self.compute_reliability.isChecked())
        PluginSettings.set("auto_datum_default", self.auto_datum.isChecked())

        # Robust
        PluginSettings.set("robust_method", self.robust_method.currentText())
        PluginSettings.set("huber_c", self.huber_c.value())
        PluginSettings.set("danish_c", self.danish_c.value())
        PluginSettings.set("igg3_k0", self.igg3_k0.value())
        PluginSettings.set("igg3_k1", self.igg3_k1.value())

        # Outputs
        PluginSettings.set("auto_open_html", self.auto_open_html.isChecked())
        PluginSettings.set("include_covariance_json", self.include_cov_json.isChecked())
        PluginSettings.set("ellipse_vertices", self.ellipse_vertices.value())
        PluginSettings.set("residual_vector_scale", self.resid_scale.value())

        self.accept()

    def _reset(self):
        """Reset all settings to defaults and reload UI."""
        PluginSettings.reset()
        self._load()
