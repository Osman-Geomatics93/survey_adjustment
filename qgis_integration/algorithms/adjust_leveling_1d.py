"""Adjust leveling network (1D) (QGIS Processing algorithm).

This module is intended to run inside QGIS. It remains importable outside QGIS
via guarded imports, but execution requires QGIS.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

try:  # pragma: no cover
    from qgis.core import (
        QgsProcessing,
        QgsProcessingAlgorithm,
        QgsProcessingParameterFile,
        QgsProcessingParameterFileDestination,
        QgsProcessingParameterNumber,
        QgsProcessingParameterBoolean,
        QgsProcessingParameterEnum,
        QgsProcessingParameterFeatureSink,
        QgsProcessingException,
        QgsFields,
        QgsField,
        QgsFeature,
        QgsGeometry,
        QgsPointXY,
        QgsWkbTypes,
        QgsCoordinateReferenceSystem,
        QgsProject,
    )
    from qgis.PyQt.QtCore import QVariant
    QGIS_AVAILABLE = True
except Exception:  # pragma: no cover
    # Stubs for import-time safety outside QGIS
    class _QgsProcessingStub:
        TypeVectorPoint = 0
        TypeVectorPolygon = 2
        TypeVectorLine = 1
        TypeVector = 5
    QgsProcessing = _QgsProcessingStub  # type: ignore
    QgsProcessingAlgorithm = object  # type: ignore
    QgsProcessingParameterFile = object  # type: ignore
    QgsProcessingParameterFileDestination = object  # type: ignore
    QgsProcessingParameterNumber = object  # type: ignore
    QgsProcessingParameterBoolean = object  # type: ignore
    QgsProcessingParameterEnum = object  # type: ignore
    QgsProcessingParameterFeatureSink = object  # type: ignore
    QgsProcessingException = RuntimeError
    QgsFields = object  # type: ignore
    QgsField = object  # type: ignore
    QgsFeature = object  # type: ignore
    QgsGeometry = object  # type: ignore
    QgsPointXY = object  # type: ignore
    QgsWkbTypes = object  # type: ignore
    QgsCoordinateReferenceSystem = object  # type: ignore
    QgsProject = object  # type: ignore
    QVariant = object  # type: ignore
    QGIS_AVAILABLE = False

from ..io.observations import (
    parse_leveling_csv,
    parse_leveling_points_csv,
)
from ..settings import PluginSettings

from ...core.models.options import AdjustmentOptions
from ...core.models.network import Network
from ...core.solver.least_squares_1d import adjust_leveling_1d
from ...core.reports.html_report import save_html_report


class AdjustLeveling1DAlgorithm(QgsProcessingAlgorithm):
    """Processing algorithm: run 1D least-squares leveling adjustment."""

    # Input parameters
    INPUT_POINTS = "POINTS_CSV"
    INPUT_HDIFF = "HDIFF_CSV"

    SIGMA_UNIT = "SIGMA_UNIT"
    COMPUTE_RELIABILITY = "COMPUTE_RELIABILITY"
    ROBUST_METHOD = "ROBUST_METHOD"
    AUTO_DATUM = "AUTO_DATUM"

    # Robust method options mapping
    ROBUST_OPTIONS = ["None (Standard LS)", "Huber", "Danish", "IGG-III"]
    ROBUST_VALUES = [None, "huber", "danish", "igg3"]

    # Output parameters
    OUTPUT_JSON = "OUTPUT_JSON"
    OUTPUT_HTML = "OUTPUT_HTML"
    OUTPUT_GPKG = "OUTPUT_GPKG"

    # Feature sink outputs
    OUTPUT_POINTS = "OUTPUT_POINTS"
    OUTPUT_RESIDUALS = "OUTPUT_RESIDUALS"

    def name(self):  # type: ignore[override]
        return "adjust_leveling_1d"

    def displayName(self):  # type: ignore[override]
        return "Adjust Leveling (1D)"

    def group(self):  # type: ignore[override]
        return "Survey Adjustment"

    def groupId(self):  # type: ignore[override]
        return "survey_adjustment"

    def shortHelpString(self):  # type: ignore[override]
        return (
            "Runs a 1D least-squares adjustment for leveling networks.\n\n"
            "Inputs:\n"
            "- Points CSV with heights (point_id, height, fixed_height)\n"
            "- Height differences CSV (from, to, dh, sigma)\n\n"
            "Outputs:\n"
            "- JSON and HTML reports\n"
            "- Adjusted Points layer (with adjusted heights)\n"
            "- Residuals Table (with statistics)\n\n"
            "Optionally exports all layers to a GeoPackage.\n\n"
            "Note: Default values come from Plugins → Survey Adjustment → Settings. "
            "QGIS may remember your last-used values per tool."
        )

    def createInstance(self):  # type: ignore[override]
        return AdjustLeveling1DAlgorithm()

    def initAlgorithm(self, config=None):  # type: ignore[override]
        # Input files
        self.addParameter(QgsProcessingParameterFile(
            self.INPUT_POINTS,
            "Points CSV (with heights)",
        ))
        self.addParameter(QgsProcessingParameterFile(
            self.INPUT_HDIFF,
            "Height Differences CSV",
        ))

        # Processing options - defaults from PluginSettings
        # Map sigma_leveling_unit setting to boolean
        sigma_unit = PluginSettings.get("sigma_leveling_unit")
        sigma_mm_default = sigma_unit == "mm"

        self.addParameter(QgsProcessingParameterBoolean(
            self.SIGMA_UNIT,
            "Sigma in millimeters (otherwise meters)",
            defaultValue=sigma_mm_default,
        ))
        self.addParameter(QgsProcessingParameterBoolean(
            self.COMPUTE_RELIABILITY,
            "Compute reliability measures",
            defaultValue=PluginSettings.get("compute_reliability"),
        ))

        # Map robust_method setting to enum index
        robust_method_setting = PluginSettings.get("robust_method")
        robust_method_map = {"none": 0, "huber": 1, "danish": 2, "igg3": 3}
        robust_default_idx = robust_method_map.get(robust_method_setting, 0)

        self.addParameter(QgsProcessingParameterEnum(
            self.ROBUST_METHOD,
            "Robust estimation method",
            options=self.ROBUST_OPTIONS,
            defaultValue=robust_default_idx,
        ))
        self.addParameter(QgsProcessingParameterBoolean(
            self.AUTO_DATUM,
            "Auto-apply minimal datum constraints",
            defaultValue=PluginSettings.get("auto_datum_default"),
        ))

        # Report outputs
        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT_JSON,
            "Adjustment result (JSON)",
            fileFilter="JSON (*.json)",
        ))
        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT_HTML,
            "Adjustment report (HTML)",
            fileFilter="HTML (*.html)",
        ))

        # GeoPackage output (optional)
        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT_GPKG,
            "GeoPackage output (optional)",
            fileFilter="GeoPackage (*.gpkg)",
            optional=True,
        ))

        # Feature sink outputs
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT_POINTS,
            "Adjusted Points",
            type=QgsProcessing.TypeVectorPoint,
        ))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT_RESIDUALS,
            "Residuals Table",
            type=QgsProcessing.TypeVector,
        ))

    def _create_points_fields(self) -> QgsFields:
        """Create fields for adjusted points layer."""
        fields = QgsFields()
        fields.append(QgsField("point_id", QVariant.String))
        fields.append(QgsField("H", QVariant.Double))
        fields.append(QgsField("sH", QVariant.Double))
        fields.append(QgsField("fixedH", QVariant.Bool))
        return fields

    def _create_residuals_fields(self) -> QgsFields:
        """Create fields for residuals table."""
        fields = QgsFields()
        fields.append(QgsField("obs_id", QVariant.String))
        fields.append(QgsField("from_id", QVariant.String))
        fields.append(QgsField("to_id", QVariant.String))
        fields.append(QgsField("observed", QVariant.Double))
        fields.append(QgsField("computed", QVariant.Double))
        fields.append(QgsField("v", QVariant.Double))
        fields.append(QgsField("w", QVariant.Double))
        fields.append(QgsField("redund", QVariant.Double))
        fields.append(QgsField("mdb", QVariant.Double))
        fields.append(QgsField("ext_rel", QVariant.Double))
        fields.append(QgsField("is_outlier", QVariant.Bool))
        return fields

    def processAlgorithm(self, parameters, context, feedback):  # type: ignore[override]
        # Read parameters
        points_path = self.parameterAsFile(parameters, self.INPUT_POINTS, context)
        hdiff_path = self.parameterAsFile(parameters, self.INPUT_HDIFF, context)

        sigma_in_mm = self.parameterAsBool(parameters, self.SIGMA_UNIT, context)
        compute_reliability = self.parameterAsBool(parameters, self.COMPUTE_RELIABILITY, context)
        robust_idx = self.parameterAsEnum(parameters, self.ROBUST_METHOD, context)
        robust_method = self.ROBUST_VALUES[robust_idx]
        auto_datum = self.parameterAsBool(parameters, self.AUTO_DATUM, context)

        out_json = self.parameterAsFileOutput(parameters, self.OUTPUT_JSON, context)
        out_html = self.parameterAsFileOutput(parameters, self.OUTPUT_HTML, context)
        out_gpkg = self.parameterAsFileOutput(parameters, self.OUTPUT_GPKG, context)

        # Parse input data
        feedback.pushInfo("Parsing input data...")

        points = parse_leveling_points_csv(points_path)
        hdiff_obs = parse_leveling_csv(
            hdiff_path,
            sigma_unit="mm" if sigma_in_mm else "m",
        )

        # Build network
        net = Network(name="Leveling Network", points=points)
        for obs in hdiff_obs:
            net.add_observation(obs)

        feedback.pushInfo(f"Network: {len(net.points)} points, {len(hdiff_obs)} height differences")

        # Configure options
        from ...core.models.options import RobustEstimator
        robust_enum = None
        if robust_method == "huber":
            robust_enum = RobustEstimator.HUBER
        elif robust_method == "danish":
            robust_enum = RobustEstimator.DANISH
        elif robust_method == "igg3":
            robust_enum = RobustEstimator.IGG3

        options = AdjustmentOptions(
            max_iterations=1,  # Linear problem
            compute_reliability=compute_reliability,
            robust_estimator=robust_enum,
            auto_datum=auto_datum,
        )

        # Run adjustment
        feedback.pushInfo("Running 1D leveling adjustment...")
        result = adjust_leveling_1d(net, options)

        if not result.success:
            feedback.reportError(f"Adjustment failed: {result.error_message}")
        else:
            feedback.pushInfo(f"Adjustment completed")
            feedback.pushInfo(f"Degrees of freedom: {result.degrees_of_freedom}")
            feedback.pushInfo(f"Variance factor: {result.variance_factor:.6f}")

        # Create output sinks
        try:
            crs = QgsProject.instance().crs()
            if not crs.isValid():
                crs = QgsCoordinateReferenceSystem()
        except Exception:
            crs = QgsCoordinateReferenceSystem()

        # GeoPackage layer names
        gpkg_layer_names = {
            self.OUTPUT_POINTS: "adjusted_points",
            self.OUTPUT_RESIDUALS: "residuals",
        }

        # Override sink destinations if GeoPackage output is specified
        if out_gpkg:
            for param_name, layer_name in gpkg_layer_names.items():
                parameters[param_name] = f"ogr:dbname='{out_gpkg}' table=\"{layer_name}\" (geom)"

        # Create feature sinks
        points_fields = self._create_points_fields()
        (points_sink, points_dest) = self.parameterAsSink(
            parameters, self.OUTPUT_POINTS, context,
            points_fields, QgsWkbTypes.Point, crs
        )

        residuals_fields = self._create_residuals_fields()
        (residuals_sink, residuals_dest) = self.parameterAsSink(
            parameters, self.OUTPUT_RESIDUALS, context,
            residuals_fields, QgsWkbTypes.NoGeometry, crs
        )

        # Populate adjusted points layer
        feedback.pushInfo("Creating adjusted points layer...")
        for point in result.adjusted_points.values():
            feat = QgsFeature(points_fields)
            # Use E, N for visualization if available
            feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(point.easting, point.northing)))
            feat.setAttributes([
                point.id,
                point.height,
                point.sigma_height if point.sigma_height else None,
                point.fixed_height,
            ])
            points_sink.addFeature(feat)

        # Populate residuals table
        feedback.pushInfo("Creating residuals table...")
        for res_info in result.residual_details:
            feat = QgsFeature(residuals_fields)
            # No geometry for residuals table

            # Safe value extraction
            redund = res_info.redundancy_number if res_info.redundancy_number else None
            mdb = res_info.mdb if res_info.mdb and math.isfinite(res_info.mdb) else None
            ext_rel = res_info.external_reliability if res_info.external_reliability and math.isfinite(res_info.external_reliability) else None

            feat.setAttributes([
                res_info.obs_id,
                res_info.from_point or "",
                res_info.to_point or "",
                res_info.observed,
                res_info.computed,
                res_info.residual,
                res_info.standardized_residual,
                redund,
                mdb,
                ext_rel,
                res_info.flagged or res_info.is_outlier_candidate,
            ])
            residuals_sink.addFeature(feat)

        # Add settings snapshot for debugging/support
        result.settings_snapshot = PluginSettings.get_computation_snapshot()

        # Write JSON and HTML reports
        feedback.pushInfo("Writing JSON report...")
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

        feedback.pushInfo("Writing HTML report...")
        Path(out_html).parent.mkdir(parents=True, exist_ok=True)
        save_html_report(out_html, result)

        feedback.pushInfo("Processing complete.")

        return {
            self.OUTPUT_JSON: out_json,
            self.OUTPUT_HTML: out_html,
            self.OUTPUT_GPKG: out_gpkg if out_gpkg else None,
            self.OUTPUT_POINTS: points_dest,
            self.OUTPUT_RESIDUALS: residuals_dest,
        }
