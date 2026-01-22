"""Adjust GNSS baseline network (3D) (QGIS Processing algorithm).

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
        QgsLineString,
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
    QgsLineString = object  # type: ignore
    QgsWkbTypes = object  # type: ignore
    QgsCoordinateReferenceSystem = object  # type: ignore
    QgsProject = object  # type: ignore
    QVariant = object  # type: ignore
    QGIS_AVAILABLE = False

from ..io.observations import (
    parse_gnss_baselines_csv,
    parse_gnss_points_csv,
)

from ...core.models.options import AdjustmentOptions
from ...core.models.network import Network
from ...core.solver.least_squares_3d import adjust_gnss_3d
from ...core.reports.html_report import save_html_report


class AdjustNetwork3DGnssAlgorithm(QgsProcessingAlgorithm):
    """Processing algorithm: run 3D least-squares GNSS baseline adjustment."""

    # Input parameters
    INPUT_POINTS = "POINTS_CSV"
    INPUT_BASELINES = "BASELINES_CSV"

    COVARIANCE_FORMAT = "COVARIANCE_FORMAT"
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
    OUTPUT_RESIDUAL_VECTORS = "OUTPUT_RESIDUAL_VECTORS"
    OUTPUT_RESIDUALS = "OUTPUT_RESIDUALS"

    def name(self):  # type: ignore[override]
        return "adjust_network_3d_gnss"

    def displayName(self):  # type: ignore[override]
        return "Adjust Network (3D GNSS Baselines)"

    def group(self):  # type: ignore[override]
        return "Survey Adjustment"

    def groupId(self):  # type: ignore[override]
        return "survey_adjustment"

    def shortHelpString(self):  # type: ignore[override]
        return (
            "Runs a 3D least-squares adjustment for GNSS baseline networks.\n\n"
            "Inputs:\n"
            "- Points CSV with 3D coordinates (point_id, E, N, H, fixed_e, fixed_n, fixed_h)\n"
            "- Baselines CSV (from, to, dE, dN, dH, covariance terms)\n\n"
            "Outputs:\n"
            "- JSON and HTML reports\n"
            "- Adjusted Points layer (with adjusted 3D coordinates)\n"
            "- Residual Vectors layer (lines showing baseline directions)\n"
            "- Residuals Table (with statistics)\n\n"
            "Optionally exports all layers to a GeoPackage."
        )

    def createInstance(self):  # type: ignore[override]
        return AdjustNetwork3DGnssAlgorithm()

    def initAlgorithm(self, config=None):  # type: ignore[override]
        # Input files
        self.addParameter(QgsProcessingParameterFile(
            self.INPUT_POINTS,
            "Points CSV (with 3D coordinates)",
        ))
        self.addParameter(QgsProcessingParameterFile(
            self.INPUT_BASELINES,
            "Baselines CSV",
        ))

        # Processing options
        self.addParameter(QgsProcessingParameterEnum(
            self.COVARIANCE_FORMAT,
            "Covariance format in CSV",
            options=["Full covariance (cov_EE, cov_EN, ...)", "Sigmas + correlations (sigma_E, rho_EN, ...)"],
            defaultValue=0,
        ))
        self.addParameter(QgsProcessingParameterBoolean(
            self.COMPUTE_RELIABILITY,
            "Compute reliability measures",
            defaultValue=True,
        ))
        self.addParameter(QgsProcessingParameterEnum(
            self.ROBUST_METHOD,
            "Robust estimation method",
            options=self.ROBUST_OPTIONS,
            defaultValue=0,  # None (Standard LS)
        ))
        self.addParameter(QgsProcessingParameterBoolean(
            self.AUTO_DATUM,
            "Auto-apply minimal datum constraints",
            defaultValue=False,
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
            "Adjusted Points (3D)",
            type=QgsProcessing.TypeVectorPoint,
        ))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT_RESIDUAL_VECTORS,
            "Residual Vectors",
            type=QgsProcessing.TypeVectorLine,
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
        fields.append(QgsField("E", QVariant.Double))
        fields.append(QgsField("N", QVariant.Double))
        fields.append(QgsField("H", QVariant.Double))
        fields.append(QgsField("sE", QVariant.Double))
        fields.append(QgsField("sN", QVariant.Double))
        fields.append(QgsField("sH", QVariant.Double))
        fields.append(QgsField("fixedE", QVariant.Bool))
        fields.append(QgsField("fixedN", QVariant.Bool))
        fields.append(QgsField("fixedH", QVariant.Bool))
        return fields

    def _create_residual_vectors_fields(self) -> QgsFields:
        """Create fields for residual vectors layer."""
        fields = QgsFields()
        fields.append(QgsField("obs_id", QVariant.String))
        fields.append(QgsField("from_id", QVariant.String))
        fields.append(QgsField("to_id", QVariant.String))
        fields.append(QgsField("vE", QVariant.Double))
        fields.append(QgsField("vN", QVariant.Double))
        fields.append(QgsField("vH", QVariant.Double))
        fields.append(QgsField("v_3d", QVariant.Double))
        fields.append(QgsField("w_max", QVariant.Double))
        fields.append(QgsField("is_outlier", QVariant.Bool))
        return fields

    def _create_residuals_fields(self) -> QgsFields:
        """Create fields for residuals table."""
        fields = QgsFields()
        fields.append(QgsField("obs_id", QVariant.String))
        fields.append(QgsField("from_id", QVariant.String))
        fields.append(QgsField("to_id", QVariant.String))
        fields.append(QgsField("length", QVariant.Double))
        fields.append(QgsField("v_3d", QVariant.Double))
        fields.append(QgsField("w_max", QVariant.Double))
        fields.append(QgsField("redund", QVariant.Double))
        fields.append(QgsField("is_outlier", QVariant.Bool))
        return fields

    def processAlgorithm(self, parameters, context, feedback):  # type: ignore[override]
        # Read parameters
        points_path = self.parameterAsFile(parameters, self.INPUT_POINTS, context)
        baselines_path = self.parameterAsFile(parameters, self.INPUT_BASELINES, context)

        cov_format_idx = self.parameterAsInt(parameters, self.COVARIANCE_FORMAT, context)
        cov_format = "sigmas_corr" if cov_format_idx == 1 else "full"
        compute_reliability = self.parameterAsBool(parameters, self.COMPUTE_RELIABILITY, context)
        robust_idx = self.parameterAsEnum(parameters, self.ROBUST_METHOD, context)
        robust_method = self.ROBUST_VALUES[robust_idx]
        auto_datum = self.parameterAsBool(parameters, self.AUTO_DATUM, context)

        out_json = self.parameterAsFileOutput(parameters, self.OUTPUT_JSON, context)
        out_html = self.parameterAsFileOutput(parameters, self.OUTPUT_HTML, context)
        out_gpkg = self.parameterAsFileOutput(parameters, self.OUTPUT_GPKG, context)

        # Parse input data
        feedback.pushInfo("Parsing input data...")

        points = parse_gnss_points_csv(points_path)
        baselines = parse_gnss_baselines_csv(baselines_path, covariance_format=cov_format)

        # Build network
        net = Network(name="GNSS Baseline Network", points=points)
        for obs in baselines:
            net.add_observation(obs)

        feedback.pushInfo(f"Network: {len(net.points)} points, {len(baselines)} baselines")

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
        feedback.pushInfo("Running 3D GNSS baseline adjustment...")
        result = adjust_gnss_3d(net, options)

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
            self.OUTPUT_POINTS: "adjusted_points_3d",
            self.OUTPUT_RESIDUAL_VECTORS: "residual_vectors_3d",
            self.OUTPUT_RESIDUALS: "residuals_3d",
        }

        # Override sink destinations if GeoPackage output is specified
        if out_gpkg:
            for param_name, layer_name in gpkg_layer_names.items():
                parameters[param_name] = f"ogr:dbname='{out_gpkg}' table=\"{layer_name}\" (geom)"

        # Create feature sinks
        points_fields = self._create_points_fields()
        (points_sink, points_dest) = self.parameterAsSink(
            parameters, self.OUTPUT_POINTS, context,
            points_fields, QgsWkbTypes.PointZ, crs
        )

        vectors_fields = self._create_residual_vectors_fields()
        (vectors_sink, vectors_dest) = self.parameterAsSink(
            parameters, self.OUTPUT_RESIDUAL_VECTORS, context,
            vectors_fields, QgsWkbTypes.LineString, crs
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
            # Create PointZ geometry
            h = point.height if point.height is not None else 0.0
            geom = QgsGeometry.fromPointXY(QgsPointXY(point.easting, point.northing))
            # Add Z value if possible (may need different approach for PointZ)
            feat.setGeometry(geom)
            feat.setAttributes([
                point.id,
                point.easting,
                point.northing,
                point.height,
                point.sigma_easting if point.sigma_easting else None,
                point.sigma_northing if point.sigma_northing else None,
                point.sigma_height if point.sigma_height else None,
                point.fixed_easting,
                point.fixed_northing,
                point.fixed_height,
            ])
            points_sink.addFeature(feat)

        # Populate residual vectors layer
        feedback.pushInfo("Creating residual vectors layer...")
        for res_info in result.residual_details:
            feat = QgsFeature(vectors_fields)

            # Get from/to point coordinates
            from_pt = result.adjusted_points.get(res_info.from_point)
            to_pt = result.adjusted_points.get(res_info.to_point)

            if from_pt and to_pt:
                # Create line geometry from from_point to to_point
                line = QgsLineString()
                line.addVertex(QgsPointXY(from_pt.easting, from_pt.northing))
                line.addVertex(QgsPointXY(to_pt.easting, to_pt.northing))
                feat.setGeometry(QgsGeometry(line))

            # Get component residuals if available
            vE = getattr(res_info, '_vE', None)
            vN = getattr(res_info, '_vN', None)
            vH = getattr(res_info, '_vH', None)

            feat.setAttributes([
                res_info.obs_id,
                res_info.from_point or "",
                res_info.to_point or "",
                vE,
                vN,
                vH,
                res_info.residual,
                res_info.standardized_residual,
                res_info.flagged or res_info.is_outlier_candidate,
            ])
            vectors_sink.addFeature(feat)

        # Populate residuals table
        feedback.pushInfo("Creating residuals table...")
        for res_info in result.residual_details:
            feat = QgsFeature(residuals_fields)
            # No geometry for residuals table

            redund = res_info.redundancy_number if res_info.redundancy_number else None

            feat.setAttributes([
                res_info.obs_id,
                res_info.from_point or "",
                res_info.to_point or "",
                res_info.observed,  # baseline length
                res_info.residual,  # 3D residual magnitude
                res_info.standardized_residual,
                redund,
                res_info.flagged or res_info.is_outlier_candidate,
            ])
            residuals_sink.addFeature(feat)

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
            self.OUTPUT_RESIDUAL_VECTORS: vectors_dest,
            self.OUTPUT_RESIDUALS: residuals_dest,
        }
