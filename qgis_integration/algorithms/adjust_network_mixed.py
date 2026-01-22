"""Adjust Mixed Network (Classical + GNSS + Leveling) (QGIS Processing algorithm).

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
        QgsPoint,
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
    QgsPoint = object  # type: ignore
    QgsPointXY = object  # type: ignore
    QgsLineString = object  # type: ignore
    QgsWkbTypes = object  # type: ignore
    QgsCoordinateReferenceSystem = object  # type: ignore
    QgsProject = object  # type: ignore
    QVariant = object  # type: ignore
    QGIS_AVAILABLE = False

from ..io.observations import (
    parse_points_csv,
    parse_distances_csv,
    parse_directions_csv,
    parse_angles_csv,
    parse_gnss_baselines_csv,
    parse_gnss_points_csv,
    parse_leveling_csv,
)
from ..settings import PluginSettings

from ...core.models.options import AdjustmentOptions
from ...core.models.network import Network
from ...core.solver.least_squares_mixed import adjust_network_mixed
from ...core.reports.html_report import save_html_report
from ...core.geometry import ellipse_polygon_points


class AdjustNetworkMixedAlgorithm(QgsProcessingAlgorithm):
    """Processing algorithm: run mixed least-squares adjustment (classical + GNSS + leveling)."""

    # Input parameters
    INPUT_POINTS = "POINTS_CSV"
    INPUT_DISTANCES = "DISTANCES_CSV"
    INPUT_DIRECTIONS = "DIRECTIONS_CSV"
    INPUT_ANGLES = "ANGLES_CSV"
    INPUT_GNSS_BASELINES = "GNSS_BASELINES_CSV"
    INPUT_LEVELING = "LEVELING_CSV"

    COMPUTE_RELIABILITY = "COMPUTE_RELIABILITY"
    MAX_ITERATIONS = "MAX_ITERATIONS"
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
    OUTPUT_ELLIPSES = "OUTPUT_ELLIPSES"
    OUTPUT_RESIDUAL_VECTORS = "OUTPUT_RESIDUAL_VECTORS"
    OUTPUT_RESIDUALS = "OUTPUT_RESIDUALS"

    def name(self):  # type: ignore[override]
        return "adjust_network_mixed"

    def displayName(self):  # type: ignore[override]
        return "Adjust Network (Mixed: Classical + GNSS + Leveling)"

    def group(self):  # type: ignore[override]
        return "Survey Adjustment"

    def groupId(self):  # type: ignore[override]
        return "survey_adjustment"

    def shortHelpString(self):  # type: ignore[override]
        return (
            "Runs a mixed least-squares adjustment combining classical, GNSS, and leveling observations.\n\n"
            "Inputs:\n"
            "- Points CSV with 3D coordinates (point_id, E, N, H, fixed_e, fixed_n, fixed_h)\n"
            "- Distances CSV (optional)\n"
            "- Directions CSV (optional)\n"
            "- Angles CSV (optional)\n"
            "- GNSS Baselines CSV (optional)\n"
            "- Leveling CSV (optional)\n\n"
            "At least one observation file (classical, GNSS, or leveling) must be provided.\n\n"
            "Outputs:\n"
            "- JSON and HTML reports\n"
            "- Adjusted Points layer (with E, N, H and sigmas)\n"
            "- Error Ellipses layer (2D horizontal)\n"
            "- Residual Vectors layer\n"
            "- Residuals Table\n\n"
            "Optionally exports all layers to a GeoPackage.\n\n"
            "Note: Default values come from Plugins → Survey Adjustment → Settings. "
            "QGIS may remember your last-used values per tool."
        )

    def createInstance(self):  # type: ignore[override]
        return AdjustNetworkMixedAlgorithm()

    def initAlgorithm(self, config=None):  # type: ignore[override]
        # Input files
        self.addParameter(QgsProcessingParameterFile(
            self.INPUT_POINTS,
            "Points CSV (with 3D coordinates)",
        ))
        self.addParameter(QgsProcessingParameterFile(
            self.INPUT_DISTANCES,
            "Distances CSV (optional)",
            optional=True,
        ))
        self.addParameter(QgsProcessingParameterFile(
            self.INPUT_DIRECTIONS,
            "Directions CSV (optional)",
            optional=True,
        ))
        self.addParameter(QgsProcessingParameterFile(
            self.INPUT_ANGLES,
            "Angles CSV (optional)",
            optional=True,
        ))
        self.addParameter(QgsProcessingParameterFile(
            self.INPUT_GNSS_BASELINES,
            "GNSS Baselines CSV (optional)",
            optional=True,
        ))
        self.addParameter(QgsProcessingParameterFile(
            self.INPUT_LEVELING,
            "Leveling CSV (optional)",
            optional=True,
        ))

        # Processing options - defaults from PluginSettings
        self.addParameter(QgsProcessingParameterBoolean(
            self.COMPUTE_RELIABILITY,
            "Compute reliability measures",
            defaultValue=PluginSettings.get("compute_reliability"),
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.MAX_ITERATIONS,
            "Maximum iterations",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=PluginSettings.get("max_iterations"),
            minValue=1,
            maxValue=100,
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
            "Adjusted Points (3D)",
            type=QgsProcessing.TypeVectorPoint,
        ))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT_ELLIPSES,
            "Error Ellipses",
            type=QgsProcessing.TypeVectorPolygon,
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

    def _create_ellipse_fields(self) -> QgsFields:
        """Create fields for error ellipses layer."""
        fields = QgsFields()
        fields.append(QgsField("point_id", QVariant.String))
        fields.append(QgsField("semi_major", QVariant.Double))
        fields.append(QgsField("semi_minor", QVariant.Double))
        fields.append(QgsField("orientation", QVariant.Double))
        fields.append(QgsField("confidence", QVariant.Double))
        return fields

    def _create_residual_vectors_fields(self) -> QgsFields:
        """Create fields for residual vectors layer."""
        fields = QgsFields()
        fields.append(QgsField("obs_id", QVariant.String))
        fields.append(QgsField("obs_type", QVariant.String))
        fields.append(QgsField("from_id", QVariant.String))
        fields.append(QgsField("to_id", QVariant.String))
        fields.append(QgsField("residual", QVariant.Double))
        fields.append(QgsField("w", QVariant.Double))
        fields.append(QgsField("is_outlier", QVariant.Bool))
        return fields

    def _create_residuals_fields(self) -> QgsFields:
        """Create fields for residuals table."""
        fields = QgsFields()
        fields.append(QgsField("obs_id", QVariant.String))
        fields.append(QgsField("obs_type", QVariant.String))
        fields.append(QgsField("from_id", QVariant.String))
        fields.append(QgsField("to_id", QVariant.String))
        fields.append(QgsField("observed", QVariant.Double))
        fields.append(QgsField("computed", QVariant.Double))
        fields.append(QgsField("residual", QVariant.Double))
        fields.append(QgsField("w", QVariant.Double))
        fields.append(QgsField("redund", QVariant.Double))
        fields.append(QgsField("is_outlier", QVariant.Bool))
        return fields

    def processAlgorithm(self, parameters, context, feedback):  # type: ignore[override]
        # Read parameters
        points_path = self.parameterAsFile(parameters, self.INPUT_POINTS, context)
        distances_path = self.parameterAsFile(parameters, self.INPUT_DISTANCES, context)
        directions_path = self.parameterAsFile(parameters, self.INPUT_DIRECTIONS, context)
        angles_path = self.parameterAsFile(parameters, self.INPUT_ANGLES, context)
        gnss_path = self.parameterAsFile(parameters, self.INPUT_GNSS_BASELINES, context)
        leveling_path = self.parameterAsFile(parameters, self.INPUT_LEVELING, context)

        compute_reliability = self.parameterAsBool(parameters, self.COMPUTE_RELIABILITY, context)
        max_iterations = self.parameterAsInt(parameters, self.MAX_ITERATIONS, context)
        robust_idx = self.parameterAsEnum(parameters, self.ROBUST_METHOD, context)
        robust_method = self.ROBUST_VALUES[robust_idx]
        auto_datum = self.parameterAsBool(parameters, self.AUTO_DATUM, context)

        out_json = self.parameterAsFileOutput(parameters, self.OUTPUT_JSON, context)
        out_html = self.parameterAsFileOutput(parameters, self.OUTPUT_HTML, context)
        out_gpkg = self.parameterAsFileOutput(parameters, self.OUTPUT_GPKG, context)

        # Check that at least one observation file is provided
        has_classical = distances_path or directions_path or angles_path
        has_gnss = gnss_path
        has_leveling = leveling_path

        if not has_classical and not has_gnss and not has_leveling:
            raise QgsProcessingException(
                "At least one observation file (classical, GNSS, or leveling) must be provided"
            )

        # Parse input data
        feedback.pushInfo("Parsing input data...")

        # Parse points (use GNSS parser for 3D if GNSS or leveling present, otherwise 2D parser)
        if has_gnss or has_leveling:
            points = parse_gnss_points_csv(points_path)
        else:
            points = parse_points_csv(points_path)

        # Build network
        net = Network(name="Mixed Network", points=points)

        # Parse classical observations
        if distances_path:
            feedback.pushInfo(f"Parsing distances from {distances_path}")
            for obs in parse_distances_csv(distances_path):
                net.add_observation(obs)

        if directions_path:
            feedback.pushInfo(f"Parsing directions from {directions_path}")
            for obs in parse_directions_csv(directions_path):
                net.add_observation(obs)

        if angles_path:
            feedback.pushInfo(f"Parsing angles from {angles_path}")
            for obs in parse_angles_csv(angles_path):
                net.add_observation(obs)

        # Parse GNSS baselines
        if gnss_path:
            feedback.pushInfo(f"Parsing GNSS baselines from {gnss_path}")
            for obs in parse_gnss_baselines_csv(gnss_path, covariance_format="full"):
                net.add_observation(obs)

        # Parse leveling observations
        if leveling_path:
            feedback.pushInfo(f"Parsing leveling observations from {leveling_path}")
            for obs in parse_leveling_csv(leveling_path):
                net.add_observation(obs)

        feedback.pushInfo(f"Network: {len(net.points)} points, {len(net.observations)} observations")

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
            max_iterations=max_iterations,
            compute_reliability=compute_reliability,
            robust_estimator=robust_enum,
            auto_datum=auto_datum,
        )

        # Run adjustment
        feedback.pushInfo("Running mixed adjustment...")
        result = adjust_network_mixed(net, options)

        if not result.success:
            feedback.reportError(f"Adjustment failed: {result.error_message}")
        else:
            feedback.pushInfo(f"Adjustment completed")
            feedback.pushInfo(f"Converged: {result.converged} in {result.iterations} iterations")
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
            self.OUTPUT_POINTS: "adjusted_points_mixed",
            self.OUTPUT_ELLIPSES: "error_ellipses_mixed",
            self.OUTPUT_RESIDUAL_VECTORS: "residual_vectors_mixed",
            self.OUTPUT_RESIDUALS: "residuals_mixed",
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

        ellipse_fields = self._create_ellipse_fields()
        (ellipse_sink, ellipse_dest) = self.parameterAsSink(
            parameters, self.OUTPUT_ELLIPSES, context,
            ellipse_fields, QgsWkbTypes.Polygon, crs
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
            h = point.height if point.height is not None else 0.0
            geom = QgsGeometry.fromPointXY(QgsPointXY(point.easting, point.northing))
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

        # Populate error ellipses layer
        feedback.pushInfo("Creating error ellipses layer...")
        for pid, ellipse in result.error_ellipses.items():
            point = result.adjusted_points.get(pid)
            if not point:
                continue

            feat = QgsFeature(ellipse_fields)

            # Generate ellipse polygon
            try:
                poly_pts = ellipse_polygon_points(
                    center_e=point.easting,
                    center_n=point.northing,
                    semi_major=ellipse.semi_major,
                    semi_minor=ellipse.semi_minor,
                    orientation=ellipse.orientation,
                    num_vertices=64,
                )
                qgs_pts = [QgsPointXY(e, n) for e, n in poly_pts]
                feat.setGeometry(QgsGeometry.fromPolygonXY([qgs_pts]))
            except Exception:
                continue

            feat.setAttributes([
                pid,
                ellipse.semi_major,
                ellipse.semi_minor,
                math.degrees(ellipse.orientation),
                ellipse.confidence_level,
            ])
            ellipse_sink.addFeature(feat)

        # Populate residual vectors layer
        feedback.pushInfo("Creating residual vectors layer...")
        for res_info in result.residual_details:
            feat = QgsFeature(vectors_fields)

            # Get from/to point coordinates
            from_id = res_info.from_point or res_info.at_point
            to_id = res_info.to_point

            from_pt = result.adjusted_points.get(from_id) if from_id else None
            to_pt = result.adjusted_points.get(to_id) if to_id else None

            if from_pt and to_pt:
                # Create line geometry
                line = QgsLineString()
                line.addVertex(QgsPoint(from_pt.easting, from_pt.northing))
                line.addVertex(QgsPoint(to_pt.easting, to_pt.northing))
                feat.setGeometry(QgsGeometry(line))

            feat.setAttributes([
                res_info.obs_id,
                res_info.obs_type,
                from_id or "",
                to_id or "",
                res_info.residual,
                res_info.standardized_residual,
                res_info.flagged or res_info.is_outlier_candidate,
            ])
            vectors_sink.addFeature(feat)

        # Populate residuals table
        feedback.pushInfo("Creating residuals table...")
        for res_info in result.residual_details:
            feat = QgsFeature(residuals_fields)

            from_id = res_info.from_point or res_info.at_point
            to_id = res_info.to_point

            feat.setAttributes([
                res_info.obs_id,
                res_info.obs_type,
                from_id or "",
                to_id or "",
                res_info.observed,
                res_info.computed,
                res_info.residual,
                res_info.standardized_residual,
                res_info.redundancy_number,
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
            self.OUTPUT_ELLIPSES: ellipse_dest,
            self.OUTPUT_RESIDUAL_VECTORS: vectors_dest,
            self.OUTPUT_RESIDUALS: residuals_dest,
        }
