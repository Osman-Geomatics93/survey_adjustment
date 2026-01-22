"""Adjust survey network (2D) (QGIS Processing algorithm).

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
    parse_traverse_file,
    parse_points_csv,
    parse_distances_csv,
    parse_directions_csv,
    parse_angles_csv,
)

from ...core.models.options import AdjustmentOptions
from ...core.solver.least_squares_2d import adjust_network_2d
from ...core.reports.html_report import save_html_report
from ...core.geometry import (
    ellipse_polygon_points,
    distance_residual_vector,
    direction_residual_vector,
    angle_residual_vector,
)


class AdjustNetwork2DAlgorithm(QgsProcessingAlgorithm):
    """Processing algorithm: run 2D least-squares adjustment."""

    # Input parameters
    INPUT_TRAVERSE = "TRAVERSE_FILE"
    INPUT_POINTS = "POINTS_CSV"
    INPUT_DIST = "DISTANCES_CSV"
    INPUT_DIR = "DIRECTIONS_CSV"
    INPUT_ANGLE = "ANGLES_CSV"

    ANGLES_IN_DEGREES = "ANGLES_IN_DEGREES"
    SIGMA_ARCSECONDS = "SIGMA_ARCSECONDS"

    MAX_ITER = "MAX_ITER"
    TOL = "TOL"
    COMPUTE_RELIABILITY = "COMPUTE_RELIABILITY"
    RESIDUAL_SCALE = "RESIDUAL_SCALE"
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
    OUTPUT_VECTORS = "OUTPUT_VECTORS"
    OUTPUT_RESIDUALS = "OUTPUT_RESIDUALS"

    def name(self):  # type: ignore[override]
        return "adjust_network_2d"

    def displayName(self):  # type: ignore[override]
        return "Adjust Network (2D)"

    def group(self):  # type: ignore[override]
        return "Survey Adjustment"

    def groupId(self):  # type: ignore[override]
        return "survey_adjustment"

    def shortHelpString(self):  # type: ignore[override]
        return (
            "Runs a 2D least-squares adjustment and produces:\n"
            "- JSON and HTML reports\n"
            "- Adjusted Points layer (point geometry)\n"
            "- Error Ellipses layer (polygon geometry)\n"
            "- Residual Vectors layer (line geometry)\n"
            "- Residuals Table (no geometry)\n\n"
            "Optionally exports all layers to a GeoPackage."
        )

    def createInstance(self):  # type: ignore[override]
        return AdjustNetwork2DAlgorithm()

    def initAlgorithm(self, config=None):  # type: ignore[override]
        # Input files
        self.addParameter(QgsProcessingParameterFile(
            self.INPUT_TRAVERSE,
            "Traverse file (optional)",
            optional=True,
        ))
        self.addParameter(QgsProcessingParameterFile(
            self.INPUT_POINTS,
            "Points CSV (if no traverse file)",
            optional=True,
        ))
        self.addParameter(QgsProcessingParameterFile(
            self.INPUT_DIST,
            "Distances CSV (optional)",
            optional=True,
        ))
        self.addParameter(QgsProcessingParameterFile(
            self.INPUT_DIR,
            "Directions CSV (optional)",
            optional=True,
        ))
        self.addParameter(QgsProcessingParameterFile(
            self.INPUT_ANGLE,
            "Angles CSV (optional)",
            optional=True,
        ))

        # Processing options
        self.addParameter(QgsProcessingParameterBoolean(
            self.ANGLES_IN_DEGREES,
            "Angles/directions in degrees",
            defaultValue=True,
        ))
        self.addParameter(QgsProcessingParameterBoolean(
            self.SIGMA_ARCSECONDS,
            "Angular sigmas in arcseconds",
            defaultValue=False,
        ))

        self.addParameter(QgsProcessingParameterNumber(
            self.MAX_ITER,
            "Max iterations",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=50,
            minValue=1,
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.TOL,
            "Convergence tolerance",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=1e-6,
            minValue=0.0,
        ))
        self.addParameter(QgsProcessingParameterBoolean(
            self.COMPUTE_RELIABILITY,
            "Compute reliability measures",
            defaultValue=True,
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.RESIDUAL_SCALE,
            "Residual vector scale factor",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=1000.0,
            minValue=1.0,
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
            "Adjusted Points",
            type=QgsProcessing.TypeVectorPoint,
        ))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT_ELLIPSES,
            "Error Ellipses",
            type=QgsProcessing.TypeVectorPolygon,
        ))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT_VECTORS,
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
        fields.append(QgsField("sE", QVariant.Double))
        fields.append(QgsField("sN", QVariant.Double))
        fields.append(QgsField("fixedE", QVariant.Bool))
        fields.append(QgsField("fixedN", QVariant.Bool))
        return fields

    def _create_ellipse_fields(self) -> QgsFields:
        """Create fields for error ellipses layer."""
        fields = QgsFields()
        fields.append(QgsField("point_id", QVariant.String))
        fields.append(QgsField("a", QVariant.Double))  # semi-major
        fields.append(QgsField("b", QVariant.Double))  # semi-minor
        fields.append(QgsField("theta", QVariant.Double))  # orientation (rad)
        fields.append(QgsField("theta_deg", QVariant.Double))  # orientation (deg)
        fields.append(QgsField("confidence", QVariant.Double))
        return fields

    def _create_vector_fields(self) -> QgsFields:
        """Create fields for residual vectors layer."""
        fields = QgsFields()
        fields.append(QgsField("obs_id", QVariant.String))
        fields.append(QgsField("type", QVariant.String))
        fields.append(QgsField("from_id", QVariant.String))
        fields.append(QgsField("to_id", QVariant.String))
        fields.append(QgsField("v", QVariant.Double))  # residual
        fields.append(QgsField("w", QVariant.Double))  # standardized residual
        fields.append(QgsField("redund", QVariant.Double))  # redundancy number
        fields.append(QgsField("mdb", QVariant.Double))
        fields.append(QgsField("is_outlier", QVariant.Bool))
        return fields

    def _create_residuals_fields(self) -> QgsFields:
        """Create fields for residuals table (no geometry)."""
        fields = QgsFields()
        fields.append(QgsField("obs_id", QVariant.String))
        fields.append(QgsField("type", QVariant.String))
        fields.append(QgsField("from_id", QVariant.String))
        fields.append(QgsField("to_id", QVariant.String))
        fields.append(QgsField("at_id", QVariant.String))
        fields.append(QgsField("observed", QVariant.Double))
        fields.append(QgsField("computed", QVariant.Double))
        fields.append(QgsField("sigma", QVariant.Double))
        fields.append(QgsField("v", QVariant.Double))
        fields.append(QgsField("w", QVariant.Double))
        fields.append(QgsField("redund", QVariant.Double))
        fields.append(QgsField("mdb", QVariant.Double))
        fields.append(QgsField("is_outlier", QVariant.Bool))
        return fields

    def processAlgorithm(self, parameters, context, feedback):  # type: ignore[override]
        # Read parameters
        traverse_path = self.parameterAsFile(parameters, self.INPUT_TRAVERSE, context)
        points_path = self.parameterAsFile(parameters, self.INPUT_POINTS, context)
        dist_path = self.parameterAsFile(parameters, self.INPUT_DIST, context)
        dir_path = self.parameterAsFile(parameters, self.INPUT_DIR, context)
        angle_path = self.parameterAsFile(parameters, self.INPUT_ANGLE, context)

        angles_in_degrees = self.parameterAsBool(parameters, self.ANGLES_IN_DEGREES, context)
        sigma_arcseconds = self.parameterAsBool(parameters, self.SIGMA_ARCSECONDS, context)

        max_iter = int(self.parameterAsInt(parameters, self.MAX_ITER, context))
        tol = float(self.parameterAsDouble(parameters, self.TOL, context))
        compute_reliability = self.parameterAsBool(parameters, self.COMPUTE_RELIABILITY, context)
        residual_scale = float(self.parameterAsDouble(parameters, self.RESIDUAL_SCALE, context))
        robust_idx = self.parameterAsEnum(parameters, self.ROBUST_METHOD, context)
        robust_method = self.ROBUST_VALUES[robust_idx]
        auto_datum = self.parameterAsBool(parameters, self.AUTO_DATUM, context)

        out_json = self.parameterAsFileOutput(parameters, self.OUTPUT_JSON, context)
        out_html = self.parameterAsFileOutput(parameters, self.OUTPUT_HTML, context)
        out_gpkg = self.parameterAsFileOutput(parameters, self.OUTPUT_GPKG, context)

        if not traverse_path and not points_path:
            raise QgsProcessingException("Provide a traverse file or a points CSV.")

        # Parse network
        feedback.pushInfo("Parsing input data...")
        if traverse_path:
            net = parse_traverse_file(
                traverse_path,
                angle_unit="degrees" if angles_in_degrees else "radians",
                direction_unit="degrees" if angles_in_degrees else "radians",
                sigma_angle_unit="arcseconds" if sigma_arcseconds else "radians",
                sigma_direction_unit="arcseconds" if sigma_arcseconds else "radians",
            )
        else:
            from ...core.models.network import Network

            points = parse_points_csv(points_path)
            net = Network(points=points)
            if dist_path:
                for o in parse_distances_csv(dist_path):
                    net.add_observation(o)
            if dir_path:
                for o in parse_directions_csv(
                    dir_path,
                    direction_unit="degrees" if angles_in_degrees else "radians",
                    sigma_unit="arcseconds" if sigma_arcseconds else "radians",
                ):
                    net.add_observation(o)
            if angle_path:
                for o in parse_angles_csv(
                    angle_path,
                    angle_unit="degrees" if angles_in_degrees else "radians",
                    sigma_unit="arcseconds" if sigma_arcseconds else "radians",
                ):
                    net.add_observation(o)

        feedback.pushInfo(f"Network: {len(net.points)} points, {len(net.observations)} observations")

        # Run adjustment
        from ...core.models.options import RobustEstimator
        robust_enum = None
        if robust_method == "huber":
            robust_enum = RobustEstimator.HUBER
        elif robust_method == "danish":
            robust_enum = RobustEstimator.DANISH
        elif robust_method == "igg3":
            robust_enum = RobustEstimator.IGG3

        options = AdjustmentOptions(
            max_iterations=max_iter,
            convergence_threshold=tol,
            compute_reliability=compute_reliability,
            robust_estimator=robust_enum,
            auto_datum=auto_datum,
        )

        feedback.pushInfo("Running least-squares adjustment...")
        result = adjust_network_2d(net, options)

        if not result.success:
            feedback.reportError(f"Adjustment failed: {result.error_message}")
        else:
            feedback.pushInfo(f"Adjustment converged in {result.iterations} iterations")
            feedback.pushInfo(f"Degrees of freedom: {result.degrees_of_freedom}")
            feedback.pushInfo(f"Variance factor: {result.variance_factor:.6f}")

        # Create output sinks
        # Determine output CRS (use project CRS or a default)
        try:
            crs = QgsProject.instance().crs()
            if not crs.isValid():
                crs = QgsCoordinateReferenceSystem()
        except Exception:
            crs = QgsCoordinateReferenceSystem()

        # Build output destinations for GeoPackage if specified
        gpkg_layer_names = {
            self.OUTPUT_POINTS: "adjusted_points",
            self.OUTPUT_ELLIPSES: "error_ellipses",
            self.OUTPUT_VECTORS: "residual_vectors",
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

        ellipse_fields = self._create_ellipse_fields()
        (ellipse_sink, ellipse_dest) = self.parameterAsSink(
            parameters, self.OUTPUT_ELLIPSES, context,
            ellipse_fields, QgsWkbTypes.Polygon, crs
        )

        vector_fields = self._create_vector_fields()
        (vector_sink, vector_dest) = self.parameterAsSink(
            parameters, self.OUTPUT_VECTORS, context,
            vector_fields, QgsWkbTypes.LineString, crs
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
            feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(point.easting, point.northing)))
            feat.setAttributes([
                point.id,
                point.easting,
                point.northing,
                point.sigma_easting if point.sigma_easting else None,
                point.sigma_northing if point.sigma_northing else None,
                point.fixed_easting,
                point.fixed_northing,
            ])
            points_sink.addFeature(feat)

        # Populate error ellipses layer
        feedback.pushInfo("Creating error ellipses layer...")
        for ellipse in result.error_ellipses.values():
            point = result.adjusted_points.get(ellipse.point_id)
            if not point:
                continue

            # Generate polygon vertices
            vertices = ellipse_polygon_points(
                center_e=point.easting,
                center_n=point.northing,
                semi_major=ellipse.semi_major,
                semi_minor=ellipse.semi_minor,
                orientation=ellipse.orientation,
                num_vertices=64,
            )

            # Create polygon geometry
            ring = [QgsPointXY(e, n) for e, n in vertices]
            geom = QgsGeometry.fromPolygonXY([ring])

            feat = QgsFeature(ellipse_fields)
            feat.setGeometry(geom)
            feat.setAttributes([
                ellipse.point_id,
                ellipse.semi_major,
                ellipse.semi_minor,
                ellipse.orientation,
                ellipse.orientation_degrees,
                ellipse.confidence_level,
            ])
            ellipse_sink.addFeature(feat)

        # Populate residual vectors layer
        feedback.pushInfo("Creating residual vectors layer...")
        # Build observation lookup for geometry
        obs_lookup = {o.id: o for o in net.observations}

        for res_info in result.residual_details:
            obs = obs_lookup.get(res_info.obs_id)
            if not obs:
                continue

            # Get geometry based on observation type
            line_coords = None
            from_id = ""
            to_id = ""

            if res_info.obs_type == "distance":
                from_pt = result.adjusted_points.get(res_info.from_point)
                to_pt = result.adjusted_points.get(res_info.to_point)
                if from_pt and to_pt:
                    line_coords = distance_residual_vector(
                        from_e=from_pt.easting,
                        from_n=from_pt.northing,
                        to_e=to_pt.easting,
                        to_n=to_pt.northing,
                        residual=res_info.residual,
                        scale=residual_scale,
                    )
                    from_id = res_info.from_point or ""
                    to_id = res_info.to_point or ""

            elif res_info.obs_type == "direction":
                from_pt = result.adjusted_points.get(res_info.from_point)
                to_pt = result.adjusted_points.get(res_info.to_point)
                if from_pt and to_pt:
                    line_coords = direction_residual_vector(
                        station_e=from_pt.easting,
                        station_n=from_pt.northing,
                        target_e=to_pt.easting,
                        target_n=to_pt.northing,
                        residual=res_info.residual,
                        scale=residual_scale,
                    )
                    from_id = res_info.from_point or ""
                    to_id = res_info.to_point or ""

            elif res_info.obs_type == "angle":
                at_pt = result.adjusted_points.get(res_info.at_point)
                from_pt = result.adjusted_points.get(res_info.from_point)
                to_pt = result.adjusted_points.get(res_info.to_point)
                if at_pt and from_pt and to_pt:
                    line_coords = angle_residual_vector(
                        at_e=at_pt.easting,
                        at_n=at_pt.northing,
                        from_e=from_pt.easting,
                        from_n=from_pt.northing,
                        to_e=to_pt.easting,
                        to_n=to_pt.northing,
                        residual=res_info.residual,
                        scale=residual_scale,
                    )
                    from_id = res_info.at_point or ""
                    to_id = res_info.to_point or ""

            if line_coords:
                (start, end) = line_coords
                geom = QgsGeometry.fromPolylineXY([
                    QgsPointXY(start[0], start[1]),
                    QgsPointXY(end[0], end[1]),
                ])

                feat = QgsFeature(vector_fields)
                feat.setGeometry(geom)

                # Safe value extraction
                redund = res_info.redundancy_number if res_info.redundancy_number else None
                mdb = res_info.mdb if res_info.mdb and math.isfinite(res_info.mdb) else None

                feat.setAttributes([
                    res_info.obs_id,
                    res_info.obs_type,
                    from_id,
                    to_id,
                    res_info.residual,
                    res_info.standardized_residual,
                    redund,
                    mdb,
                    res_info.flagged or res_info.is_outlier_candidate,
                ])
                vector_sink.addFeature(feat)

        # Populate residuals table (no geometry)
        feedback.pushInfo("Creating residuals table...")
        for res_info in result.residual_details:
            obs = obs_lookup.get(res_info.obs_id)

            feat = QgsFeature(residuals_fields)
            # No geometry for this table

            # Safe value extraction
            redund = res_info.redundancy_number if res_info.redundancy_number else None
            mdb = res_info.mdb if res_info.mdb and math.isfinite(res_info.mdb) else None
            sigma = obs.sigma if obs else None

            feat.setAttributes([
                res_info.obs_id,
                res_info.obs_type,
                res_info.from_point or "",
                res_info.to_point or "",
                res_info.at_point or "",
                res_info.observed,
                res_info.computed,
                sigma,
                res_info.residual,
                res_info.standardized_residual,
                redund,
                mdb,
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
            self.OUTPUT_ELLIPSES: ellipse_dest,
            self.OUTPUT_VECTORS: vector_dest,
            self.OUTPUT_RESIDUALS: residuals_dest,
        }
