"""Adjust survey network (2D) (QGIS Processing algorithm).

This module is intended to run inside QGIS. It remains importable outside QGIS
via guarded imports, but execution requires QGIS.
"""

from __future__ import annotations

import json
from pathlib import Path

try:  # pragma: no cover
    from qgis.core import (
        QgsProcessingAlgorithm,
        QgsProcessingParameterFile,
        QgsProcessingParameterFileDestination,
        QgsProcessingParameterNumber,
        QgsProcessingParameterBoolean,
        QgsProcessingException,
    )
except Exception:  # pragma: no cover
    QgsProcessingAlgorithm = object  # type: ignore
    QgsProcessingParameterFile = object  # type: ignore
    QgsProcessingParameterFileDestination = object  # type: ignore
    QgsProcessingParameterNumber = object  # type: ignore
    QgsProcessingParameterBoolean = object  # type: ignore
    QgsProcessingException = RuntimeError

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


class AdjustNetwork2DAlgorithm(QgsProcessingAlgorithm):
    """Processing algorithm: run 2D least-squares adjustment."""

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

    OUTPUT_JSON = "OUTPUT_JSON"
    OUTPUT_HTML = "OUTPUT_HTML"

    def name(self):  # type: ignore[override]
        return "adjust_network_2d"

    def displayName(self):  # type: ignore[override]
        return "Adjust Network (2D)"

    def group(self):  # type: ignore[override]
        return "Survey Adjustment"

    def groupId(self):  # type: ignore[override]
        return "survey_adjustment"

    def shortHelpString(self):  # type: ignore[override]
        return "Runs a 2D least-squares adjustment and exports JSON and HTML reports."

    def initAlgorithm(self, config=None):  # type: ignore[override]
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

    def processAlgorithm(self, parameters, context, feedback):  # type: ignore[override]
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

        out_json = self.parameterAsFileOutput(parameters, self.OUTPUT_JSON, context)
        out_html = self.parameterAsFileOutput(parameters, self.OUTPUT_HTML, context)

        if not traverse_path and not points_path:
            raise QgsProcessingException("Provide a traverse file or a points CSV.")

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

        options = AdjustmentOptions(
            max_iterations=max_iter,
            convergence_threshold=tol,
            compute_reliability=compute_reliability,
        )

        result = adjust_network_2d(net, options)

        # Write outputs
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

        Path(out_html).parent.mkdir(parents=True, exist_ok=True)
        save_html_report(out_html, result)

        return {self.OUTPUT_JSON: out_json, self.OUTPUT_HTML: out_html}
