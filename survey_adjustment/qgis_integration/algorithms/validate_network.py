"""Validate survey network inputs (QGIS Processing algorithm).

This module is only intended to run inside QGIS. It must remain importable
outside QGIS, so QGIS imports are guarded.
"""

from __future__ import annotations

import json
from pathlib import Path

try:  # pragma: no cover
    from qgis.core import (
        QgsProcessingAlgorithm,
        QgsProcessingParameterFile,
        QgsProcessingParameterBoolean,
        QgsProcessingParameterFileDestination,
        QgsProcessingException,
    )
except Exception:  # pragma: no cover
    QgsProcessingAlgorithm = object  # type: ignore
    QgsProcessingParameterFile = object  # type: ignore
    QgsProcessingParameterBoolean = object  # type: ignore
    QgsProcessingParameterFileDestination = object  # type: ignore
    QgsProcessingException = RuntimeError

from ..io.observations import (
    parse_traverse_file,
    parse_points_csv,
    parse_distances_csv,
    parse_directions_csv,
    parse_angles_csv,
)

from ...core.solver.indexing import build_parameter_index, validate_network_for_adjustment


class ValidateNetworkAlgorithm(QgsProcessingAlgorithm):
    """Processing algorithm: validate observation files and network constraints."""

    INPUT_TRAVERSE = "TRAVERSE_FILE"
    INPUT_POINTS = "POINTS_CSV"
    INPUT_DIST = "DISTANCES_CSV"
    INPUT_DIR = "DIRECTIONS_CSV"
    INPUT_ANGLE = "ANGLES_CSV"

    ANGLES_IN_DEGREES = "ANGLES_IN_DEGREES"
    SIGMA_ARCSECONDS = "SIGMA_ARCSECONDS"

    OUTPUT_JSON = "OUTPUT_JSON"

    def name(self):  # type: ignore[override]
        return "validate_network"

    def displayName(self):  # type: ignore[override]
        return "Validate Survey Network"

    def group(self):  # type: ignore[override]
        return "Survey Adjustment"

    def groupId(self):  # type: ignore[override]
        return "survey_adjustment"

    def shortHelpString(self):  # type: ignore[override]
        return "Validates survey network inputs and datum constraints for least-squares adjustment."

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
        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT_JSON,
            "Validation report (JSON)",
            fileFilter="JSON (*.json)",
        ))

    def processAlgorithm(self, parameters, context, feedback):  # type: ignore[override]
        traverse_path = self.parameterAsFile(parameters, self.INPUT_TRAVERSE, context)
        points_path = self.parameterAsFile(parameters, self.INPUT_POINTS, context)
        dist_path = self.parameterAsFile(parameters, self.INPUT_DIST, context)
        dir_path = self.parameterAsFile(parameters, self.INPUT_DIR, context)
        angle_path = self.parameterAsFile(parameters, self.INPUT_ANGLE, context)

        angles_in_degrees = self.parameterAsBool(parameters, self.ANGLES_IN_DEGREES, context)
        sigma_arcseconds = self.parameterAsBool(parameters, self.SIGMA_ARCSECONDS, context)

        out_json = self.parameterAsFileOutput(parameters, self.OUTPUT_JSON, context)

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
            points = parse_points_csv(points_path)
            net = None
            from ...core.models.network import Network

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

        errors = list(net.validate())
        idx = build_parameter_index(net)
        errors.extend(validate_network_for_adjustment(net, idx))

        report = {
            "success": len(errors) == 0,
            "num_points": len(net.points),
            "num_observations": len(net.get_enabled_observations()),
            "errors": errors,
        }

        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return {self.OUTPUT_JSON: out_json}

