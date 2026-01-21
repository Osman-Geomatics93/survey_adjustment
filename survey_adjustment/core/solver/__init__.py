"""survey_adjustment.core.solver

Pure-Python least-squares solvers (no QGIS imports).
"""

from .least_squares_2d import adjust_network_2d

__all__ = [
    "adjust_network_2d",
]
