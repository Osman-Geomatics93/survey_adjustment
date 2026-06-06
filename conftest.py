"""Pytest bootstrap for the flattened QGIS-plugin layout.

A QGIS plugin is installed as a folder named ``survey_adjustment`` that contains
``__init__.py`` directly, so in the published repository the repo root *is* the
``survey_adjustment`` package. The test modules import ``survey_adjustment.core...``,
which would otherwise only resolve when the checkout directory happens to be named
``survey_adjustment``.

To make the tests run regardless of the checkout directory name (and without
requiring QGIS — the package ``__init__`` keeps all QGIS imports lazy), we register
the repo root under the canonical package name before collection begins.
"""

import importlib.util
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))

if "survey_adjustment" not in sys.modules:
    _init = os.path.join(_ROOT, "__init__.py")
    _spec = importlib.util.spec_from_file_location(
        "survey_adjustment",
        _init,
        submodule_search_locations=[_ROOT],
    )
    _module = importlib.util.module_from_spec(_spec)
    sys.modules["survey_adjustment"] = _module
    _spec.loader.exec_module(_module)
