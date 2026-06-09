# Survey Adjustment & Network Analysis

A free, open-source **least-squares adjustment** engine for survey networks,
delivered as a [QGIS](https://qgis.org) Processing provider. It turns QGIS into a
geodetic computation platform — from a single leveling run to a mixed
GNSS/classical network — and returns results as GIS layers, JSON, and HTML
reports.

The numerical core depends only on **NumPy** and can be used as a standalone
Python library, independent of QGIS.

## What it does

| Adjustment type | Observations |
|:----------------|:-------------|
| **2D classical** | distances, directions, angles |
| **1D leveling** | height differences |
| **3D GNSS** | baseline vectors with full 3×3 covariance |
| **Mixed** | any combination of the above |

Beyond the solution, every adjustment reports the quality-control diagnostics
expected of professional software: a chi-square global test, standardized
residuals, redundancy numbers, Baarda internal and external reliability (MDB),
error ellipses, and robust estimation (Huber, Danish, IGG-III).

## Where to start

<div class="grid cards" markdown>

- :material-download: **[Installation](installation.md)** — install in QGIS or use the core as a Python library.
- :material-table: **[Input formats](input-formats.md)** — the CSV schemas for points and every observation type.
- :material-school: **[Tutorials](tutorials/2d-network.md)** — worked walkthroughs for each solver.
- :material-function-variant: **[Theory & QC](theory.md)** — the mathematical model and statistics.

</div>

## Citing

If you use this software in research or professional work, please cite it. See
the [`CITATION.cff`](https://github.com/Osman-Geomatics93/survey_adjustment/blob/master/CITATION.cff)
in the repository, or use the "Cite this repository" button on GitHub.

!!! note "Documentation in progress"
    This documentation site is being built out alongside the project. Pages are
    added and expanded iteratively — see the
    [roadmap](https://github.com/Osman-Geomatics93/survey_adjustment/blob/master/ROADMAP.md).
    Contributions are welcome.
