# API reference

The numerical core is a QGIS-independent Python package that depends only on
NumPy. It can be imported and scripted directly — for automated pipelines,
testing, and reproducible research — without launching QGIS.

## Package layout

```
core/
├── models/        # Point, Network, Observation, options
├── solver/        # Least-squares engines: 1D, 2D, 3D GNSS, mixed; robust
├── statistics/    # Distributions, global/local tests, reliability
├── geometry/      # Error ellipses, residual vectors
├── results/       # Result structures and schemas
├── reports/       # HTML report generation
└── validation/    # Constraint-health checks
```

## Key modules

| Module | Responsibility |
|:-------|:---------------|
| `core.models.point` | Point with coordinates, fix flags, and prior sigmas |
| `core.models.network` | Network container of points and observations |
| `core.models.observation` | Observation types (distance, direction, angle, dh, baseline) |
| `core.solver.least_squares_1d` | Leveling adjustment |
| `core.solver.least_squares_2d` | Classical 2D adjustment |
| `core.solver.least_squares_3d` | GNSS baseline adjustment |
| `core.solver.least_squares_mixed` | Unified mixed adjustment |
| `core.solver.robust` | IRLS weight functions (Huber, Danish, IGG-III) |
| `core.statistics.distributions` | Chi-square / normal functions (SciPy-free) |
| `core.statistics.reliability` | Redundancy numbers, MDB, external reliability |
| `core.geometry.ellipse` | Error-ellipse computation |

## Example

```python
from survey_adjustment.core.models import Network, Point
from survey_adjustment.core.solver import adjust_network_2d

# Build a network, add observations, then:
result = adjust_network_2d(network)
print(result.variance_factor)
```

!!! note "Auto-generated reference coming"
    A full, auto-generated API reference (via `mkdocstrings`) is planned so that
    signatures and docstrings stay in sync with the code. See the
    [roadmap](https://github.com/Osman-Geomatics93/survey_adjustment/blob/master/ROADMAP.md).
    Until then, this page gives the high-level map; consult the docstrings in
    `core/` for details.
