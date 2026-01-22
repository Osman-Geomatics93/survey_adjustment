# Survey Adjustment & Network Analysis

<div align="center">

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![QGIS](https://img.shields.io/badge/QGIS-3.22+-green.svg)
![Python](https://img.shields.io/badge/python-3.9+-yellow.svg)
![Tests](https://img.shields.io/badge/tests-286%20passed-brightgreen.svg)
![License](https://img.shields.io/badge/license-GPL--2.0-lightgrey.svg)

**Rigorous least-squares adjustment for survey networks with statistical testing and reliability measures**

[Installation](#installation) •
[Features](#features) •
[Quick Start](#quick-start) •
[Documentation](#documentation) •
[Development](#development)

</div>

---

## Overview

A professional QGIS plugin for **least-squares adjustment** of survey networks, providing:

- **2D Network Adjustment** — Distances, directions, angles with error ellipses
- **1D Leveling Adjustment** — Height difference networks
- **3D GNSS Baseline Adjustment** — Full covariance support
- **Unified Mixed Adjustment** — Classical + GNSS + Leveling in one solve
- **Robust Estimation** — IRLS with Huber, Danish, IGG-III weight functions *(New in v1.0.0)*
- **Constraint Health Analysis** — Actionable messages for datum issues *(New in v1.0.0)*

All adjustments include comprehensive **statistical testing**, **reliability measures**, and **outlier detection**.

---

## Installation

### From ZIP (Recommended)

1. Download `survey_adjustment_v1.0.0.zip`
2. In QGIS: **Plugins → Manage and Install Plugins → Install from ZIP**
3. Select the downloaded ZIP file
4. Enable the plugin

Algorithms appear in the **Processing Toolbox** under **Survey Adjustment**.

### Requirements

- QGIS 3.22 or higher (tested on 3.40 LTR)
- Python 3.9+
- NumPy (included with QGIS)

---

## Features

### Processing Algorithms

| Algorithm | Description |
|-----------|-------------|
| **Validate Survey Network** | Connectivity, datum constraints, solvability checks |
| **Adjust Network (2D)** | Classical horizontal networks |
| **Adjust Leveling (1D)** | Height difference networks |
| **Adjust Network (3D GNSS)** | GNSS baselines with full covariance |
| **Adjust Network (Mixed)** | Unified classical + GNSS + leveling |

### Statistical Analysis

- **Global Chi-square test** with p-value
- **Standardized residuals** for local testing
- **Redundancy numbers** (internal reliability)
- **Minimal Detectable Bias (MDB)**
- **External reliability** (coordinate impact)
- **Error ellipses** at configurable confidence levels

### Output Formats

- **GIS Layers**: Adjusted points, error ellipses, residual vectors
- **Reports**: JSON (machine-readable) and HTML (human-readable)
- **GeoPackage**: Optional consolidated output

---

## Quick Start

### Example: 2D Traverse Adjustment

```
Processing → Survey Adjustment → Adjust Network (2D)
```

**Inputs:**
- Points CSV with coordinates and fixed flags
- Distances, directions, and/or angles CSV

**Outputs:**
- Adjusted coordinates with sigmas
- Error ellipses at 95% confidence
- Residual vectors and statistics

### Example: Mixed Adjustment (Classical + GNSS + Leveling)

```
Processing → Survey Adjustment → Adjust Network (Mixed)
```

**Inputs:**
- Points CSV (E, N, H with fixed flags)
- Classical observations (distances, directions, angles)
- GNSS baselines CSV (optional)
- Leveling CSV (optional)

**Outputs:**
- Unified adjusted coordinates (E, N, H)
- Combined residual analysis
- Comprehensive HTML report

---

## Documentation

### Coordinate Conventions

| Coordinate | Convention |
|------------|------------|
| **Easting (E)** | Meters, positive east |
| **Northing (N)** | Meters, positive north |
| **Height (H)** | Meters (ellipsoidal or orthometric) |
| **Azimuth** | From North, clockwise positive |
| **Angles** | Radians internally, degrees in CSV |

### Mathematical Model

Standard weighted least-squares adjustment:

```
l ≈ A·Δx + v

Solution: (AᵀPA)·Δx = AᵀP·l

Variance factor: σ₀² = vᵀPv / f
```

Where:
- **l** = misclosure vector (observed − computed)
- **A** = design matrix (partial derivatives)
- **P** = weight matrix
- **v** = residuals
- **f** = degrees of freedom

### Observation Equations

#### Distance
```
s_calc = √[(Ej-Ei)² + (Nj-Ni)²]
l = s_obs - s_calc
p = 1/σ²
```

#### Direction (with orientation unknown)
```
d_calc = atan2(Ej-Ei, Nj-Ni) + ω
l = wrap_π(d_obs - d_calc)
p = 1/σ²
```

#### Angle
```
θ_calc = wrap_2π(α_AT - α_AF)
l = wrap_π(θ_obs - θ_calc)
p = 1/σ²
```

#### Height Difference (Leveling)
```
ΔH_calc = Hj - Hi
l = ΔH_obs - ΔH_calc
p = 1/σ²
```

#### GNSS Baseline
```
b_calc = [Ej-Ei, Nj-Ni, Hj-Hi]ᵀ
l = b_obs - b_calc
P = C⁻¹ (inverse covariance)
```

### Input File Formats

#### Points CSV
```csv
point_id,easting,northing,height,fixed_e,fixed_n,fixed_h
BASE,1000.000,2000.000,100.000,true,true,true
P1,1100.000,2050.000,102.500,false,false,false
```

#### Distances CSV
```csv
from,to,distance,sigma
BASE,P1,111.803,0.005
P1,P2,100.000,0.005
```

#### Directions CSV
```csv
from,to,direction,sigma,set_id
BASE,P1,26.565,0.00015,SET1
BASE,P2,45.000,0.00015,SET1
```

#### Leveling CSV
```csv
from,to,dh,sigma
BM1,P1,2.500,0.002
P1,P2,1.200,0.002
```

#### GNSS Baselines CSV (Full Covariance)
```csv
from,to,dE,dN,dH,cov_EE,cov_EN,cov_EH,cov_NN,cov_NH,cov_HH
BASE,P1,100.0,50.0,5.0,0.000004,0.0,0.0,0.000004,0.0,0.000009
```

---

## Interpreting Results

### Variance Factor (σ₀²)

| Value | Interpretation |
|-------|----------------|
| ≈ 1.0 | Sigmas are realistic |
| > 1.0 | Sigmas too optimistic or unmodeled errors |
| < 1.0 | Sigmas too pessimistic |

### Global Test (Chi-square)

- **Passed**: Observations consistent with model
- **Failed**: Possible outliers, unrealistic sigmas, or missing constraints

### Standardized Residuals (w)

- **|w| < 3.0**: Typically acceptable
- **|w| > 3.0**: Flagged as outlier candidate

### Redundancy Numbers (r)

| Value | Controllability |
|-------|-----------------|
| r > 0.5 | Well-controlled observation |
| r < 0.3 | Poorly controlled (weak point) |
| r ≈ 0 | Observation defines the parameter |

---

## Development

### Project Structure

```
survey_adjustment/
├── core/                    # Pure Python (no QGIS)
│   ├── models/              # Point, Observation, Network
│   ├── solver/              # Least-squares engines
│   ├── statistics/          # Chi-square, reliability
│   └── reports/             # HTML generation
├── qgis_integration/        # QGIS-specific
│   ├── algorithms/          # Processing algorithms
│   └── io/                  # CSV/file parsers
└── tests/                   # Unit tests
```

### Running Tests

```bash
cd survey_adjustment_phase5
python -m pytest -v
```

Current status: **286 tests passing**

### Building the Plugin

```bash
git archive --format=zip --prefix=survey_adjustment/ \
  -o survey_adjustment_v1.0.0.zip HEAD:survey_adjustment
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| **1.0.0** | 2026-01 | Robust estimation (IRLS), constraint health analysis, auto-datum |
| 0.9.0 | 2026-01 | Robust estimation with Huber, Danish, IGG-III |
| 0.8.0 | 2025-01 | Unified mixed adjustment (classical + GNSS + leveling) |
| 0.7.0 | 2025-01 | Mixed adjustment (classical + GNSS) |
| 0.6.0 | 2025-01 | 3D GNSS baseline adjustment |
| 0.5.0 | 2025-01 | 1D leveling adjustment |
| 0.4.0 | 2025-01 | QGIS Processing integration |
| 0.3.0 | 2025-01 | Statistical testing & reliability |
| 0.2.0 | 2025-01 | 2D least-squares solver |
| 0.1.0 | 2025-01 | Data model foundation |

---

## Author

**Osman Ibrahim**

---

## License

GPL-2.0-or-later - See [LICENSE](LICENSE) for details.

---

## Citation

If you use this plugin in academic work, please cite:

```
Survey Adjustment & Network Analysis QGIS Plugin, v1.0.0
https://github.com/Osman-Geomatics93/survey_adjustment
```

---

## Disclaimer

This plugin implements standard least-squares adjustment theory and common surveying diagnostics. Always validate inputs, datum constraints, and results against your organization's QA/QC procedures before using in production deliverables.
