<p align="center">
  <img src="icon.png" alt="Survey Adjustment Logo" width="120">
</p>

<h1 align="center">Survey Adjustment & Network Analysis</h1>

<p align="center">
  <strong>Production-ready least-squares adjustment plugin for QGIS</strong>
</p>

<p align="center">
  <a href="https://qgis.org/"><img src="https://img.shields.io/badge/QGIS-3.22%2B-34a853?style=for-the-badge&logo=qgis&logoColor=white" alt="QGIS 3.22+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-GPLv2+-blue?style=for-the-badge" alt="License: GPL v2+"></a>
  <a href="https://github.com/Osman-Geomatics93/survey_adjustment/releases"><img src="https://img.shields.io/github/v/release/Osman-Geomatics93/survey_adjustment?style=for-the-badge&color=orange" alt="GitHub Release"></a>
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-video-tutorials">Tutorials</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-documentation">Documentation</a>
</p>

---

## Overview

A comprehensive **least-squares adjustment** engine for professional surveying workflows inside QGIS. This plugin transforms QGIS into a powerful geodetic computation platform, supporting everything from simple leveling runs to complex mixed GNSS/classical networks.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Survey Adjustment Plugin                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│  │   2D    │  │   1D    │  │   3D    │  │  Mixed  │            │
│  │Classical│  │Leveling │  │  GNSS   │  │Combined │            │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘            │
│       │            │            │            │                  │
│       └────────────┴────────────┴────────────┘                  │
│                         │                                       │
│              ┌──────────▼──────────┐                           │
│              │  Least Squares      │                           │
│              │  + Robust IRLS      │                           │
│              │  + Statistics       │                           │
│              └──────────┬──────────┘                           │
│                         │                                       │
│       ┌─────────────────┼─────────────────┐                    │
│       ▼                 ▼                 ▼                    │
│  ┌─────────┐      ┌─────────┐      ┌─────────┐                │
│  │  JSON   │      │  HTML   │      │GeoPackage│                │
│  │ Report  │      │ Report  │      │ Layers  │                │
│  └─────────┘      └─────────┘      └─────────┘                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Features

### Adjustment Types

| Type | Description | Observations |
|:-----|:------------|:-------------|
| **2D Classical** | Traditional traverse & network adjustment | Distances, directions, angles |
| **1D Leveling** | Precise height determination | Height differences |
| **3D GNSS** | Baseline vector adjustment | Full 3×3 covariance support |
| **Mixed** | Unified multi-technique solution | All observation types combined |

### Advanced Capabilities

- **Robust Estimation (IRLS)** — Huber, Danish, IGG-III weight functions for automatic outlier handling
- **Constraint Health Analysis** — Clear diagnostics with actionable error messages
- **Auto-Datum** — Automatic minimal constraint application with full audit trail
- **Statistical Testing** — Chi-square global test, standardized residuals, redundancy analysis
- **Reliability Analysis** — MDB (Minimal Detectable Bias) and external reliability metrics
- **Error Ellipses** — Confidence ellipses with configurable probability level

### Output Formats

| Format | Description |
|:-------|:------------|
| **JSON Report** | Machine-readable results for pipelines and automation |
| **HTML Report** | Publication-ready report with statistics and visualizations |
| **GeoPackage** | Spatial layers: adjusted points, error ellipses, residual vectors |

---

## Video Tutorials

Learn how to use every feature of the plugin with these comprehensive video guides:

### Complete Plugin Tutorial
> **Full walkthrough of all plugin features and workflows**

[![Full Tutorial](https://img.shields.io/badge/Watch-Full_Tutorial-FF0000?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/file/d/1yjOBM-kpYZ6LB9uowH4fpO04oKj1tjH3/view?usp=sharing)

### Algorithm-Specific Tutorials

| Algorithm | Description | Video Link |
|:----------|:------------|:-----------|
| **Adjust Leveling (1D)** | Height difference adjustment with benchmarks | [![Watch](https://img.shields.io/badge/Watch-Video-FF0000?style=flat-square&logo=googledrive)](https://drive.google.com/file/d/1sLe45rJhrYYpp8DASJ-lYSWszHqzlgDm/view?usp=sharing) |
| **Adjust Network (2D)** | Classical traverse and network adjustment | [![Watch](https://img.shields.io/badge/Watch-Video-FF0000?style=flat-square&logo=googledrive)](https://drive.google.com/file/d/1N9_t8FQtLPAAnLLZiUSlpwdZr4dW0zyi/view?usp=sharing) |
| **Adjust Network (3D GNSS)** | GNSS baseline vector adjustment | [![Watch](https://img.shields.io/badge/Watch-Video-FF0000?style=flat-square&logo=googledrive)](https://drive.google.com/file/d/1VOEii6IZvkwYTJ2lhaaT_6GqXhuGgHA8/view?usp=sharing) |
| **Adjust Network (Mixed)** | Combined classical + GNSS + leveling | [![Watch](https://img.shields.io/badge/Watch-Video-FF0000?style=flat-square&logo=googledrive)](https://drive.google.com/file/d/1kH2ZBakFIvABrYZLGPI3Q5YUouHIkIeZ/view?usp=sharing) |

---

## Installation

### Method 1: QGIS Plugin Manager (Recommended)
```
1. Open QGIS
2. Navigate to: Plugins → Manage and Install Plugins
3. Search for "Survey Adjustment"
4. Click "Install Plugin"
```

### Method 2: Install from ZIP
```
1. Download the latest release from GitHub Releases
2. In QGIS: Plugins → Manage and Install Plugins → Install from ZIP
3. Select the downloaded ZIP file
4. Click "Install Plugin"
```

### Requirements
- **QGIS 3.22** or later
- **NumPy** (included with QGIS)
- No additional dependencies required (statistics implemented without SciPy)

---

## Quick Start

### Step 1: Prepare Your Data

Create CSV files for your survey data following the formats below.

### Step 2: Run an Algorithm

```
Processing Toolbox → Survey Adjustment → [Select Algorithm]
```

### Step 3: Review Results

- Open the **HTML report** for a comprehensive summary
- Load the **GeoPackage** layers to visualize results in QGIS
- Use the **JSON report** for further processing or archival

### Interpreting Results

| Indicator | Good Value | What It Means |
|:----------|:-----------|:--------------|
| Variance Factor | ≈ 1.0 | Your observation sigmas are realistic |
| Standardized Residual \|w\| | < 3.0 | No outliers detected |
| Chi-Square Test | Pass | Model fits data at chosen confidence level |
| Weight Factor (robust) | = 1.0 | Observation not downweighted |

---

## Algorithms

### Validate Survey Network
> Pre-flight check for your network before adjustment

Checks topology, connectivity, and datum constraints. Identifies issues before they cause adjustment failures.

### Adjust Network (2D)
> Classical horizontal network adjustment

**Inputs:** Points CSV + Distance/Direction/Angle observations
**Use for:** Traverse adjustments, control networks, boundary surveys

### Adjust Leveling (1D)
> Precise leveling adjustment

**Inputs:** Leveling points CSV + Height differences CSV
**Use for:** Benchmark networks, vertical control, construction leveling

### Adjust Network (3D GNSS Baselines)
> GNSS vector network adjustment

**Inputs:** GNSS points CSV + Baselines CSV (with full covariance)
**Use for:** GNSS control networks, RTK base networks

### Adjust Network (Mixed)
> Unified multi-technique adjustment

**Inputs:** Points CSV + Any combination of observation types
**Use for:** Integrated surveys combining classical, GNSS, and leveling data

---

## Input Formats

> **Units:** Internally uses meters and radians. QGIS interface provides unit selection.

### Points CSV (2D Classical)

| Column | Required | Description |
|:-------|:---------|:------------|
| `point_id` | Yes | Unique point identifier |
| `easting` | Yes | E coordinate (meters) |
| `northing` | Yes | N coordinate (meters) |
| `fixed_easting` | No | true/false - fix E coordinate |
| `fixed_northing` | No | true/false - fix N coordinate |
| `sigma_easting` | No | Prior std dev for E (meters) |
| `sigma_northing` | No | Prior std dev for N (meters) |

```csv
point_id,easting,northing,fixed_easting,fixed_northing
A,1000.000,1000.000,true,true
B,1100.000,1000.000,false,false
C,1100.000,1100.000,false,false
```

### Distances CSV

```csv
obs_id,from_id,to_id,distance,sigma
D01,A,B,100.000,0.005
D02,B,C,100.000,0.005
```

### Directions CSV

```csv
obs_id,from_id,to_id,direction,sigma,set_id
R01,A,B,45.000000,5.0,SET_A
R02,A,C,90.000000,5.0,SET_A
```

### Angles CSV

```csv
obs_id,from_id,at_id,to_id,angle,sigma
A01,B,A,C,45.000000,10.0
```

### Leveling Points CSV

```csv
point_id,height,fixed_height,sigma_height
BM1,100.000,true,0.000
P2,100.120,false,0.002
```

### Height Differences CSV

```csv
obs_id,from_id,to_id,dh,sigma
H01,BM1,P2,0.120,0.002
```

### GNSS Points CSV (3D)

```csv
point_id,easting,northing,height,fixed_easting,fixed_northing,fixed_height
REF,500000.0,4500000.0,120.0,true,true,true
P1,500050.0,4500020.0,121.1,false,false,false
```

### GNSS Baselines CSV

Supports either **full covariance** or **sigmas + correlations**:

```csv
obs_id,from_id,to_id,dE,dN,dH,cov_EE,cov_EN,cov_EH,cov_NN,cov_NH,cov_HH
G01,REF,P1,50.012,20.001,1.102,0.000004,0.000000,0.000000,0.000004,0.000000,0.000009
```

---

## Outputs

### GeoPackage Layers

| Layer | Contents |
|:------|:---------|
| `adjusted_points` | Adjusted coordinates with standard deviations |
| `error_ellipses` | 2D confidence ellipses (a, b, θ, confidence) |
| `residual_vectors` | Visual QA vectors (scaled by configurable factor) |
| `residuals` | Full residual statistics table |

### HTML Report Contents

- Adjustment summary and settings
- Constraint health analysis
- Adjusted coordinates with precision
- Residual analysis with outlier flags
- Chi-square test results
- Error ellipse parameters

### JSON Report Structure

Machine-readable format including:
- Complete adjustment results
- Full covariance matrix
- Settings snapshot for reproducibility
- All statistical test results

---

## Mathematical Model

### Observation Equations

**2D Distance:**
$$l_{ij} = \sqrt{(E_j-E_i)^2 + (N_j-N_i)^2}$$

**2D Direction (Azimuth from North, clockwise):**
$$\alpha_{ij}=\text{atan2}(E_j-E_i,\;N_j-N_i)$$

**Direction with orientation unknown:**
$$l_{ij}=\alpha_{ij}+\omega_s$$

**2D Angle:**
$$l_{ikj} = \alpha_{kj} - \alpha_{ki}$$

**1D Leveling:**
$$l_{ij} = H_j - H_i$$

**3D GNSS Baseline:**
$$\mathbf{l}_{ij}= \begin{bmatrix} dE \\ dN \\ dH \end{bmatrix} = \begin{bmatrix} E_j-E_i \\ N_j-N_i \\ H_j-H_i \end{bmatrix}$$

### Least Squares Solution

Linearized model around initial estimate:
$$\mathbf{w} = \mathbf{l} - \mathbf{f}(\mathbf{x}_0),\quad \mathbf{A}=\frac{\partial \mathbf{f}}{\partial \mathbf{x}}\bigg\rvert_{\mathbf{x}_0}$$

Normal equations:
$$\mathbf{N} = \mathbf{A}^T\mathbf{P}\mathbf{A},\quad \mathbf{n} = \mathbf{A}^T\mathbf{P}\mathbf{w},\quad \delta\mathbf{x} = \mathbf{N}^{-1}\mathbf{n}$$

A-posteriori variance factor:
$$\hat{\sigma}_0^2 = \frac{\mathbf{v}^T\mathbf{P}\mathbf{v}}{\nu},\quad \nu = m-u$$

### Error Ellipses

From the 2×2 covariance matrix, eigenvalues define semi-axes:
$$a = \sqrt{\lambda_1}\cdot k,\quad b = \sqrt{\lambda_2}\cdot k,\quad k=\sqrt{\chi^2_{p,2}}$$

---

## Robust Estimation (IRLS)

Iteratively Reweighted Least Squares automatically handles outliers:

```
1. Solve standard least squares → residuals v
2. Compute standardized residuals w
3. Update weights: p ← p × φ(|w|)
4. Repeat until convergence
```

### Weight Functions

| Method | Formula | Default Parameters |
|:-------|:--------|:-------------------|
| **Huber** | φ(t) = min(1, c/t) | c = 1.5 |
| **Danish** | φ(t) = exp(-(t-c)²) for t > c | c = 2.0 |
| **IGG-III** | Piecewise with hard rejection | k₀ = 1.5, k₁ = 3.0 |

---

## Statistics & Reliability

### Global Test (Chi-Square)
$$T = \frac{\mathbf{v}^T\mathbf{P}\mathbf{v}}{\sigma_0^2} \sim \chi^2_\nu$$

### Local Test (Standardized Residuals)
$$w_i = \frac{v_i}{\sigma_0\sqrt{q_{vv,ii}}}$$

### Reliability Metrics

| Metric | Formula | Purpose |
|:-------|:--------|:--------|
| Redundancy Number | $r_i = q_{vv,ii} \cdot p_i$ | Internal reliability |
| MDB | $(k_\alpha + k_\beta)\hat{\sigma}_0\frac{\sigma_i}{\sqrt{r_i}}$ | Minimal detectable bias |
| External Reliability | $\delta\mathbf{x}_i = \mathbf{Q}_{xx}(p_i\mathbf{A}_i^T)\text{MDB}_i$ | Parameter impact |

---

## Settings

Access via: **Plugins → Survey Adjustment → Survey Adjustment Settings...**

| Setting | Default | Description |
|:--------|:--------|:------------|
| Confidence Level | 0.95 | For ellipses and statistical tests |
| Outlier Threshold | 3.0 σ | Standardized residual flag limit |
| Robust Method | None | Huber / Danish / IGG-III |
| Huber c | 1.5 | Huber tuning constant |
| Danish c | 2.0 | Danish tuning constant |
| IGG-III k₀, k₁ | 1.5, 3.0 | IGG-III bounds |
| Ellipse Vertices | 64 | Polygon approximation |
| Residual Scale | 1000 | Vector visualization scale |

All settings are saved in JSON reports for **full reproducibility**.

---

## Development

### Project Structure

```
survey_adjustment/
├── core/                    # QGIS-independent computation
│   ├── models/              # Point, Network, Observation classes
│   ├── solver/              # Least squares engines (1D, 2D, 3D, mixed)
│   ├── statistics/          # Statistical tests and reliability
│   ├── geometry/            # Error ellipses, residual vectors
│   ├── results/             # Result structures
│   └── reports/             # HTML report generation
├── qgis_integration/        # QGIS-specific code
│   ├── algorithms/          # Processing algorithms
│   ├── gui/                 # Settings dialog
│   ├── io/                  # CSV/layer parsers
│   ├── plugin.py            # Plugin lifecycle
│   └── provider.py          # Processing provider
├── __init__.py              # Entry point
├── metadata.txt             # Plugin metadata
└── icon.png                 # Plugin icon
```

### Running Tests

```bash
pytest
```

### Design Principles

- **Core is QGIS-free** — Can be used as a standalone Python library
- **No SciPy dependency** — Statistics implemented from scratch for portability
- **Lazy imports** — QGIS code only loaded when running in QGIS

---

## Support & Contributing

### Found a Bug?

Please open an issue with:
- Input CSV files
- Output JSON/HTML report
- QGIS version and plugin version
- Steps to reproduce

### Want to Contribute?

Pull requests are welcome! Please ensure:
- Tests pass (`pytest`)
- Code follows existing patterns
- Documentation is updated

### Links

- **Repository:** https://github.com/Osman-Geomatics93/survey_adjustment
- **Issues:** https://github.com/Osman-Geomatics93/survey_adjustment/issues
- **Releases:** https://github.com/Osman-Geomatics93/survey_adjustment/releases

---

## License

This project is licensed under the **GNU General Public License v2.0 or later**.

See [LICENSE](LICENSE) for details.

---

## Author

**Osman Ibrahim**
Email: 422436@ogr.ktu.edu.tr
GitHub: [@Osman-Geomatics93](https://github.com/Osman-Geomatics93)

---

<p align="center">
  <sub>Made with dedication for the surveying community</sub>
</p>
