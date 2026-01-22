# Survey Adjustment & Network Analysis

Production-ready QGIS plugin for least-squares adjustment of survey networks.

## Features

### Adjustment Types
- **2D Classical** - Distances, directions, angles
- **1D Leveling** - Height differences with benchmarks
- **3D GNSS Baselines** - Full 3x3 covariance support
- **Mixed** - Combine classical + GNSS + leveling in one adjustment

### Robust Estimation
- IRLS (Iteratively Reweighted Least Squares)
- Huber, Danish, and IGG-III weight functions
- Automatic outlier downweighting with weight factors in output

### Constraint Health & Auto-Datum
- Structured analysis of datum constraints (horizontal, height, orientation)
- Clear error messages when constraints are missing
- Optional auto-datum applies minimal constraints deterministically

### Statistics & Reliability
- Chi-square global test with p-value
- Redundancy numbers per observation
- Minimal Detectable Bias (MDB)
- External reliability
- Error ellipses with confidence scaling

### Outputs
- **JSON** - Machine-readable results with full statistics
- **HTML** - Human-readable report with constraint health summary
- **GeoPackage** - Adjusted points, error ellipses, residual vectors

## Installation

### From QGIS Plugin Manager
1. Plugins → Manage and Install Plugins
2. Search "Survey Adjustment"
3. Click Install

### From ZIP
1. Download the plugin ZIP
2. Plugins → Manage and Install Plugins → Install from ZIP
3. Select the downloaded file

## Usage

Open **Processing Toolbox** → **Survey Adjustment** and select:

- `Adjust Network (2D)` - Classical observations
- `Adjust Leveling (1D)` - Height differences
- `Adjust Network (3D GNSS Baselines)` - GNSS vectors
- `Adjust Network (Mixed)` - Combined observations

### Input Formats

**Points CSV:**
```csv
point_id,easting,northing,height,fixed_e,fixed_n,fixed_h
A,1000.000,2000.000,100.0,true,true,true
B,1100.000,2000.000,100.5,false,false,false
```

**Distances CSV:**
```csv
obs_id,from_point,to_point,distance,sigma
D001,A,B,100.003,0.003
```

**Directions CSV:**
```csv
obs_id,from_point,to_point,direction,sigma,set_id
DIR001,A,B,45.0000,0.00015,SET1
```

**Height Differences CSV:**
```csv
obs_id,from_point,to_point,dh,sigma
HD001,BM1,P1,1.523,0.002
```

**GNSS Baselines CSV:**
```csv
obs_id,from_point,to_point,dE,dN,dH,cov_EE,cov_EN,cov_EH,cov_NN,cov_NH,cov_HH
B001,BASE,P1,100.001,50.002,1.003,0.0001,0.0,0.0,0.0001,0.0,0.0001
```

## Requirements

- QGIS 3.22 or later
- NumPy (included with QGIS)

## License

GPL-2.0-or-later

## Author

Osman Ibrahim (422436@ogr.ktu.edu.tr)
