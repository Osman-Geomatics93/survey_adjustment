# Test Data for Survey Adjustment Plugin

Realistic test datasets for all adjustment types.

## Datasets

### 1. 2D Traverse (`2d_traverse/`)

A closed traverse network with 2 control points and 3 traverse stations.

| File | Description |
|------|-------------|
| `points.csv` | 5 points: CP1, CP2 (fixed), T1, T2, T3 (free) |
| `distances.csv` | 8 distance observations |
| `directions.csv` | 10 direction observations in 4 sets |
| `angles.csv` | 5 angle observations |

**Use with:** Adjust Network (2D)

---

### 2. 2D Trilateration (`2d_trilateration/`)

Distance-only network with 3 control points and 3 free points.

| File | Description |
|------|-------------|
| `points.csv` | 6 points: BM1, BM2, BM3 (fixed), P1, P2, P3 (free) |
| `distances.csv` | 15 distance observations |

**Use with:** Adjust Network (2D) with auto-datum enabled

---

### 3. 1D Leveling (`1d_leveling/`)

A leveling line between two benchmarks with 4 turning points.

| File | Description |
|------|-------------|
| `points.csv` | 6 points: BM_A, BM_B (fixed height), TP1-TP4 (free) |
| `height_differences.csv` | 12 height difference observations |

**Use with:** Adjust Leveling (1D)

---

### 4. 3D GNSS (`3d_gnss/`)

GNSS baseline network with 2 reference stations and 4 rover points.

| File | Description |
|------|-------------|
| `points.csv` | 6 points: BASE, REF2 (fixed), GPS1-GPS4 (free) |
| `baselines.csv` | 13 GNSS baselines with full 3x3 covariances |

**Use with:** Adjust Network (3D GNSS Baselines)

---

### 5. Mixed Network (`mixed_network/`)

Combined classical + GNSS + leveling network.

| File | Description |
|------|-------------|
| `points.csv` | 6 points: CORS, BM01 (fixed), TS01-TS04 (free) |
| `distances.csv` | 8 total station distances |
| `directions.csv` | 8 directions in 4 sets |
| `baselines.csv` | 6 GNSS baselines |
| `height_differences.csv` | 7 leveling observations |

**Use with:** Adjust Network (Mixed)

---

## Coordinate Systems

- **2D Traverse/Trilateration:** Local grid (meters)
- **1D Leveling:** Local grid, heights in meters
- **3D GNSS:** UTM-like projection (meters)
- **Mixed:** Local grid (meters)


## Units & QGIS Parameter Settings

These files are meant to work with the **default** settings of the QGIS algorithms unless noted.

### 2D Traverse / 2D Trilateration / Mixed (Classical)

- `direction` and `angle` values are in **degrees**.
- `sigma` for directions/angles is in **radians** (≈ 5 arcsec ≈ 2.4e-5 rad).

In QGIS:
- **Angles/directions in degrees** = ON
- **Angular sigmas in arcseconds** = OFF (because sigmas are radians)

Distances:
- `distance` in **meters**, `sigma` in **meters**.

### 1D Leveling

- `dh` in **meters**, `sigma` in **meters**.

In QGIS:
- **Sigma in millimeters (otherwise meters)** = OFF

### 3D GNSS / Mixed (GNSS)

- `dE`, `dN`, `dH` in **meters**
- Covariances (`cov_*`) in **m²** (full 3×3 covariance per baseline)

In QGIS:
- **Covariance format** = Full covariance (for the provided `baselines.csv`)

## Expected Results

All datasets contain small random errors and produce:
- Converged solutions (1-3 iterations)
- No flagged outliers (all |standardized residuals| < 3)
- Variance factors well below 1.0 (0.006-0.19): the observations agree much
  better than their stated sigmas imply. The two-sided chi-square global test
  therefore fails at the **lower** bound, which is the expected result when the
  a-priori sigmas are pessimistic. Tightening the sigmas in the input files
  brings the variance factor towards 1.0 and lets the test pass.

| Example | DOF | Variance factor | Global test |
|---|---|---|---|
| 1d_leveling | 8 | 0.011 | fails (lower bound) |
| 2d_traverse | 13 | 0.193 | fails (lower bound) |
| 3d_gnss | 27 | 0.006 | fails (lower bound) |
| mixed_network | 25 | 0.155 | fails (lower bound) |

The committed outputs in each `Result/` folder were generated with plugin
version 1.0.6 in QGIS 3.44 (1D, 3D, and mixed: regenerated after the fixes in
#21 and #23; the 2D traverse output dates from version 1.0.1 and is not
affected by those fixes).

## Testing Robust Estimation

To test outlier detection, modify any observation value by adding a large error:
- Distance: add 0.05-0.10 m
- Direction: add 0.01-0.02 degrees
- Height difference: add 0.02-0.05 m
- GNSS baseline: add 0.05-0.10 m to any component

Then run with robust estimation (Huber, Danish, or IGG-III) to see downweighting.
