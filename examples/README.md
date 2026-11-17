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

## How the Observations Were Generated

The observations are synthetic and fully reproducible:

```bash
python examples/generate_observations.py          # seed 2026 (default)
```

The script takes the design coordinates in each `points.csv` as the true
geometry, recomputes every observation from it, and adds a random error drawn
from that observation's own stated precision: `sigma` for distances,
directions, angles and height differences, and the full 3×3 covariance for
GNSS baselines. Observation IDs, sigmas, covariances and decimal places are
unchanged. The errors therefore match the stated sigmas, as they would in good
field data.

## Expected Results

With the default seed, every adjustment converges and passes the two-sided
chi-square global test at 95 %:

| Example | DOF | Variance factor | Global test | Flagged (\|w\| > 3) |
|---|---|---|---|---|
| 1d_leveling | 8 | 0.894 | passes | none |
| 2d_traverse | 13 | 0.547 | passes | none |
| 2d_trilateration | 9 | 1.144 | passes | none |
| 3d_gnss | 27 | 1.113 | passes | none |
| mixed_network | 25 | 1.282 | passes | DIR007, DIR008, BL004 |

The flags in `mixed_network` are not blunders. The generator drew a large but
genuine random error (−2.6σ) for direction DIR008. Because DIR007 and DIR008
form a two-direction set at BM01, the set's orientation unknown splits that
error equally between them (|w| = 3.63 for both), and the resulting shift of
TS03 also raises the standardized residual of baseline BL004 (BM01 → TS03).
This is how data snooping behaves with purely random errors: a 3σ threshold
also flags some good observations. Over 300 random seeds, the mixed network
averaged 0.12 flags per adjustment and passed the global test in 94 % of the
runs, as expected for a correct model.

The committed outputs in each `Result/` folder were generated with plugin
version 1.0.6 in QGIS 3.44.

## Testing Robust Estimation

To test outlier detection, modify any observation value by adding a large error:
- Distance: add 0.05-0.10 m
- Direction: add 0.01-0.02 degrees
- Height difference: add 0.02-0.05 m
- GNSS baseline: add 0.05-0.10 m to any component

Then run with robust estimation (Huber, Danish, or IGG-III) to see downweighting.
