# Tutorial: 2D classical network

This walkthrough adjusts a small horizontal network from distance, direction,
and angle observations. A ready-to-run version of this dataset ships in the
repository under
[`examples/2d_traverse/`](https://github.com/Osman-Geomatics93/survey_adjustment/tree/master/examples/2d_traverse).

## 1. Prepare the inputs

**`points.csv`** — fix two control points to define the datum:

```csv
point_id,easting,northing,fixed_easting,fixed_northing
CP1,1000.000,2000.000,true,true
CP2,1500.000,2000.000,true,true
T1,1100.000,2150.000,false,false
T2,1250.000,2250.000,false,false
T3,1400.000,2150.000,false,false
```

**`distances.csv`**:

```csv
obs_id,from_id,to_id,distance,sigma
D01,CP1,T1,180.278,0.005
D02,T1,T2,180.278,0.005
D03,T2,T3,180.278,0.005
D04,T3,CP2,180.278,0.005
```

Add `directions.csv` and `angles.csv` as needed (see
[Input formats](../input-formats.md)).

## 2. Run the adjustment

In the QGIS **Processing Toolbox**:

```
Survey Adjustment → Adjust Network (2D)
```

Select the points and observation tables, choose a confidence level (default
0.95), and run. Optionally enable **robust estimation** to downweight outliers,
or **Auto-Datum** for quick exploration.

!!! tip "Validate first"
    Run **Validate Survey Network** before adjusting. It checks connectivity,
    datum constraints, and degrees of freedom, and reports actionable issues
    before they cause an adjustment to fail.

## 3. Read the results

The HTML report opens with a constraint-health summary and the adjusted
coordinates with precisions:

```
Status: ✓ Success | Converged | Iterations: 2 | DOF: 13 | σ₀² = 0.193

ID    E           N           σE       σN
CP1   1000.0000   2000.0000   0.0000   0.0000
CP2   1500.0000   2000.0000   0.0000   0.0000
T1    1100.0028   2150.0026   0.0014   0.0014
T2    1250.0020   2250.0014   0.0018   0.0013
T3    1400.0037   2150.0001   0.0015   0.0013
```

## 4. Interpret the quality control

| Indicator | Good value | Meaning |
|:----------|:-----------|:--------|
| Variance factor $\hat{\sigma}_0^2$ | ≈ 1.0 | Observation sigmas are realistic |
| Standardized residual $\lvert w_i \rvert$ | < 3.0 | No outlier detected |
| Chi-square global test | Pass | Model fits the data at the chosen level |

If the chi-square test fails, inspect the observations flagged with the largest
$\lvert w_i \rvert$ and consider robust estimation. The mathematics behind these
diagnostics is described in [Theory & quality control](../theory.md).

## 5. Map the output

Load the GeoPackage to see the adjusted points, **error ellipses**, and
**residual vectors** on the canvas for visual QA.
