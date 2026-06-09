# Tutorial: 3D GNSS baselines

Adjust a GNSS baseline network using full per-baseline covariance. A worked
dataset ships under
[`examples/3d_gnss/`](https://github.com/Osman-Geomatics93/survey_adjustment/tree/master/examples/3d_gnss).

## Inputs

**`points.csv`** — fix reference stations in E, N, and H:

```csv
point_id,easting,northing,height,fixed_easting,fixed_northing,fixed_height
BASE,500000.0,4500000.0,150.0,true,true,true
GPS1,500150.0,4500200.0,155.2,false,false,false
GPS2,500350.0,4500250.0,148.8,false,false,false
```

**`baselines.csv`** — each baseline carries its full 3×3 covariance:

```csv
obs_id,from_id,to_id,dE,dN,dH,cov_EE,cov_EN,cov_EH,cov_NN,cov_NH,cov_HH
BL001,BASE,GPS1,150.002,200.002,5.203,0.000004,0.0,0.0,0.000004,0.0,0.000009
```

## Run

```
Survey Adjustment → Adjust Network (3D GNSS Baselines)
```

## Result

```
Status: ✓ Success | Converged | Iterations: 1 | DOF: 27 | σ₀² = 0.006

ID     E             N              H          σE       σN       σH
BASE   500000.0000   4500000.0000   150.0000   0.0000   0.0000   0.0000
GPS1   500150.0024   4500200.0022   155.2032   0.0002   0.0002   0.0005
GPS2   500350.0012   4500249.9986   148.8018   0.0002   0.0002   0.0005
```

Per-baseline residuals are reported with standardized residuals and outlier
flags. See [Theory & quality control](../theory.md).

!!! note "Page under development"
    This tutorial will be expanded with covariance-input variants (sigmas +
    correlations) and a reliability walkthrough.
