# Tutorial: 1D leveling

Adjust a benchmark network from levelled height differences. A worked dataset
ships under
[`examples/1d_leveling/`](https://github.com/Osman-Geomatics93/survey_adjustment/tree/master/examples/1d_leveling).

## Inputs

**`points.csv`** — fix at least one benchmark to define the height datum:

```csv
point_id,height,fixed_height,sigma_height
BM_A,100.000,true,0.000
BM_B,102.500,true,0.000
TP1,100.850,false,0.002
TP2,101.420,false,0.002
```

**`height_differences.csv`**:

```csv
obs_id,from_id,to_id,dh,sigma
H01,BM_A,TP1,0.850,0.002
H02,TP1,TP2,0.570,0.002
H03,TP2,BM_B,1.080,0.002
```

## Run

```
Survey Adjustment → Adjust Leveling (1D)
```

## Result

```
Status: ✓ Success | Converged | Iterations: 1 | DOF: 8 | σ₀² = 0.0113

ID     H (m)      σH (m)   Fixed
BM_A   100.0000   0.0000   Yes
BM_B   102.5000   0.0000   Yes
TP1    100.8501   0.0001   No
TP2    101.4199   0.0001   No
```

See [Theory & quality control](../theory.md) for how the variance factor and
standardized residuals are computed.

!!! note "Page under development"
    This tutorial will be expanded with a full residual and reliability
    walkthrough. See the
    [roadmap](https://github.com/Osman-Geomatics93/survey_adjustment/blob/master/ROADMAP.md).
