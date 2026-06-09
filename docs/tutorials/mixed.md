# Tutorial: mixed network

A mixed adjustment combines classical observations, GNSS baselines, and leveling
in a single unified solution. A worked dataset ships under
[`examples/mixed_network/`](https://github.com/Osman-Geomatics93/survey_adjustment/tree/master/examples/mixed_network).

## Inputs

Supply a single points file plus any combination of the observation files
documented in [Input formats](../input-formats.md): distances, directions,
angles, height differences, and GNSS baselines. Points may be constrained in any
subset of E, N, and H.

## Run

```
Survey Adjustment → Adjust Network (Mixed)
```

## Result

```
Status: ✓ Success | Converged | Iterations: 3 | DOF: 25 | σ₀² = 0.155

Observation Mix: 16 classical + 6 GNSS baselines + 7 leveling

ID     E            N            H          σE       σN       σH
CORS   10000.0000   20000.0000   500.0000   0.0000   0.0000   0.0000
TS01   10100.0001   20099.9986   501.8005   0.0010   0.0010   0.0004
TS02   10150.0009   20200.0013   503.2008   0.0010   0.0011   0.0006
```

The report lists the observation mix and flags residuals across all observation
types together. See [Theory & quality control](../theory.md).

!!! note "Page under development"
    This tutorial will be expanded with guidance on combining heterogeneous
    observation weights and interpreting the joint quality control.
