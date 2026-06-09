# Input formats

All inputs are plain **CSV** files: comma-delimited, period (`.`) as the decimal
separator, UTF-8 (without BOM). Point IDs are case-sensitive and must match
exactly across the points file and the observation files.

!!! info "Units"
    Internally the engine works in **meters** and **radians**. In CSV input,
    directions and angles are given in **degrees** and their sigmas in
    **arc-seconds**; the QGIS interface also offers unit selection. Azimuth is
    measured from North, clockwise positive.

## Points

### 2D classical points

| Column | Required | Description |
|:-------|:--------:|:------------|
| `point_id` | ✓ | Unique point identifier |
| `easting` | ✓ | E coordinate (m) |
| `northing` | ✓ | N coordinate (m) |
| `fixed_easting` | | `true`/`false` — fix the E coordinate |
| `fixed_northing` | | `true`/`false` — fix the N coordinate |
| `sigma_easting` | | Prior standard deviation for E (m) |
| `sigma_northing` | | Prior standard deviation for N (m) |

```csv
point_id,easting,northing,fixed_easting,fixed_northing
A,1000.000,2000.000,true,true
B,1000.000,2100.000,true,true
C,1050.000,2050.000,false,false
```

### Leveling points (1D)

```csv
point_id,height,fixed_height,sigma_height
BM1,100.000,true,0.000
P2,100.120,false,0.002
```

### GNSS points (3D)

```csv
point_id,easting,northing,height,fixed_easting,fixed_northing,fixed_height
REF,500000.0,4500000.0,120.0,true,true,true
P1,500050.0,4500020.0,121.1,false,false,false
```

## Observations

### Distances

```csv
obs_id,from_id,to_id,distance,sigma
D01,A,B,100.000,0.005
D02,B,C,100.000,0.005
```

### Directions

Directions from a common setup share a `set_id`; each set carries one
orientation unknown.

```csv
obs_id,from_id,to_id,direction,sigma,set_id
R01,A,B,45.000000,5.0,SET_A
R02,A,C,90.000000,5.0,SET_A
```

### Angles

```csv
obs_id,from_id,at_id,to_id,angle,sigma
A01,B,A,C,45.000000,10.0
```

### Height differences (1D)

```csv
obs_id,from_id,to_id,dh,sigma
H01,BM1,P2,0.120,0.002
```

### GNSS baselines (3D)

Provide either a full per-baseline covariance matrix (shown) or sigmas with
correlations.

```csv
obs_id,from_id,to_id,dE,dN,dH,cov_EE,cov_EN,cov_EH,cov_NN,cov_NH,cov_HH
G01,REF,P1,50.012,20.001,1.102,0.000004,0.000000,0.000000,0.000004,0.000000,0.000009
```

## Outputs

Every adjustment produces:

- a **JSON report** (machine-readable: full results, complete covariance matrix,
  and a settings snapshot for reproducibility),
- an **HTML report** (summary, constraint health, residuals, statistics,
  ellipses), and
- a **GeoPackage** with `adjusted_points`, `error_ellipses`, `residual_vectors`,
  and a `residuals` table.
