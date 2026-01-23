# Test Data Fixes

This folder contains the original structure from `test_data.zip`, with the following fixes applied:

## Fixed data errors (typos)

### 2d_trilateration/distances.csv
Some distances were inconsistent with the provided point coordinates (multi-meter misclosures).
The file has been regenerated from the point coordinates with small random noise consistent with each row's `sigma`.

### mixed_network/distances.csv
One distance (`D007`) was inconsistent with the provided point coordinates (≈19 m misclosure).
It has been corrected to be consistent with the points (within the provided `sigma`).

## Documentation
- `README.md`:
  - Fixed a small path typo.
  - Added a clear **Units & QGIS Parameter Settings** section.
