# Changelog

All notable changes to the Survey Adjustment & Network Analysis plugin will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Network visualization with automatic sketching
- Batch processing for multiple networks
- Export to additional formats (DXF, KML)

---

## [1.0.2] - 2024-12-XX

### Fixed
- Fixed `QgsLineString.addVertex()` TypeError in 3D GNSS and Mixed algorithms
- Improved error handling for edge cases in baseline processing

### Changed
- Enhanced residual vector visualization for 3D networks

---

## [1.0.1] - 2024-11-XX

### Fixed
- Minor bug fixes in constraint health analysis
- Improved CSV parsing for international number formats

### Changed
- Updated documentation and examples

---

## [1.0.0] - 2024-10-XX

### Added
- **2D Classical Network Adjustment**
  - Distance observations
  - Direction observations with orientation unknowns
  - Angle observations
  - Full covariance propagation

- **1D Leveling Adjustment**
  - Height difference observations
  - Multiple benchmark support
  - Loop closure analysis

- **3D GNSS Baseline Adjustment**
  - Full 3×3 covariance matrix support
  - Correlation handling between components
  - Combined horizontal and vertical adjustment

- **Mixed Network Adjustment**
  - Unified solution combining classical + GNSS + leveling
  - Automatic observation type detection
  - Weighted combination of different techniques

- **Robust Estimation (IRLS)**
  - Huber weight function
  - Danish weight function
  - IGG-III weight function
  - Automatic outlier downweighting with audit trail

- **Statistical Analysis**
  - Chi-square global test with p-value
  - Standardized residuals for local testing
  - Redundancy numbers per observation
  - Configurable outlier threshold

- **Reliability Analysis**
  - Minimal Detectable Bias (MDB) computation
  - External reliability metrics
  - Internal reliability assessment

- **Constraint Health Analysis**
  - Automatic datum defect detection
  - Clear error messages with suggested fixes
  - Optional auto-datum with transparency

- **Output Formats**
  - JSON reports for automation
  - HTML reports for documentation
  - GeoPackage with spatial layers

- **Error Ellipses**
  - Confidence ellipse computation
  - Configurable confidence level
  - Export as polygon geometries

- **QGIS Integration**
  - Processing toolbox algorithms
  - Settings dialog with persistence
  - Layer styling for results

---

## Version History Summary

| Version | Date | Highlights |
|:--------|:-----|:-----------|
| 1.0.2 | 2024-12 | Bug fixes for 3D algorithms |
| 1.0.1 | 2024-11 | Documentation and parsing improvements |
| 1.0.0 | 2024-10 | Initial release with full functionality |

---

[Unreleased]: https://github.com/Osman-Geomatics93/survey_adjustment/compare/v1.0.2...HEAD
[1.0.2]: https://github.com/Osman-Geomatics93/survey_adjustment/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/Osman-Geomatics93/survey_adjustment/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/Osman-Geomatics93/survey_adjustment/releases/tag/v1.0.0
