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

## [1.1.1] - 2026-11-17

### Fixed
- Exceptions were silently swallowed when unloading the plugin and when an error-ellipse polygon could not be built; they are now logged, or reported as a Processing warning. The QGIS plugin repository's Bandit scan blocked v1.1.0 because of these four findings (#29).

### Added
- Bandit security scan in CI, matching the check run by plugins.qgis.org (#29).

---

## [1.1.0] - 2026-11-17

### Added
- Test suites for the 1D leveling, 3D GNSS, mixed and robust solvers, checked against closed-form results (#5).
- `examples/generate_observations.py`: reproducible generator for the example observations (#25).
- Committed outputs for the `2d_trilateration` example (#25).

- Global chi-square test checks against table values for all four solvers (#22).

### Changed
- The example observations now carry random errors consistent with their stated sigmas, so every example passes the global chi-square test; previously the errors were 3–13 times smaller than the sigmas and every example failed at the lower bound (#25).
- JOSS paper rewritten for the current submission requirements, with a results figure and verified references (#3).

### Fixed
- The global chi-square test in the 1D leveling and 3D GNSS solvers used the confidence level as its significance level, so with the default settings it accepted only the central 5 % of the distribution and failed almost every network (#22).
- Robust estimation now isolates gross errors reliably in all four solvers (#16). Standardized residuals in the IRLS loop use the residual cofactors of the ordinary least-squares solution, so down-weighted observations no longer bounce back to full weight; the Danish and IGG-III methods start from a converged Huber solution, so they no longer reject good observations; and the Danish weight never underflows to exactly zero.
- The plugin failed to import on Python 3.9, the interpreter bundled with QGIS 3.22 and 3.28 LTR; CI now also tests Python 3.9 (#1).
- The 3D GNSS and mixed solvers now standardize residuals with the a-priori sigma0, like the 1D and 2D solvers, so outlier flags are consistent across solvers; consistent data no longer produces NaN standardized residuals (#17).
- GNSS redundancy numbers are computed from the full 3×3 covariance blocks and now sum to the degrees of freedom for correlated baselines (#17).

---

## [1.0.6] - 2026-06-10

### Added
- Hosted documentation site built with MkDocs Material and deployed to GitHub Pages, covering installation, input formats, per-solver tutorials, the mathematical theory, and an API overview.
- Development roadmap (`ROADMAP.md`) outlining iterative work toward a future JOSS submission.
- `flake8` linting enforced in CI, configured via `setup.cfg`.

### Changed
- Expanded the JOSS paper with Software design, Research impact statement, and AI usage disclosure sections; added the Baarda (1968) DOI and the affiliation ROR identifier.
- Internal code cleanup: removed unused imports and variables and resolved all flake8 findings. No functional changes.

---

## [1.0.5] - 2026-06-06

### Changed
- Maintenance release published to trigger Zenodo archival for the JOSS submission; no functional changes.

---

## [1.0.4] - 2026-06-06

### Added
- Paper-build workflow (`Draft Paper PDF`) that compiles the JOSS paper with Open Journals' `inara`
- Full equation-rich version of the paper, retained as `paper/paper-full.md`

### Changed
- Use the complete canonical GPL-2.0 license text so the license is auto-detected
- Use the concise (~900-word) JOSS paper as `paper/paper.md`

### Removed
- AI-usage section from the paper

---

## [1.0.3] - 2026-06-06

### Added
- Continuous integration (GitHub Actions) running the test suite on Python 3.10–3.12
- Automated test suite (113 tests) and a `conftest.py` so the core is testable without QGIS
- Software paper for the Journal of Open Source Software (`paper/paper.md`, `paper/paper.bib`)
- Author ORCID in `CITATION.cff`

### Changed
- Use the full author name "Osman Osama Ahmed Ibrahim" across project metadata
- Reconcile documented release dates to January 2026

---

## [1.0.2] - 2026-01-23

### Fixed
- Fixed `QgsLineString.addVertex()` TypeError in 3D GNSS and Mixed algorithms
- Improved error handling for edge cases in baseline processing

### Changed
- Enhanced residual vector visualization for 3D networks

---

## [1.0.1] - 2026-01-22

### Fixed
- Minor bug fixes in constraint health analysis
- Improved CSV parsing for international number formats

### Changed
- Updated documentation and examples

---

## [1.0.0] - 2026-01-22

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
| 1.0.6 | 2026-06-10 | Documentation site, CI linting, paper & code-quality improvements |
| 1.0.5 | 2026-06-06 | Maintenance release (Zenodo archival for JOSS) |
| 1.0.4 | 2026-06-06 | Full GPL text, JOSS paper finalized, paper-build CI |
| 1.0.3 | 2026-06-06 | CI, automated tests, and JOSS paper |
| 1.0.2 | 2026-01-23 | Bug fixes for 3D algorithms |
| 1.0.1 | 2026-01-22 | Documentation and parsing improvements |
| 1.0.0 | 2026-01-22 | Initial release with full functionality |

---

[Unreleased]: https://github.com/Osman-Geomatics93/survey_adjustment/compare/v1.0.6...HEAD
[1.0.6]: https://github.com/Osman-Geomatics93/survey_adjustment/compare/v1.0.5...v1.0.6
[1.0.5]: https://github.com/Osman-Geomatics93/survey_adjustment/compare/v1.0.4...v1.0.5
[1.0.4]: https://github.com/Osman-Geomatics93/survey_adjustment/compare/v1.0.3...v1.0.4
[1.0.3]: https://github.com/Osman-Geomatics93/survey_adjustment/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/Osman-Geomatics93/survey_adjustment/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/Osman-Geomatics93/survey_adjustment/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/Osman-Geomatics93/survey_adjustment/releases/tag/v1.0.0
