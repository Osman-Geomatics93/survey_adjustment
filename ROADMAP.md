# Roadmap

This roadmap tracks the planned, incremental development of **Survey Adjustment &
Network Analysis** through to a JOSS resubmission in early 2027. Items are
delivered and committed as the work is actually done — the goal is steady,
iterative progress rather than large batched changes.

Target resubmission: **late January 2027**.

## July 2026 — Documentation site

- [ ] Stand up a hosted documentation site (MkDocs Material / GitHub Pages).
- [ ] Installation guide and input-format reference.
- [ ] One tutorial per solver: 1D leveling, 2D classical, 3D GNSS, mixed.
- [ ] API reference for the NumPy-only core.
- [ ] Theory and quality-control notes (chi-square test, data snooping,
      redundancy numbers, internal/external reliability, robust estimation).

## August 2026 — Test depth & validation

- [ ] Dedicated test suites for the 1D, 3D GNSS, mixed, and robust solvers.
- [ ] Reliability tests (MDB, redundancy numbers, external reliability).
- [ ] Validate solver outputs against published textbook examples
      (Ghilani; Mikhail & Ackermann) with asserted reference values.
- [ ] Add coverage reporting (pytest-cov) and a coverage badge to CI.

## September 2026 — Cross-validation against reference software

- [ ] Reproduce GNU GaMa results on shared networks.
- [ ] Commit a documented comparison report quantifying agreement.

## October 2026 — Feature work from real use

- [ ] Free-network / inner-constraint datum definition.
- [ ] Variance-component estimation.
- [ ] Additional weight models and import ergonomics.
- [ ] Track each feature as a GitHub issue under a milestone before closing it.

## November 2026 — Community & adoption

- [ ] Publish to the official QGIS Plugin Repository for download metrics.
- [ ] Write a tutorial / worked case study from a real surveying project.
- [ ] Gather and act on external user feedback and issues.

## December 2026 — Robustness & polish

- [ ] Edge-case hardening (singular / rank-deficient networks, graceful errors).
- [ ] Performance on larger networks.
- [ ] Docstring and type-hint pass.
- [ ] Tag a minor release reflecting the half-year of work.

## January 2027 — Resubmission prep

- [ ] Update the development-history note with what changed since the
      July 2026 screening.
- [ ] Regenerate the paper PDF, bump the version, cut a release and Zenodo
      archive, and resubmit.

---

Contributions and suggestions are welcome — please open an issue to discuss
ideas before submitting a pull request. See `CONTRIBUTING.md`.
