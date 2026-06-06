# Development-history / substantial-effort statement (for the JOSS editor)

**Purpose.** JOSS's submission guidance suggests projects normally have a public
development history (often cited as roughly six months) with releases and public
issues/PRs. The public GitHub repository for this project was created recently and
therefore shows a short commit timeline. This note explains why the work
nonetheless meets JOSS's actual bar — *substantial scholarly effort and a
feature-complete, well-documented, tested research tool* — and is intended to be
posted in the review thread if an editor or reviewer raises the question.

The author-specific context below has been completed from the author's own
account of the project's development.

---

## A. Short version (paste into the review issue)

> Thank you for raising the development-history point. The public GitHub
> repository is recent because it was published as the canonical home for an
> already-developed tool rather than grown commit-by-commit in the open; the
> short commit log reflects the publication date, not the total effort behind the
> software. The plugin has been independently released through the **official QGIS
> plugin repository** (https://plugins.qgis.org/plugins/survey_adjustment/) across
> versions 1.0.0–1.0.2, which provides a third-party, versioned distribution
> record. The codebase itself reflects substantial scholarly effort: a
> dependency-light (NumPy-only) least-squares engine with four solvers (1D
> leveling, 2D classical, 3D GNSS, mixed), from-scratch statistical distributions
> (no SciPy), robust IRLS estimation, Baarda-style internal/external reliability,
> error-ellipse geometry, an automated test suite (113 tests, CI on Python
> 3.10–3.12), five worked example datasets with committed expected outputs, and
> full user documentation. The plugin was developed independently over roughly
> four to six months to support the author's own professional surveying work, and
> it is already used in practice — in real survey projects, in teaching, and as a
> research component. I am committed to maintaining the project openly going
> forward and am happy to address reviewer issues in public on this repository.

---

## B. Longer version (if a fuller written justification is requested)

### B.1 The relevant criterion
JOSS evaluates whether a submission represents *substantial scholarly effort* and
is a *feature-complete, documented, tested* research tool — the ~6-month figure is
guidance for gauging maturity, not a hard gate. A project that was developed before
being opened to the public can qualify when the repository becomes its genuine,
maintained home and the software substance is evident. This submission is in that
category.

### B.2 Evidence of substantial effort (verifiable in the repository)
- **Breadth of functionality.** Five Processing tools: a network validator plus
  four distinct least-squares solvers (1D leveling, 2D classical with
  distances/directions/angles and per-setup orientation unknowns, 3D GNSS
  baselines with full 3×3 covariance, and a unified mixed model).
- **From-scratch numerical/statistical implementation.** The chi-square cumulative
  and quantile functions are implemented directly (regularized incomplete-gamma via
  series/continued-fraction) so the plugin needs only NumPy and installs cleanly
  against a stock QGIS Python — a deliberate, non-trivial engineering choice rather
  than a thin wrapper around an existing library.
- **Quality-control depth.** Chi-square global test, standardized-residual data
  snooping, redundancy numbers, robust IRLS (Huber/Danish/IGG-III), and Baarda-style
  internal (MDB) and external reliability — the diagnostic layer normally found only
  in high-end commercial adjustment packages.
- **Engineering discipline.** A QGIS-independent core that is importable as a
  standalone library, lazy QGIS imports, an automated test suite (**113 tests**)
  with **continuous integration passing on Python 3.10, 3.11, and 3.12**, and a
  `.gitignore`/`conftest` setup that lets the core be tested without QGIS.
- **Documentation and reproducibility.** A comprehensive README (installation,
  input formats, mathematical model, FAQ, troubleshooting), `CONTRIBUTING.md`,
  `SECURITY.md`, issue/PR templates, a `CHANGELOG.md`, a `CITATION.cff`, and
  machine-readable JSON reports that carry the full covariance matrix and a settings
  snapshot for reproducibility.
- **Worked, validated examples.** Five committed datasets (1D leveling, 2D traverse,
  2D trilateration, 3D GNSS, mixed) each shipped with expected JSON/HTML/GeoPackage
  outputs that double as de-facto validation fixtures.

### B.3 Independent release record
The plugin is published on the **official QGIS plugin repository**
(https://plugins.qgis.org/plugins/survey_adjustment/) and has gone through
versions **1.0.0, 1.0.1, and 1.0.2**, each with documented changes in
`CHANGELOG.md`. This is an independent, third-party distribution channel with its
own moderation and versioning, demonstrating that the tool was released and
iterated, not freshly assembled for submission.

### B.4 Context and provenance
The plugin was developed independently by the author over approximately four to
six months, motivated by the needs of his own professional surveying work and the
absence of a free, QGIS-native adjustment engine with rigorous quality-control
diagnostics. The public GitHub history is recent because it marks the point at
which an already-working tool was consolidated and opened for release, not the full
development effort. The software is already applied in practice — in real survey
projects, in teaching surveying and geodesy, and as a component in research — and
the author intends to continue developing and maintaining it openly.

### B.5 Maintenance commitment
The GitHub repository is now the canonical, public home of the project. I will
conduct the JOSS review openly here, respond to issues and pull requests, and
continue maintenance and the published roadmap (network sketching, batch
processing, additional export formats) as the sole maintainer.

---

## C. Housekeeping to do BEFORE posting this (so the record is consistent)
An editor may cross-check dates, so reconcile these first:
1. **Date consistency — RESOLVED.** Release dates are now reconciled to January
   2026 across `CHANGELOG.md` (v1.0.0 = 2026-01-22, v1.0.1 = 2026-01-22,
   v1.0.2 = 2026-01-23), `CITATION.cff` (`date-released: 2026-01-23`), and the
   BibTeX key in `README.md`, matching the commit history. No `-XX` placeholders
   remain.
2. **Grow a little public signal where cheap and honest.** Before/at submission,
   create real **tagged GitHub Releases** matching the versions, enable Issues, and
   (optionally) open a couple of genuine tracking issues for the roadmap items. This
   converts "14 commits in two days" into a repository that visibly has releases and
   an issue tracker.
3. **Keep claims truthful.** This statement only asserts things visible in the repo
   plus whatever you fill into the [AUTHOR] blanks — keep it that way.
