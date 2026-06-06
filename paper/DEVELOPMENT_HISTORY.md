# Development-history / substantial-effort statement (for the JOSS editor)

**Purpose.** JOSS's submission guidance suggests projects normally have a public
development history (often cited as roughly six months) with releases and public
issues/PRs. The public GitHub repository for this project was created recently and
therefore shows a short commit timeline. This note explains why the work
nonetheless meets JOSS's actual bar — *substantial scholarly effort and a
feature-complete, well-documented, tested research tool* — and is intended to be
posted in the review thread if an editor or reviewer raises the question.

Items in **[AUTHOR: …]** are placeholders only you can fill accurately — please
complete or delete them before posting. **Do not invent dates or facts.**

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
> full user documentation. [AUTHOR: add one sentence on the real development
> timeline and context, e.g. "It was developed over [PERIOD] as part of
> [coursework / MSc study / professional surveying work] at Karadeniz Technical
> University."] I am committed to maintaining the project openly going forward and
> am happy to address reviewer issues in public on this repository.

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
[AUTHOR: if the QGIS repository page shows a download/usage count, cite it here —
e.g. "The plugin has been downloaded N times via the QGIS repository." Only include
a real figure.]

### B.4 Context and provenance
[AUTHOR: provide the true background in 2–4 sentences. Useful points to include if
accurate:
 - When and over what period the software was actually developed (private repo,
   local development, etc.).
 - Whether it arose from MSc/PhD study, a course, or professional surveying work,
   and at which institution (Karadeniz Technical University).
 - Whether it is used in teaching or in real survey projects.
Do not fabricate; if the work was developed over a short, intense period, say so
plainly — substance, not duration, is what matters.]

### B.5 Maintenance commitment
The GitHub repository is now the canonical, public home of the project. I will
conduct the JOSS review openly here, respond to issues and pull requests, and
continue maintenance and the published roadmap (network sketching, batch
processing, additional export formats). [AUTHOR: adjust to reflect your real
intentions; optionally name any co-maintainers or none.]

---

## C. Housekeeping to do BEFORE posting this (so the record is consistent)
An editor may cross-check dates, so reconcile these first:
1. **Date inconsistency.** `CHANGELOG.md` lists releases as 2024-10/11/12
   ("2024-…-XX"), while `metadata.txt`/`CITATION.cff` and the repo activity point to
   2026, and `paper.md` is dated 2026. Pick the correct real dates and make the
   CHANGELOG, CITATION.cff, metadata.txt, and paper.md agree. Replace the "-XX" day
   placeholders with actual days.
2. **Grow a little public signal where cheap and honest.** Before/at submission,
   create real **tagged GitHub Releases** matching the versions, enable Issues, and
   (optionally) open a couple of genuine tracking issues for the roadmap items. This
   converts "14 commits in two days" into a repository that visibly has releases and
   an issue tracker.
3. **Keep claims truthful.** This statement only asserts things visible in the repo
   plus whatever you fill into the [AUTHOR] blanks — keep it that way.
