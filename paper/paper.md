---
title: 'Survey Adjustment & Network Analysis: rigorous least-squares adjustment and quality control of survey networks inside QGIS'
tags:
  - QGIS
  - surveying
  - geodesy
  - least squares
  - network adjustment
  - reliability analysis
  - GNSS
  - Python
authors:
  - name: Osman Osama Ahmed Ibrahim
    orcid: 0009-0003-2594-6310
    corresponding: true
    affiliation: "1, 2"
affiliations:
  - index: 1
    name: Karadeniz Technical University, Trabzon, TÃ¼rkiye
    ror: 03z8fyr40
  - index: 2
    name: Hydraulics Research Center, Ministry of Irrigation and Water Resources, Wad Madani, Sudan
date: 17 November 2026
bibliography: paper.bib
---

# Summary

Every coordinate in a control survey, a deformation-monitoring network, or a
cadastral framework is derived from redundant measurements â€” distances,
directions, angles, levelled height differences, and GNSS baselines â€” that never
agree exactly. Least-squares adjustment reconciles them, delivering the most
probable coordinates together with statistical evidence of how *precise* and how
*trustworthy* those coordinates are [@ghilani2017; @mikhail1976].

`Survey Adjustment & Network Analysis` brings this workflow into QGIS
[@qgis2026]. It registers a Processing provider with five tools: a network
validator and four least-squares solvers for 1D leveling, 2D classical networks,
3D GNSS baseline networks with full per-baseline covariance, and mixed networks
that combine all three observation families in a single estimation. Each run
returns adjusted points, error ellipses, residual vectors, and a residual table as
GeoPackage layers that load directly onto the map, plus an HTML report for people
and a JSON report for machines. The numerical engine depends only on NumPy
[@harris2020] and imports nothing from QGIS, so it is equally usable as a
stand-alone Python library.

# Statement of need

Adjustment is not complete when coordinates are computed: practitioners must also
show that the observations are statistically consistent, that gross errors would
have been detected, and how large an undetected error could distort the result.
This quality-control layer â€” the global model test, data snooping, redundancy
numbers, and internal and external reliability of the Delft school
[@baarda1968; @teunissen2024; @koch1999] â€” is standard in commercial adjustment
packages but largely absent from the open GIS ecosystem. Three groups are
affected:

- **Surveyors and geomatics engineers** whose data already live in QGIS must
  export observations to a separate, usually licensed, package and re-import the
  results by hand.
- **Educators and students** of adjustment computations lack an open engine
  whose equations, intermediate matrices, and diagnostics can be inspected and
  reproduced alongside a textbook.
- **Researchers** building automated geospatial pipelines â€” for example,
  repeated monitoring epochs â€” need an adjustment component that can be scripted
  and versioned rather than driven through a GUI.

The plugin addresses these needs with (i) one engine spanning 1D, 2D, 3D GNSS,
and mixed adjustment; (ii) the full Delft-school diagnostic suite together with
robust iteratively reweighted least squares (IRLS) using Huber [@huber1964],
Danish [@krarup1980], and IGG-III [@yang1994; @yang2002] weight functions;
(iii) outputs that
are native GIS layers and algorithms that run from the Processing toolbox, the
graphical modeler, batch mode, or the Python console; and (iv) reproducible
results, since every JSON report carries the full parameter covariance matrix and
a snapshot of the settings used.

# State of the field

| Software | Licence | Relation to QGIS | Spatial output |
|:--|:--|:--|:--|
| STAR\*NET [@starnet], MOVE3 [@move3], Trimble Business Center [@tbc], Leica Infinity [@infinity] | Commercial | None (stand-alone) | Proprietary formats |
| GNU GaMa [@cepek2002] | Open source | None (command line, XML) | Tabular reports |
| `SurveyingCalculation` [@surveyingcalculation] | Open source | Plugin calling a GaMa binary; QGIS 2.x | Layers |
| `QNET` [@qnet2025] | Open source | Dialog-based plugin; QGIS 3.34+ | Point layer + text report |
| **This work** | **Open source (GPL-2.0)** | **Processing provider, QGIS 3.22+** | **GeoPackage layers + JSON/HTML** |

Commercial packages are mature and reliability-aware but are licensed and
stand-alone. GNU GaMa is the authoritative open-source adjustment engine, and we
considered contributing to it rather than building anew. We did not, for a
practical reason: a QGIS plugin that drives GaMa must ship or locate a compiled
C++ binary for every platform, the integration path that left
`SurveyingCalculation` tied to QGIS 2.x. A pure-Python engine installs from the
official plugin repository with no compilation, can be read and modified by the
geomatics users who rely on it, and exchanges data with QGIS in memory rather
than through XML files. The recent `QNET` plugin [@qnet2025] adjusts classical
and GNSS observations with free-network datum and robust weighting from a
dialog; its documented outputs (adjusted coordinates, sigmas, error ellipses,
and residuals) do not include the global model test, standardized residuals,
or reliability measures, and it is not exposed as Processing algorithms. Those
two gaps, Delft-school quality control and a scriptable Processing interface
that can be chained in models and tested, are the ones this work fills.

# Software design

The code base (about 11,000 lines of Python) is split into a QGIS-independent
**core** and a thin **integration layer**. The core contains the observation and
network models, the solvers, the statistics and reliability modules, a
constraint-health validator, and the report writers. The integration layer
registers the Processing algorithms, imports the QGIS API lazily, and converts
between QGIS data structures and core objects. Four design decisions shape the
software:

1. **QGIS-free core.** Because the core imports nothing from QGIS, its 212
   automated tests run in continuous integration on Python 3.9â€“3.12 without a
   QGIS installation, and the same solvers can be called from notebooks or
   servers.
2. **NumPy only.** SciPy is not reliably available in every QGIS distribution,
   so the chi-square distribution is implemented from regularized
   incomplete-gamma functions [@press2007] and the standard-normal quantile is
   taken from the Python standard library. This trades a small amount of
   numerical code for zero-friction installation.
3. **Shared parameter indexing.** All solvers use a common index of unknowns
   (coordinates and one orientation unknown per direction set), so the mixed
   solver reuses the observation models of the specialised solvers instead of
   duplicating them, and new observation types can be added in one place.
4. **Validation before estimation.** A constraint-health analysis checks
   connectivity, datum definition, and degrees of freedom before any matrix is
   formed, and reports problems as actionable messages; an optional auto-datum
   applies minimal constraints for exploratory runs.

# Mathematical core

Each solver linearizes the observation equations about approximate parameters
$x_0$ and minimizes $v^{\mathsf{T}} P v$, where $v = w - A\,\delta x$,
$w = l - f(x_0)$ is the misclosure vector, $A$ the Jacobian, and
$P = Q_{ll}^{-1}$ the weight matrix â€” diagonal for classical and leveling
observations, with dense $3\times3$ blocks for GNSS baselines [@leick2015]. The
Gaussâ€“Newton iteration solves

$$N\,\delta x = u,\quad N = A^{\mathsf{T}} P A,\quad u = A^{\mathsf{T}} P w,\qquad \hat{\sigma}_0^2 = \frac{v^{\mathsf{T}} P v}{r},\quad \Sigma_{\hat{x}} = \hat{\sigma}_0^2\, N^{-1},$$

with redundancy $r = m - n$. From the residual cofactor matrix
$Q_{vv} = P^{-1} - A N^{-1} A^{\mathsf{T}}$ the plugin computes the two-sided
chi-square global test, standardized residuals
$w_i = v_i / (\sigma_0 \sqrt{(Q_{vv})_{ii}})$ for data snooping, redundancy
numbers $r_i = (Q_{vv} P)_{ii}$ [@baarda1968], the minimal detectable bias
$\mathrm{MDB}_i = (k_\alpha + k_\beta)\,\hat{\sigma}_0\,\sigma_i / \sqrt{r_i}$,
and its effect on the coordinates (external reliability) [@teunissen2024]. Error
ellipses follow from the eigen-decomposition of each point's $2\times2$
covariance. Robust IRLS rescales weights by $u(|w_i|)$ in an outer loop. Its
standardized residuals use the a-priori $\sigma_0$ and the cofactors of the
ordinary least-squares solution, so an outlier can neither mask itself nor
regain full weight once down-weighted, and the redescending Danish and IGG-III
functions start from a converged Huber solution, so that a gross error smeared
over many residuals does not cause good observations to be rejected.

# Validation and example usage

The repository ships five worked datasets â€” 1D leveling, a 2D traverse, 2D
trilateration, a 3D GNSS network, and a mixed network â€” each with committed
JSON, HTML, and GeoPackage outputs. Their observations are generated
reproducibly from the design geometry, with random errors drawn from each
observation's stated precision, and all five adjustments pass the global
model test. \autoref{fig:traverse} shows the result for the 2D traverse. The
tools run from the Processing toolbox or from Python:

```python
import processing

processing.run("survey_adjustment:adjust_network_2d", {
    "POINTS_CSV": "points.csv",
    "DISTANCES_CSV": "distances.csv",
    "DIRECTIONS_CSV": "directions.csv",
    "ROBUST_METHOD": 1,              # Huber IRLS
    "COMPUTE_RELIABILITY": True,
    "OUTPUT_JSON": "adjustment.json",
    "OUTPUT_HTML": "report.html",
})
```

![Adjustment of the example 2D traverse (two fixed control points, three new
points, 23 distance, direction, and angle observations). (a) Adjusted points and
95 % error ellipses, drawn 5,000 times enlarged; labels give the semi-axes.
(b) Absolute standardized residuals used for data snooping; no observation
exceeds the outlier threshold. Generated from the plugin's JSON report by
`paper/make_figure.py`.\label{fig:traverse}](figure.png)

Tutorials for each solver, the input formats, and the underlying theory are
published as a documentation site built from the repository.

# Research impact statement

The plugin has been distributed through the official QGIS plugin repository
since January 2026 and had been downloaded more than 1,300 times across three
releases by November 2026. It is in use in two settings:

- **Training.** At the Hydraulics Research Center (Sudan), where the author
  trains engineers and researchers in surveying and GIS, the plugin was used
  in March 2026 to teach least-squares network adjustment and quality control
  to 15 engineers, who adjusted their own total-station and leveling data.
- **Practice.** It was used to adjust the leveling and GNSS network of the
  Gezira canal control survey (Wad Madani, Sudan, 2026), with 12 points and
  40 observations, which provides control for canal rehabilitation.

Its significance lies in making reliability-aware adjustment â€” previously tied
to licensed software â€” available where geospatial data are already managed, and
in a form that supports reproducible research: the core can be embedded in
automated monitoring pipelines, and each JSON report contains everything needed
to re-derive and audit a published result.

# AI usage disclosure

Anthropic's Claude, used through the Claude Code assistant, was used during
development of the software: Claude Opus 4.5 and Claude Sonnet 4.5 (January to
June 2026) assisted with implementing parts of the solvers and the QGIS
integration layer, writing unit tests, and drafting the README. Claude Opus 5.5 (November 2026) was used to restructure and
copy-edit this paper, to write the figure script and the example-data
generator, to write test suites for the 1D, 3D, mixed, and robust solvers, and
to diagnose and fix the defects those tests revealed (Python 3.9
compatibility, inconsistent scaling of standardized residuals, the
significance level of the global test, and non-converging robust
estimation). The
author chose the adjustment models, algorithms, and software architecture,
reviewed and edited all AI-assisted code and text, and verified correctness
against the automated test suite and the worked examples.

# Acknowledgements

The author thanks the QGIS and NumPy communities, whose open-source projects
this plugin builds upon. No external financial support was received for this
work.

# References
