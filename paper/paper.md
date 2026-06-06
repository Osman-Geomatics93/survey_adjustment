---
title: 'Survey Adjustment & Network Analysis: A QGIS plugin for least-squares adjustment of survey networks'
tags:
  - QGIS
  - surveying
  - geodesy
  - least squares
  - network adjustment
  - GNSS
  - Python
authors:
  - name: Osman Osama Ahmed Ibrahim
    orcid: 0009-0003-2594-6310
    corresponding: true
    affiliation: 1
affiliations:
  - index: 1
    name: Karadeniz Technical University, Trabzon, Türkiye
date: 6 June 2026
bibliography: paper.bib
---

# Summary

Surveyors and geodesists measure redundant quantities — distances, angles, directions, height differences, and GNSS baselines — to determine the coordinates of points in a network. Because every measurement carries error, these observations are inconsistent and must be reconciled by least-squares adjustment, which yields the most probable coordinates together with rigorous estimates of their precision and reliability [@ghilani2017; @mikhail1976].

`Survey Adjustment & Network Analysis` is a QGIS plugin that brings rigorous least-squares network adjustment directly into the QGIS geospatial environment [@qgis2024]. It is delivered as a QGIS Processing provider exposing five tools: a pre-flight network validator and four adjustment solvers covering 1D leveling, 2D classical networks (distances, directions, and angles), 3D GNSS baseline networks with full per-baseline covariance, and unified mixed networks combining all observation types. Adjustment results are returned both as machine-readable JSON and HTML reports and as GeoPackage layers — adjusted points, error ellipses, residual vectors, and a residual table — so that they can be mapped and analysed immediately within QGIS. Beyond coordinate estimation, the plugin computes the statistical quality-control diagnostics expected of professional adjustment software: a chi-square global model test, standardized residuals for outlier detection, redundancy numbers, and Baarda-style internal and external reliability measures, together with robust (outlier-resistant) estimation. The numerical core depends only on NumPy [@harris2020] and is importable and usable outside QGIS as a standalone Python module.

# Statement of need

Rigorous least-squares network adjustment that is simultaneously free, open source, QGIS-native, and scriptable is not currently available. Practicing surveyors and geomatics engineers who already work in QGIS must either export their observations to a separate, often paid, desktop package, or fall back to command-line tools that emit formatted tables rather than spatial layers. Students and educators lack an open, documented engine whose adjustment mathematics and quality-control diagnostics they can inspect and reproduce, and researchers have no scriptable, reproducible adjustment component that can be embedded directly in automated geospatial processing chains. This plugin addresses that gap by delivering a least-squares engine as a QGIS Processing provider — usable from the Processing toolbox, the Python console, and the graphical model designer — that spans 1D, 2D, 3D GNSS, and mixed adjustment in one tool and returns results as spatial layers that drop directly into a GIS map. It pairs that with a quality-control layer normally found only in high-end commercial tools: robust iteratively reweighted least squares (IRLS) with Huber, Danish, and IGG-III weight functions [@huber1964; @yang1999; @yang2001], a chi-square global test, standardized-residual data snooping, redundancy numbers, and internal and external reliability following the Delft testing school [@baarda1968; @teunissen2006; @koch1999]. It emphasizes reproducibility by emitting a JSON report carrying the full covariance matrix and a complete settings snapshot, and it depends on NumPy alone — the chi-square distribution functions are implemented from scratch (with the standard-normal quantile taken from the Python standard library) — lowering installation friction inside a stock QGIS.

The intended community is threefold: practicing surveyors and geomatics engineers who already use QGIS and want auditable adjustment with QC diagnostics without buying a separate package; students and educators in surveying and geodesy, for whom the open, documented mathematics and the bundled example datasets serve as a teaching and self-checking tool; and researchers who need a scriptable, reproducible engine that can be embedded in automated processing chains. The plugin operates on projected coordinates (easting/northing in metres) and does not perform datum or coordinate transformations.

# State of the field

Rigorous least-squares network adjustment is well served at two extremes and under-served in the middle. Mature commercial desktop packages such as MicroSurvey STAR\*NET, Sweco MOVE3, Trimble Business Center, and Leica Infinity are powerful and reliability-aware, but they are paid, run as standalone or vendor-locked applications, and do not integrate with QGIS or expose a scriptable, reproducible pipeline. At the other extreme, GNU GaMa is a free and authoritative adjustment engine [@cepek2002], but it is command-line and library oriented, consumes XML input, and produces formatted tables rather than spatial layers, with no native QGIS workflow. Within QGIS itself the options are thin: the older `SurveyingCalculation` plugin wraps GaMa but targets QGIS 2.x and requires a separately installed external binary, while the more recent `QNET` plugin is QGIS-native but is a GUI dialog rather than a Processing provider, depends on an external library, and emits text reports and point shapefiles rather than machine-readable JSON and publication-ready HTML.

The unmet need is therefore a free, open-source, QGIS-native least-squares engine delivered as a Processing provider that spans 1D, 2D, 3D GNSS, and mixed adjustment in one tool and returns results as spatial layers. Rather than contributing patches to GaMa (whose XML-and-tables design is intentionally engine-oriented and does not target an interactive GIS) or to the QGIS-2-era `SurveyingCalculation` wrapper, this project was built as a new, dependency-light Processing provider so that the adjustment, its quality-control diagnostics, and its spatial outputs live inside the QGIS data model and scripting surface where the target users already work.

# Software design

The plugin is organised as a UI-independent numerical core wrapped by a thin QGIS integration layer. The core (`core/`) implements the observation models, the Gauss–Newton solver, the statistical tests, the reliability analysis, and the report writers, and depends only on NumPy; it has no QGIS imports and can be imported and exercised as a standalone Python module outside QGIS. The QGIS integration layer (`qgis_integration/`) registers a Processing provider with five algorithms and adapts QGIS feature tables to and from the core's data structures; the heavy QGIS imports are loaded lazily so that the core remains usable where QGIS is absent. Each adjustment algorithm writes adjusted-point, error-ellipse, and residual-vector feature sinks plus a no-geometry residuals table, all of which can be directed into a single GeoPackage, alongside the JSON and HTML reports. Avoiding SciPy is a deliberate trade-off: the chi-square cumulative distribution and quantile functions are implemented directly so that the plugin installs cleanly against a stock QGIS Python environment, accepting a small amount of additional numerical code in exchange for a minimal dependency footprint.

# Mathematical background and functionality

Each solver linearizes the observation equations about approximate coordinates and solves the resulting Gauss–Markov model by weighted least squares using Gauss–Newton iteration [@ghilani2017; @koch1999]. Given the design matrix $A$, the observation weight matrix $P$ (the inverse of the cofactor matrix of the observations), and the misclosure vector $w$, the normal equations $N\,\hat{x} = u$ are formed explicitly with $N = A^{\mathsf{T}} P A$ and $u = A^{\mathsf{T}} P w$, and solved via `numpy.linalg.solve` rather than explicit inversion; a singular normal matrix is reported as a datum or connectivity problem. The parameter cofactor matrix $Q_{xx} = N^{-1}$ is computed by solving $N Q_{xx} = I$ rather than forming an explicit inverse, and the a-posteriori variance factor is

$$\hat{\sigma}_0^2 = \frac{v^{\mathsf{T}} P v}{r},$$

where $v$ are the residuals and $r$ is the redundancy (degrees of freedom). The posterior coordinate covariance is $\hat{\sigma}_0^2 Q_{xx}$. Directions carry a per-setup orientation unknown, and angular misclosures are wrapped to $(-\pi, \pi]$. GNSS baselines are handled with full $3\times3$ per-baseline covariance assembled block-diagonally.

For quality control the plugin evaluates the chi-square global model test on $v^{\mathsf{T}} P v / \sigma_0^2$, standardized residuals $w_i = v_i / (\hat{\sigma}_0 \sqrt{(Q_{vv})_{ii}})$ for data snooping, and redundancy numbers $r_i = (Q_{vv})_{ii}\, p_i$. Internal reliability is reported as the Minimal Detectable Bias using the common scalar Baarda approximation $\mathrm{MDB}_i = (k_\alpha + k_\beta)\,\hat{\sigma}_0\, \sigma_i / \sqrt{r_i}$, and external reliability as the corresponding maximum coordinate impact [@baarda1968; @teunissen2006]. The chi-square cumulative distribution and quantile functions are computed without SciPy, via custom regularized incomplete-gamma routines (series expansion and continued fraction), with the standard-normal quantile taken from the Python standard library. Error ellipses are formed by eigen-decomposition of the $2\times2$ coordinate covariance, scaled to a configurable confidence level by the chi-square quantile.

Robust estimation wraps the Gauss–Newton solver in an outer IRLS loop using the Huber, Danish, or IGG-III weight functions [@huber1964; @yang1999; @yang2001]; standardized residuals for reweighting use the a-priori variance factor to mitigate the masking effect, in which large outliers inflate $\hat{\sigma}_0$ and hide themselves. An optional auto-datum routine applies minimal (inner) constraints by fixing reference parameters when a network is otherwise unsolvable, recording an audit trail of every applied constraint. GNSS baseline processing follows standard practice for correlated three-component observations [@leick2015].

# Example usage

The repository ships five worked example datasets — 1D leveling, a 2D traverse, 2D trilateration, a 3D GNSS network, and a mixed network — each with committed expected JSON, HTML, and GeoPackage outputs under `examples/`. These serve both as a how-to walkthrough for new users and as de-facto validation fixtures that reproduce the documented results. After installing the plugin, a user loads the input tables, runs the relevant adjustment algorithm from the Processing toolbox, and inspects the adjusted points, error ellipses, and residual diagnostics that are added to the map canvas. Installation, input formats, and the full tool reference are documented in the project README; community guidelines, a security policy, and issue and pull-request templates are provided for contributors.

# Research impact

The software targets a recurring need in surveying and geodesy: an auditable, reproducible adjustment that lives inside the GIS where the spatial data already resides. By exposing the engine as a Processing provider with both machine-readable JSON (including the full covariance matrix and a settings snapshot) and spatial-layer outputs, it is positioned to be cited in two ways. First, as a teaching and methods reference: the open, from-scratch implementation of the Gauss–Markov model, data snooping, redundancy numbers, and Baarda reliability — together with the five committed example datasets and their expected outputs — provides a self-checkable resource for surveying and geodesy courses and for practitioners validating their own workflows. Second, as a reusable processing component: because the numerical core is importable outside QGIS and the QGIS algorithms are scriptable from the Python console and the model designer, adjustment with quality control can be embedded directly in reproducible geospatial pipelines and downstream studies, which is the typical pathway by which open geospatial tooling accrues citations. As an early-stage release the project does not yet report download or adoption metrics; the impact argument rests on the absence of a comparable free, QGIS-native, scriptable adjustment tool and on the reproducibility affordances described above.

# AI usage

Generative AI assistance (a large language model) was used to support drafting and editing of documentation and of this paper, and to assist with routine coding tasks. All software design decisions, the mathematical formulation and its implementation, and the verification of results against the committed example datasets were carried out and reviewed by the author, who is responsible for the correctness of the software and the content of this paper.

# Acknowledgements

The author thanks the QGIS and NumPy communities, whose open-source projects this plugin builds upon. No external financial support was received for this work.

# References
