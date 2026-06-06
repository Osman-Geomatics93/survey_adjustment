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
    ror: 03z8fyr40
date: 6 June 2026
bibliography: paper.bib
---

# Summary

Surveying networks are determined from redundant, error-bearing observations — distances, directions, angles, levelled height differences, and GNSS baselines — that must be reconciled by least-squares adjustment to obtain the most probable coordinates together with rigorous precision and reliability estimates [@ghilani2017; @mikhail1976]. `Survey Adjustment & Network Analysis` is a QGIS plugin [@qgis2024] that delivers this capability as a Processing provider exposing five tools: a network validator and four solvers for 1D leveling, 2D classical networks (distances, directions, angles), 3D GNSS baseline networks with full per-baseline covariance, and unified mixed networks. Results are returned as JSON and HTML reports and as GeoPackage layers — adjusted points, error ellipses, residual vectors, and a residual table — for immediate mapping in QGIS. The plugin also computes the quality-control diagnostics expected of professional software (a chi-square global test, standardized residuals, redundancy numbers, Baarda internal and external reliability, and robust estimation), and its numerical core depends only on NumPy [@harris2020] and is usable as a standalone Python library outside QGIS.

# Statement of need

Rigorous least-squares network adjustment that is simultaneously free, open source, QGIS-native, and scriptable is not currently available. Surveyors and geomatics engineers working in QGIS must export observations to a separate, often paid, package or fall back to command-line tools that emit tables rather than spatial layers; students and educators lack an open, documented engine whose mathematics and diagnostics they can inspect and reproduce; and researchers have no scriptable adjustment component for automated geospatial pipelines. This plugin fills that gap with a single Processing provider — usable from the toolbox, the Python console, and the graphical model designer — spanning 1D, 2D, 3D GNSS, and mixed adjustment and returning results as GIS layers. It pairs this with a quality-control layer usually found only in commercial tools: robust iteratively reweighted least squares (IRLS) with Huber, Danish, and IGG-III weight functions [@huber1964; @yang1999; @yang2001] and the chi-square test, data snooping, redundancy numbers, and internal and external reliability of the Delft testing school [@baarda1968; @teunissen2006; @koch1999]. Reproducibility is emphasized through a JSON report carrying the full covariance matrix and a settings snapshot, and through a NumPy-only dependency that installs cleanly in a stock QGIS.

# State of the field

Mature commercial packages (MicroSurvey STAR\*NET, Sweco MOVE3, Trimble Business Center, Leica Infinity) are reliability-aware but paid, standalone, and not QGIS-integrated. GNU GaMa is a free, authoritative engine [@cepek2002] but is command-line and XML oriented and emits tables rather than spatial layers; within QGIS, the older `SurveyingCalculation` plugin wraps GaMa for QGIS 2.x via an external binary, and the more recent `QNET` is a GUI dialog rather than a scriptable Processing provider. This plugin was therefore built as a new, dependency-light Processing provider so that the adjustment, its diagnostics, and its spatial outputs live inside the QGIS data model and scripting surface where the target users already work.

# Implementation and methods

A UI-independent, NumPy-only core (observation models, Gauss–Newton solver, statistics, reliability, and report writers) is wrapped by a thin QGIS layer that registers the Processing algorithms and imports QGIS lazily, so the core runs and can be tested without QGIS. SciPy is deliberately avoided: the chi-square distribution functions are implemented from regularized incomplete-gamma routines [@press2007], with the standard-normal quantile taken from the Python standard library.

Each solver linearizes the observation equations about approximate parameters $x_0$ and minimizes $v^{\mathsf{T}} P v$, where $v = w - A\,\delta x$ is the residual vector, $w = l - f(x_0)$ the misclosure, $A$ the Jacobian, and $P = Q_{ll}^{-1}$ the weight matrix (diagonal $p_i = 1/\sigma_i^2$, with dense $3\times3$ blocks for GNSS baselines) [@ghilani2017; @koch1999]. This yields the normal equations and a-posteriori statistics

$$N\,\delta x = u,\quad N = A^{\mathsf{T}} P A,\quad u = A^{\mathsf{T}} P w,\qquad \hat{\sigma}_0^2 = \frac{v^{\mathsf{T}} P v}{r},\quad \Sigma_{\hat{x}} = \hat{\sigma}_0^2\, N^{-1},$$

with redundancy $r = m - n$ (observations minus unknowns); $N$ is solved, not inverted, via `numpy.linalg.solve`, and Gauss–Newton iterates to convergence. Quality control uses the residual cofactor matrix $Q_{vv} = P^{-1} - A N^{-1} A^{\mathsf{T}}$: the two-sided global test compares $T = v^{\mathsf{T}} P v / \sigma_0^2$ against $\chi^2_{\alpha/2,\,r}$ and $\chi^2_{1-\alpha/2,\,r}$, while Baarda data snooping flags observations by standardized residual and redundancy number [@baarda1968],

$$w_i = \frac{v_i}{\sigma_0 \sqrt{(Q_{vv})_{ii}}}, \qquad r_i = (Q_{vv})_{ii}\, p_i .$$

Internal reliability is reported as the Minimal Detectable Bias $\mathrm{MDB}_i = (k_\alpha + k_\beta)\,\hat{\sigma}_0\,\sigma_i / \sqrt{r_i}$ and external reliability as its coordinate impact [@teunissen2006]. Error ellipses follow from the eigen-decomposition of each point's $2\times2$ covariance (semi-axes $a = \sqrt{\lambda_1\,\chi^2_{p,2}}$, $b = \sqrt{\lambda_2\,\chi^2_{p,2}}$), and robust estimation rescales weights by a Huber, Danish, or IGG-III factor $u(|w_i|)$ in an outer IRLS loop, using the a-priori $\sigma_0$ to avoid the masking effect [@huber1964; @yang1999; @yang2001]. GNSS baselines are processed with full $3\times3$ per-baseline covariance [@leick2015].

# Example usage

The repository ships five worked datasets — 1D leveling, a 2D traverse, 2D trilateration, a 3D GNSS network, and a mixed network — each with committed JSON, HTML, and GeoPackage outputs under `examples/` that serve as both a walkthrough and validation fixtures. A user loads the input tables, runs the relevant algorithm from the Processing toolbox, and inspects the adjusted points, error ellipses, and residual diagnostics added to the map canvas. Installation and input formats are documented in the README, with contributing guidelines, a security policy, and issue and pull-request templates for the community.

# Acknowledgements

The author thanks the QGIS and NumPy communities, whose open-source projects this plugin builds upon. No external financial support was received for this work.

# References
