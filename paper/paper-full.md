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

# Mathematical formulation

All four solvers share a common weighted least-squares core built on the Gauss–Markov model [@ghilani2017; @mikhail1976; @koch1999]. This section states the equations the implementation actually evaluates. Coordinates are projected easting/northing in metres and azimuths follow the surveying convention (north $=0$, clockwise positive), so that $\alpha_{ij}=\operatorname{atan2}(E_j-E_i,\,N_j-N_i)$.

## Functional model

Each measurement is related to the unknown coordinates by an observation equation:

$$d_{ij} = \sqrt{(E_j-E_i)^2 + (N_j-N_i)^2} \qquad \text{(distance)},$$

$$r_{ij} = \alpha_{ij} + \omega_s \qquad \text{(direction, with a per-setup orientation unknown } \omega_s),$$

$$\beta_{jik} = \alpha_{jk} - \alpha_{ji} \qquad \text{(angle at station } j \text{ from } i \text{ to } k),$$

$$\Delta h_{ij} = H_j - H_i \qquad \text{(levelled height difference)},$$

$$\mathbf{b}_{ij} = [\,\Delta E,\ \Delta N,\ \Delta H\,]^{\mathsf{T}} = [\,E_j - E_i,\ N_j - N_i,\ H_j - H_i\,]^{\mathsf{T}} \qquad \text{(GNSS baseline)} .$$

Angular misclosures are wrapped to $(-\pi, \pi]$, and each GNSS baseline contributes a full $3\times3$ covariance block [@leick2015]. The mixed solver assembles all of the above into a single system.

## Linearized weighted least squares

Linearizing about approximate parameters $x_0$ yields the misclosure vector $w = l - f(x_0)$ and the Jacobian (design matrix) $A = \partial f / \partial x \big|_{x_0}$, with residuals $v = w - A\,\delta x$. With the weight matrix $P = Q_{ll}^{-1}$ — diagonal entries $p_i = 1/\sigma_i^2$ for uncorrelated observations and dense $3\times3$ blocks $\Sigma_b^{-1}$ for GNSS baselines — minimizing $v^{\mathsf{T}} P v$ gives the normal equations

$$N\,\delta x = u, \qquad N = A^{\mathsf{T}} P A, \qquad u = A^{\mathsf{T}} P w ,$$

solved with `numpy.linalg.solve` (no explicit inverse); a singular $N$ is reported as a datum or connectivity defect. The solution is iterated (Gauss–Newton), updating $x_0 \leftarrow x_0 + \delta x$ until the largest coordinate and orientation corrections fall below tolerance. The a-posteriori variance factor and parameter covariance are

$$\hat{\sigma}_0^2 = \frac{v^{\mathsf{T}} P v}{r}, \qquad r = m - n, \qquad \Sigma_{\hat{x}} = \hat{\sigma}_0^2\, Q_{xx}, \qquad Q_{xx} = N^{-1},$$

where $m$ is the number of observations, $n$ the number of unknowns, and $r$ the redundancy; $Q_{xx}$ is obtained by solving $N Q_{xx} = I$ rather than forming an explicit inverse.

## Statistical testing and reliability

The cofactor matrix of the residuals is $Q_{vv} = P^{-1} - A N^{-1} A^{\mathsf{T}}$, of which the solver forms only the diagonal. Overall model fit is assessed with the two-sided global chi-square test [@baarda1968; @koch1999]

$$T = \frac{v^{\mathsf{T}} P v}{\sigma_0^2}, \qquad \text{accept if}\quad \chi^2_{\alpha/2,\,r} \le T \le \chi^2_{1-\alpha/2,\,r} ,$$

evaluated against the a-priori $\sigma_0^2$. Individual blunders are screened by Baarda data snooping, using standardized residuals and per-observation redundancy numbers,

$$w_i = \frac{v_i}{\sigma_0 \sqrt{(Q_{vv})_{ii}}}, \qquad r_i = (Q_{vv})_{ii}\, p_i, \qquad \sum_i r_i = r ,$$

where $|w_i|$ is compared with $k_\alpha = \Phi^{-1}(1 - \alpha_{\text{local}}/2)$ and a user threshold (default $3$). Internal reliability is reported as the Minimal Detectable Bias and external reliability as the induced coordinate shift [@baarda1968; @teunissen2006],

$$\mathrm{MDB}_i = (k_\alpha + k_\beta)\,\hat{\sigma}_0\,\frac{\sigma_i}{\sqrt{r_i}}, \qquad \delta x_i = Q_{xx}\,\big(p_i A_i^{\mathsf{T}}\big)\,\mathrm{MDB}_i ,$$

with $k_\beta = \Phi^{-1}(\text{power})$; the reported external-reliability metric is $\max_j |(\delta x_i)_j|$ over the coordinate components. The required $\chi^2$ and standard-normal quantiles are obtained without SciPy: the chi-square cumulative and quantile functions use regularized incomplete-gamma routines (series expansion and continued fraction) [@press2007], and the standard-normal quantile is taken from the Python standard library.

## Error ellipses and robust estimation

For each point the $2\times2$ posterior covariance is eigendecomposed, $\Sigma_p = V \operatorname{diag}(\lambda_1, \lambda_2) V^{\mathsf{T}}$ with $\lambda_1 \ge \lambda_2$, giving the confidence-ellipse semi-axes and orientation

$$a = k\sqrt{\lambda_1}, \qquad b = k\sqrt{\lambda_2}, \qquad k = \sqrt{\chi^2_{p,2}}, \qquad \theta = \operatorname{atan2}(v_E, v_N) ,$$

where $(v_E, v_N)$ is the major eigenvector and $p$ the confidence level. Robust estimation wraps the solver in an outer iteratively reweighted least-squares (IRLS) loop [@huber1964; @yang1999; @yang2001; @ghilani2017] that rescales each weight by a factor $u(|w_i|)$:

$$u_{\text{Hub}}(|w|) = \begin{cases} 1 & |w| \le c \\[2pt] c/|w| & |w| > c \end{cases}, \qquad u_{\text{Dan}}(|w|) = \begin{cases} 1 & |w| \le c \\[2pt] e^{-\left((|w|-c)/c\right)^2} & |w| > c \end{cases},$$

$$u_{\text{IGG3}}(|w|) = \begin{cases} 1 & |w| \le k_0 \\[3pt] \dfrac{k_0}{|w|}\left(\dfrac{k_1 - |w|}{k_1 - k_0}\right)^{2} & k_0 < |w| < k_1 \\[6pt] 0 & |w| \ge k_1 \end{cases},$$

so that $p_i \leftarrow p_i\, u(|w_i|)$. Reweighting uses the a-priori $\sigma_0$ to avoid the masking effect, in which a large blunder inflates $\hat{\sigma}_0$ and hides itself. An optional auto-datum routine applies minimal (inner) constraints with a full audit trail when a network is otherwise rank-deficient.

# Example usage

The repository ships five worked example datasets — 1D leveling, a 2D traverse, 2D trilateration, a 3D GNSS network, and a mixed network — each with committed expected JSON, HTML, and GeoPackage outputs under `examples/`. These serve both as a how-to walkthrough for new users and as de-facto validation fixtures that reproduce the documented results. After installing the plugin, a user loads the input tables, runs the relevant adjustment algorithm from the Processing toolbox, and inspects the adjusted points, error ellipses, and residual diagnostics that are added to the map canvas. Installation, input formats, and the full tool reference are documented in the project README; community guidelines, a security policy, and issue and pull-request templates are provided for contributors.

# Research impact

The software targets a recurring need in surveying and geodesy: an auditable, reproducible adjustment that lives inside the GIS where the spatial data already resides. By exposing the engine as a Processing provider with both machine-readable JSON (including the full covariance matrix and a settings snapshot) and spatial-layer outputs, it is positioned to be cited in two ways. First, as a teaching and methods reference: the open, from-scratch implementation of the Gauss–Markov model, data snooping, redundancy numbers, and Baarda reliability — together with the five committed example datasets and their expected outputs — provides a self-checkable resource for surveying and geodesy courses and for practitioners validating their own workflows. Second, as a reusable processing component: because the numerical core is importable outside QGIS and the QGIS algorithms are scriptable from the Python console and the model designer, adjustment with quality control can be embedded directly in reproducible geospatial pipelines and downstream studies, which is the typical pathway by which open geospatial tooling accrues citations. As an early-stage release the project does not yet report download or adoption metrics; the impact argument rests on the absence of a comparable free, QGIS-native, scriptable adjustment tool and on the reproducibility affordances described above.

# Acknowledgements

The author thanks the QGIS and NumPy communities, whose open-source projects this plugin builds upon. No external financial support was received for this work.

# References
