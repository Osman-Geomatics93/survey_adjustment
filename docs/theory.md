# Theory & quality control

This page summarizes the mathematical model and the statistical diagnostics the
engine reports. The references cited here are listed in the paper's bibliography.

## Observation equations

Each solver linearizes its observation equations about approximate parameters
$\mathbf{x}_0$:

| Observation | Equation |
|:------------|:---------|
| 2D distance | $l_{ij} = \sqrt{(E_j - E_i)^2 + (N_j - N_i)^2}$ |
| 2D direction | $\alpha_{ij} = \operatorname{atan2}(E_j - E_i,\; N_j - N_i) + \omega_s$ |
| 2D angle | $l_{ikj} = \alpha_{kj} - \alpha_{ki}$ |
| 1D leveling | $l_{ij} = H_j - H_i$ |
| 3D GNSS baseline | $\mathbf{l}_{ij} = [E_j - E_i,\; N_j - N_i,\; H_j - H_i]^{\mathsf{T}}$ |

Here $\omega_s$ is the orientation unknown for direction set $s$. Azimuth is
measured from North, clockwise positive.

## Least-squares solution

With residual vector $\mathbf{v} = \mathbf{w} - \mathbf{A}\,\delta\mathbf{x}$,
misclosure $\mathbf{w} = \mathbf{l} - \mathbf{f}(\mathbf{x}_0)$, Jacobian
$\mathbf{A}$, and weight matrix $\mathbf{P} = \mathbf{Q}_{ll}^{-1}$ (diagonal
$p_i = 1/\sigma_i^2$, with dense $3\times3$ blocks for GNSS baselines), the
adjustment minimizes $\mathbf{v}^{\mathsf{T}}\mathbf{P}\mathbf{v}$ via the normal
equations:

\[
\mathbf{N}\,\delta\mathbf{x} = \mathbf{u}, \qquad
\mathbf{N} = \mathbf{A}^{\mathsf{T}}\mathbf{P}\mathbf{A}, \qquad
\mathbf{u} = \mathbf{A}^{\mathsf{T}}\mathbf{P}\mathbf{w}.
\]

The system is **solved**, not explicitly inverted (`numpy.linalg.solve`), and
Gauss–Newton iterates to convergence. The a-posteriori variance factor and
parameter covariance are

\[
\hat{\sigma}_0^2 = \frac{\mathbf{v}^{\mathsf{T}}\mathbf{P}\mathbf{v}}{r},
\qquad
\boldsymbol{\Sigma}_{\hat{x}} = \hat{\sigma}_0^2\,\mathbf{N}^{-1},
\]

with redundancy $r = m - n$ (observations minus unknowns).

## Global test (chi-square)

The two-sided global test compares
$T = \mathbf{v}^{\mathsf{T}}\mathbf{P}\mathbf{v} / \sigma_0^2$ against the
$\chi^2_{\alpha/2,\,r}$ and $\chi^2_{1-\alpha/2,\,r}$ critical values. The
chi-square distribution functions are implemented from regularized
incomplete-gamma routines (no SciPy dependency).

## Data snooping and reliability

Using the residual cofactor matrix
$\mathbf{Q}_{vv} = \mathbf{P}^{-1} - \mathbf{A}\mathbf{N}^{-1}\mathbf{A}^{\mathsf{T}}$,
Baarda data snooping flags observations by standardized residual and redundancy
number:

\[
w_i = \frac{v_i}{\sigma_0 \sqrt{(Q_{vv})_{ii}}}, \qquad
r_i = (Q_{vv})_{ii}\, p_i .
\]

Internal reliability is reported as the **Minimal Detectable Bias**

\[
\mathrm{MDB}_i = (k_\alpha + k_\beta)\,\hat{\sigma}_0\,\frac{\sigma_i}{\sqrt{r_i}},
\]

and external reliability as the impact of an undetected blunder on the adjusted
coordinates.

## Error ellipses

For each point's $2\times2$ covariance with eigenvalues
$\lambda_1 \geq \lambda_2$, the confidence-ellipse semi-axes are

\[
a = \sqrt{\lambda_1\,\chi^2_{p,2}}, \qquad b = \sqrt{\lambda_2\,\chi^2_{p,2}},
\]

where $p$ is the confidence level (default 0.95).

## Robust estimation (IRLS)

Robust estimation rescales weights by a Huber, Danish, or IGG-III factor
$\varphi(\lvert w_i \rvert)$ in an outer iteratively-reweighted least-squares
loop, using the a-priori $\sigma_0$ to avoid the masking effect.

| Method | Factor | Default parameters |
|:-------|:-------|:-------------------|
| Huber | $\varphi(t) = \min(1,\, c/t)$ | $c = 1.5$ |
| Danish | down-weights $t > c$ exponentially | $c = 2.0$ |
| IGG-III | piecewise, with hard rejection | $k_0 = 1.5,\; k_1 = 3.0$ |
