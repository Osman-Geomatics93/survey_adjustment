"""HTML report generation (QGIS-free).

Produces a standalone HTML report for an :class:`~survey_adjustment.core.results.adjustment_result.AdjustmentResult`.
"""

from __future__ import annotations

import html
import math
from dataclasses import asdict
from typing import Optional

from ..results.adjustment_result import AdjustmentResult, ResidualInfo, ErrorEllipse


def _is_leveling_result(result: AdjustmentResult) -> bool:
    """Detect if this is a 1D leveling adjustment based on observation types."""
    if not result.residual_details:
        return False
    # Check if all observations are height_diff type
    for r in result.residual_details:
        if r.obs_type != "height_diff":
            return False
    return True


def _is_gnss_result(result: AdjustmentResult) -> bool:
    """Detect if this is a 3D GNSS baseline adjustment based on observation types."""
    if not result.residual_details:
        return False
    # Check if all observations are gnss_baseline type
    for r in result.residual_details:
        if r.obs_type != "gnss_baseline":
            return False
    return True


def _is_mixed_result(result: AdjustmentResult) -> bool:
    """Detect if this is a mixed adjustment (classical + GNSS + leveling).

    Returns True if there are at least two different categories:
    - Classical (distance, direction, angle)
    - GNSS baselines
    - Leveling (height_diff)
    """
    if not result.residual_details:
        return False

    has_gnss = False
    has_classical = False
    has_leveling = False

    for r in result.residual_details:
        if r.obs_type == "gnss_baseline":
            has_gnss = True
        elif r.obs_type == "height_diff":
            has_leveling = True
        elif r.obs_type in ("distance", "direction", "angle"):
            has_classical = True

    # Mixed if at least two categories present
    categories = sum([has_gnss, has_classical, has_leveling])
    return categories >= 2


def _has_leveling_observations(result: AdjustmentResult) -> bool:
    """Check if result contains any leveling (height_diff) observations."""
    if not result.residual_details:
        return False
    for r in result.residual_details:
        if r.obs_type == "height_diff":
            return True
    return False


def _has_gnss_observations(result: AdjustmentResult) -> bool:
    """Check if result contains any GNSS baseline observations."""
    if not result.residual_details:
        return False
    for r in result.residual_details:
        if r.obs_type == "gnss_baseline":
            return True
    return False


def render_html_report(result: AdjustmentResult, title: str | None = None) -> str:
    """Render an :class:`AdjustmentResult` as a standalone HTML document.

    Automatically detects leveling (1D), GNSS (3D), or 2D adjustments and formats accordingly.
    """
    is_leveling = _is_leveling_result(result)
    is_gnss = _is_gnss_result(result)
    is_mixed = _is_mixed_result(result)
    has_gnss = _has_gnss_observations(result)
    has_leveling = _has_leveling_observations(result)

    if title is None:
        if is_leveling:
            title = "Leveling Adjustment Report"
        elif is_mixed:
            # Build dynamic title based on what's in the mix
            components = []
            has_classical = any(r.obs_type in ("distance", "direction", "angle") for r in result.residual_details)
            if has_classical:
                components.append("Classical")
            if has_gnss:
                components.append("GNSS")
            if has_leveling:
                components.append("Leveling")
            title = f"Mixed Adjustment Report ({' + '.join(components)})"
        elif is_gnss:
            title = "GNSS Baseline Adjustment Report"
        else:
            title = "Survey Adjustment Report"

    def esc(s: object) -> str:
        return html.escape(str(s))

    css = """
    body { font-family: Arial, sans-serif; margin: 24px; }
    h1 { margin-bottom: 4px; }
    .meta { color: #555; margin-bottom: 16px; }
    table { border-collapse: collapse; width: 100%; margin: 12px 0 24px 0; }
    th, td { border: 1px solid #ddd; padding: 6px 8px; font-size: 13px; }
    th { background: #f5f5f5; text-align: left; }
    .ok { color: #067d00; font-weight: bold; }
    .bad { color: #b00020; font-weight: bold; }
    .flag { background: #fff3cd; }
    .small { font-size: 12px; color: #666; }
    """

    parts: list[str] = []
    parts.append("<!doctype html>")
    parts.append("<html><head><meta charset='utf-8'>")
    parts.append(f"<title>{esc(title)}</title>")
    parts.append(f"<style>{css}</style>")
    parts.append("</head><body>")

    parts.append(f"<h1>{esc(title)}</h1>")
    meta_parts = [
        f"Success: <span class='{'ok' if result.success else 'bad'}'>{esc(result.success)}</span>",
        f"Converged: <span class='{'ok' if result.converged else 'bad'}'>{esc(result.converged)}</span>",
        f"Iterations: {esc(result.iterations)}",
        f"DOF: {esc(result.degrees_of_freedom)}",
        f"σ₀²: {esc(result.variance_factor):s}",
    ]
    # Add robust estimation info if enabled
    if result.robust_method:
        meta_parts.append(f"Robust: {esc(result.robust_method.upper())}")
    parts.append(f"<div class='meta'>{' | '.join(meta_parts)}</div>")

    if result.chi_square_test is not None:
        c = result.chi_square_test
        parts.append("<h2>Chi-square Global Test</h2>")
        parts.append("<table><thead><tr><th>Statistic</th><th>Lower</th><th>Upper</th><th>p-value</th><th>DOF</th><th>Passed</th></tr></thead><tbody>")
        parts.append(
            "<tr>"
            f"<td>{esc(c.test_statistic)}</td>"
            f"<td>{esc(c.critical_lower)}</td>"
            f"<td>{esc(c.critical_upper)}</td>"
            f"<td>{esc(c.p_value)}</td>"
            f"<td>{esc(c.degrees_of_freedom)}</td>"
            f"<td class='{'ok' if c.passed else 'bad'}'>{esc(c.passed)}</td>"
            "</tr>"
        )
        parts.append("</tbody></table>")

    # Robust estimation section
    if result.robust_method:
        parts.append("<h2>Robust Estimation</h2>")
        parts.append("<table><thead><tr><th>Parameter</th><th>Value</th></tr></thead><tbody>")
        parts.append(f"<tr><td>Method</td><td>{esc(result.robust_method.upper())}</td></tr>")
        parts.append(f"<tr><td>IRLS Iterations</td><td>{esc(result.robust_iterations)}</td></tr>")
        converged_class = 'ok' if result.robust_converged else 'bad'
        parts.append(f"<tr><td>Converged</td><td class='{converged_class}'>{esc(result.robust_converged)}</td></tr>")
        if result.robust_message:
            parts.append(f"<tr><td>Message</td><td>{esc(result.robust_message)}</td></tr>")

        # Count downweighted observations
        downweighted_count = 0
        min_weight = 1.0
        for r in result.residual_details:
            if r.weight_factor is not None and r.weight_factor < 0.999:
                downweighted_count += 1
                if r.weight_factor < min_weight:
                    min_weight = r.weight_factor

        parts.append(f"<tr><td>Downweighted Observations</td><td>{downweighted_count}</td></tr>")
        if downweighted_count > 0:
            parts.append(f"<tr><td>Minimum Weight Factor</td><td>{min_weight:.4f}</td></tr>")
        parts.append("</tbody></table>")

    parts.append("<h2>Adjusted Points</h2>")
    if is_leveling:
        # Leveling adjustment: show heights
        parts.append("<table><thead><tr><th>ID</th><th>Name</th><th>H (m)</th><th>σH (m)</th><th>Fixed</th></tr></thead><tbody>")
        for pid in sorted(result.adjusted_points.keys()):
            p = result.adjusted_points[pid]
            h_val = f"{p.height:.4f}" if p.height is not None else "-"
            sh_val = f"{p.sigma_height:.4f}" if p.sigma_height is not None else "-"
            fixed = "Yes" if p.fixed_height else "No"
            parts.append(
                "<tr>"
                f"<td>{esc(p.id)}</td><td>{esc(p.name)}</td>"
                f"<td>{h_val}</td><td>{sh_val}</td><td>{fixed}</td>"
                "</tr>"
            )
        parts.append("</tbody></table>")
    elif is_gnss or is_mixed or has_leveling:
        # 3D adjustment (GNSS, mixed, or with leveling): show E, N, H with all sigmas
        parts.append("<table><thead><tr><th>ID</th><th>Name</th><th>E</th><th>N</th><th>H</th><th>σE</th><th>σN</th><th>σH</th></tr></thead><tbody>")
        for pid in sorted(result.adjusted_points.keys()):
            p = result.adjusted_points[pid]
            h_val = f"{p.height:.4f}" if p.height is not None else "-"
            se_val = f"{p.sigma_easting:.4f}" if p.sigma_easting is not None else "-"
            sn_val = f"{p.sigma_northing:.4f}" if p.sigma_northing is not None else "-"
            sh_val = f"{p.sigma_height:.4f}" if p.sigma_height is not None else "-"
            parts.append(
                "<tr>"
                f"<td>{esc(p.id)}</td><td>{esc(p.name)}</td>"
                f"<td>{p.easting:.4f}</td><td>{p.northing:.4f}</td><td>{h_val}</td>"
                f"<td>{se_val}</td><td>{sn_val}</td><td>{sh_val}</td>"
                "</tr>"
            )
        parts.append("</tbody></table>")
    else:
        # 2D adjustment: show E, N coordinates
        parts.append("<table><thead><tr><th>ID</th><th>Name</th><th>E</th><th>N</th><th>σE</th><th>σN</th></tr></thead><tbody>")
        for pid in sorted(result.adjusted_points.keys()):
            p = result.adjusted_points[pid]
            se_val = f"{p.sigma_easting:.4f}" if p.sigma_easting is not None else "-"
            sn_val = f"{p.sigma_northing:.4f}" if p.sigma_northing is not None else "-"
            parts.append(
                "<tr>"
                f"<td>{esc(p.id)}</td><td>{esc(p.name)}</td>"
                f"<td>{p.easting:.4f}</td><td>{p.northing:.4f}</td>"
                f"<td>{se_val}</td><td>{sn_val}</td>"
                "</tr>"
            )
        parts.append("</tbody></table>")

    if result.error_ellipses:
        parts.append("<h2>Error Ellipses</h2>")
        parts.append("<table><thead><tr><th>Point</th><th>a (m)</th><th>b (m)</th><th>θ (deg)</th><th>Confidence</th></tr></thead><tbody>")
        for pid in sorted(result.error_ellipses.keys()):
            e = result.error_ellipses[pid]
            parts.append(
                "<tr>"
                f"<td>{esc(pid)}</td>"
                f"<td>{e.semi_major:.4f}</td>"
                f"<td>{e.semi_minor:.4f}</td>"
                f"<td>{math.degrees(e.orientation):.2f}</td>"
                f"<td>{e.confidence_level:.3f}</td>"
                "</tr>"
            )
        parts.append("</tbody></table>")

    parts.append("<h2>Residuals</h2>")
    has_robust = result.robust_method is not None
    if is_leveling:
        # Leveling residuals with from/to columns
        header = "<th>ID</th><th>From</th><th>To</th><th>ΔH obs</th><th>ΔH comp</th><th>v (mm)</th><th>w</th><th>r</th><th>MDB</th><th>Ext.Rel.</th>"
        if has_robust:
            header += "<th>u</th>"
        header += "<th>Flag</th>"
        parts.append(f"<table><thead><tr>{header}</tr></thead><tbody>")
        for r in result.residual_details:
            cls = "flag" if (r.flagged or r.is_outlier_candidate) else ""
            v_mm = r.residual * 1000 if r.residual is not None else None
            r_val = f"{r.redundancy_number:.3f}" if r.redundancy_number is not None else "-"
            mdb_val = f"{r.mdb:.4f}" if r.mdb is not None and math.isfinite(r.mdb) else "-"
            ext_val = f"{r.external_reliability:.4f}" if r.external_reliability is not None and math.isfinite(r.external_reliability) else "-"
            row = (
                f"<tr class='{cls}'>"
                f"<td>{esc(r.obs_id)}</td>"
                f"<td>{esc(r.from_point or '')}</td><td>{esc(r.to_point or '')}</td>"
                f"<td>{r.observed:.4f}</td><td>{r.computed:.4f}</td>"
                f"<td>{v_mm:.2f}</td><td>{r.standardized_residual:.3f}</td>"
                f"<td>{r_val}</td><td>{mdb_val}</td><td>{ext_val}</td>"
            )
            if has_robust:
                u_val = f"{r.weight_factor:.4f}" if r.weight_factor is not None else "-"
                row += f"<td>{u_val}</td>"
            row += f"<td>{'⚑' if (r.flagged or r.is_outlier_candidate) else ''}</td></tr>"
            parts.append(row)
        parts.append("</tbody></table>")
    elif is_gnss:
        # GNSS baseline residuals with from/to and 3D residual
        header = "<th>ID</th><th>From</th><th>To</th><th>Length (m)</th><th>|v| (mm)</th><th>w_max</th><th>r</th>"
        if has_robust:
            header += "<th>u</th>"
        header += "<th>Flag</th>"
        parts.append(f"<table><thead><tr>{header}</tr></thead><tbody>")
        for r in result.residual_details:
            cls = "flag" if (r.flagged or r.is_outlier_candidate) else ""
            v_mm = r.residual * 1000 if r.residual is not None else None
            r_val = f"{r.redundancy_number:.3f}" if r.redundancy_number is not None else "-"
            w_val = f"{r.standardized_residual:.3f}" if r.standardized_residual is not None else "-"
            row = (
                f"<tr class='{cls}'>"
                f"<td>{esc(r.obs_id)}</td>"
                f"<td>{esc(r.from_point or '')}</td><td>{esc(r.to_point or '')}</td>"
                f"<td>{r.observed:.3f}</td>"
                f"<td>{v_mm:.2f}</td><td>{w_val}</td>"
                f"<td>{r_val}</td>"
            )
            if has_robust:
                u_val = f"{r.weight_factor:.4f}" if r.weight_factor is not None else "-"
                row += f"<td>{u_val}</td>"
            row += f"<td>{'⚑' if (r.flagged or r.is_outlier_candidate) else ''}</td></tr>"
            parts.append(row)
        parts.append("</tbody></table>")
    elif is_mixed:
        # Mixed residuals: classical (scalar), GNSS (3D), and leveling (scalar) in one table
        header = "<th>ID</th><th>Type</th><th>From</th><th>To</th><th>Obs/Length</th><th>v/|v| (mm)</th><th>w</th><th>r</th>"
        if has_robust:
            header += "<th>u</th>"
        header += "<th>Flag</th>"
        parts.append(f"<table><thead><tr>{header}</tr></thead><tbody>")
        for r in result.residual_details:
            cls = "flag" if (r.flagged or r.is_outlier_candidate) else ""
            r_val = f"{r.redundancy_number:.3f}" if r.redundancy_number is not None else "-"
            w_val = f"{r.standardized_residual:.3f}" if r.standardized_residual is not None else "-"

            if r.obs_type == "gnss_baseline":
                # GNSS: show length and 3D residual magnitude
                v_mm = r.residual * 1000 if r.residual is not None else None
                obs_val = f"{r.observed:.3f}"
                v_str = f"{v_mm:.2f}" if v_mm is not None else "-"
            elif r.obs_type == "height_diff":
                # Leveling: show ΔH and residual in mm
                obs_val = f"{r.observed:.4f}"
                if r.residual is not None:
                    v_str = f"{r.residual * 1000:.2f}"  # mm
                else:
                    v_str = "-"
            else:
                # Classical: show value and scalar residual
                obs_val = f"{r.observed:.4f}" if isinstance(r.observed, float) else str(r.observed)
                if r.residual is not None:
                    if r.obs_type == "distance":
                        v_str = f"{r.residual * 1000:.2f}"  # mm
                    else:
                        v_str = f"{r.residual * 206265:.2f}"  # arcsec for angles
                else:
                    v_str = "-"

            row = (
                f"<tr class='{cls}'>"
                f"<td>{esc(r.obs_id)}</td>"
                f"<td>{esc(r.obs_type)}</td>"
                f"<td>{esc(r.from_point or r.at_point or '')}</td>"
                f"<td>{esc(r.to_point or '')}</td>"
                f"<td>{obs_val}</td>"
                f"<td>{v_str}</td><td>{w_val}</td>"
                f"<td>{r_val}</td>"
            )
            if has_robust:
                u_val = f"{r.weight_factor:.4f}" if r.weight_factor is not None else "-"
                row += f"<td>{u_val}</td>"
            row += f"<td>{'⚑' if (r.flagged or r.is_outlier_candidate) else ''}</td></tr>"
            parts.append(row)
        parts.append("</tbody></table>")
    else:
        # 2D residuals
        header = "<th>ID</th><th>Type</th><th>Observed</th><th>Computed</th><th>v</th><th>w</th><th>Redundancy</th><th>MDB</th><th>Ext.Rel.</th>"
        if has_robust:
            header += "<th>u</th>"
        header += "<th>Flag</th>"
        parts.append(f"<table><thead><tr>{header}</tr></thead><tbody>")
        for r in result.residual_details:
            cls = "flag" if (r.flagged or r.is_outlier_candidate) else ""
            row = (
                f"<tr class='{cls}'>"
                f"<td>{esc(r.obs_id)}</td><td>{esc(r.obs_type)}</td>"
                f"<td>{esc(r.observed)}</td><td>{esc(r.computed)}</td>"
                f"<td>{esc(r.residual)}</td><td>{esc(r.standardized_residual)}</td>"
                f"<td>{esc(r.redundancy_number)}</td><td>{esc(r.mdb)}</td><td>{esc(r.external_reliability)}</td>"
            )
            if has_robust:
                u_val = f"{r.weight_factor:.4f}" if r.weight_factor is not None else "-"
                row += f"<td>{u_val}</td>"
            row += f"<td>{'⚑' if (r.flagged or r.is_outlier_candidate) else ''}</td></tr>"
            parts.append(row)
        parts.append("</tbody></table>")

    if result.messages:
        parts.append("<h2>Messages</h2><ul>")
        for m in result.messages:
            parts.append(f"<li>{esc(m)}</li>")
        parts.append("</ul>")

    if result.error_message:
        parts.append("<h2>Errors</h2>")
        parts.append(f"<pre>{esc(result.error_message)}</pre>")

    parts.append("<div class='small'>Generated by Survey Adjustment core engine</div>")
    parts.append("</body></html>")
    return "\n".join(parts)


def save_html_report(path: str, result: AdjustmentResult, title: str | None = None) -> None:
    """Write an HTML report to disk."""
    html_str = render_html_report(result, title=title)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_str)
