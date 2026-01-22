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


def render_html_report(result: AdjustmentResult, title: str | None = None) -> str:
    """Render an :class:`AdjustmentResult` as a standalone HTML document.

    Automatically detects leveling (1D) vs 2D adjustments and formats accordingly.
    """
    is_leveling = _is_leveling_result(result)

    if title is None:
        title = "Leveling Adjustment Report" if is_leveling else "Survey Adjustment Report"

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
    parts.append(
        f"<div class='meta'>Success: <span class='{'ok' if result.success else 'bad'}'>{esc(result.success)}</span> | "
        f"Converged: <span class='{'ok' if result.converged else 'bad'}'>{esc(result.converged)}</span> | "
        f"Iterations: {esc(result.iterations)} | DOF: {esc(result.degrees_of_freedom)} | "
        f"σ₀²: {esc(result.variance_factor):s}</div>"
    )

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
    if is_leveling:
        # Leveling residuals with from/to columns
        parts.append(
            "<table><thead><tr>"
            "<th>ID</th><th>From</th><th>To</th><th>ΔH obs</th><th>ΔH comp</th><th>v (mm)</th><th>w</th>"
            "<th>r</th><th>MDB</th><th>Ext.Rel.</th><th>Flag</th>"
            "</tr></thead><tbody>"
        )
        for r in result.residual_details:
            cls = "flag" if (r.flagged or r.is_outlier_candidate) else ""
            v_mm = r.residual * 1000 if r.residual is not None else None
            r_val = f"{r.redundancy_number:.3f}" if r.redundancy_number is not None else "-"
            mdb_val = f"{r.mdb:.4f}" if r.mdb is not None and math.isfinite(r.mdb) else "-"
            ext_val = f"{r.external_reliability:.4f}" if r.external_reliability is not None and math.isfinite(r.external_reliability) else "-"
            parts.append(
                f"<tr class='{cls}'>"
                f"<td>{esc(r.obs_id)}</td>"
                f"<td>{esc(r.from_point or '')}</td><td>{esc(r.to_point or '')}</td>"
                f"<td>{r.observed:.4f}</td><td>{r.computed:.4f}</td>"
                f"<td>{v_mm:.2f}</td><td>{r.standardized_residual:.3f}</td>"
                f"<td>{r_val}</td><td>{mdb_val}</td><td>{ext_val}</td>"
                f"<td>{'⚑' if (r.flagged or r.is_outlier_candidate) else ''}</td>"
                "</tr>"
            )
        parts.append("</tbody></table>")
    else:
        # 2D residuals
        parts.append(
            "<table><thead><tr>"
            "<th>ID</th><th>Type</th><th>Observed</th><th>Computed</th><th>v</th><th>w</th>"
            "<th>Redundancy</th><th>MDB</th><th>Ext.Rel.</th><th>Flag</th>"
            "</tr></thead><tbody>"
        )
        for r in result.residual_details:
            cls = "flag" if (r.flagged or r.is_outlier_candidate) else ""
            parts.append(
                f"<tr class='{cls}'>"
                f"<td>{esc(r.obs_id)}</td><td>{esc(r.obs_type)}</td>"
                f"<td>{esc(r.observed)}</td><td>{esc(r.computed)}</td>"
                f"<td>{esc(r.residual)}</td><td>{esc(r.standardized_residual)}</td>"
                f"<td>{esc(r.redundancy_number)}</td><td>{esc(r.mdb)}</td><td>{esc(r.external_reliability)}</td>"
                f"<td>{'⚑' if (r.flagged or r.is_outlier_candidate) else ''}</td>"
                "</tr>"
            )
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
