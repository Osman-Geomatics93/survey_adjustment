"""Build the JOSS paper figure from the committed 2D traverse example output.

Usage (from the repository root):
    python paper/make_figure.py

Reads examples/2d_traverse/Result/Adjustment_Result.json and writes
paper/figure.png. Requires numpy and matplotlib.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

ROOT = Path(__file__).resolve().parent.parent
RESULT = ROOT / "examples" / "2d_traverse" / "Result" / "Adjustment_Result.json"
OUTPUT = ROOT / "paper" / "figure.png"

ELLIPSE_SCALE = 5000          # exaggeration factor for drawing the ellipses
CHI2_2DOF_95 = 5.991464547    # chi-square quantile, 2 degrees of freedom, 95 %

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
NETWORK = "#b5b4ae"
ELLIPSE = "#2a78d6"
TYPE_STYLE = {                # colour + marker, so type is never colour-alone
    "distance": ("#2a78d6", "o"),
    "direction": ("#eb6834", "s"),
    "angle": ("#1baf7a", "^"),
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.edgecolor": INK_2,
    "axes.labelcolor": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

data = json.loads(RESULT.read_text(encoding="utf-8"))
points = {p["id"]: p for p in data["adjusted_points"]}
residuals = data["residuals"]
threshold = data["metadata"]["settings_snapshot"]["outlier_threshold"]

fig, (ax_map, ax_res) = plt.subplots(
    1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [1.05, 1]}
)

# (a) Network, adjusted points and 95 % error ellipses ----------------------
edges = {tuple(sorted((r["from_point"], r["to_point"])))
         for r in residuals if r["obs_type"] in ("distance", "direction")}
for a, b in sorted(edges):
    pa, pb = points[a], points[b]
    ax_map.plot([pa["easting"], pb["easting"]], [pa["northing"], pb["northing"]],
                color=NETWORK, lw=1.2, zorder=1, solid_capstyle="round")

for pid, cov in data["point_covariances"].items():
    p = points[pid]
    if p["fixed_easting"] and p["fixed_northing"]:
        continue
    eigval, eigvec = np.linalg.eigh(np.asarray(cov))      # ascending order
    major, minor = np.sqrt(eigval[::-1] * CHI2_2DOF_95)
    angle = np.degrees(np.arctan2(eigvec[1, 1], eigvec[0, 1]))
    ax_map.add_patch(Ellipse(
        (p["easting"], p["northing"]), 2 * major * ELLIPSE_SCALE,
        2 * minor * ELLIPSE_SCALE, angle=angle, facecolor=ELLIPSE + "26",
        edgecolor=ELLIPSE, lw=1.5, zorder=2))
    p["ellipse_mm"] = (major * 1000, minor * 1000)

# Place each label on the side facing away from the network centroid.
centre_e = np.mean([p["easting"] for p in points.values()])
centre_n = np.mean([p["northing"] for p in points.values()])
for pid, p in points.items():
    fixed = p["fixed_easting"] and p["fixed_northing"]
    ax_map.scatter(p["easting"], p["northing"], s=46 if fixed else 26,
                   marker="^" if fixed else "o", color=INK if fixed else "white",
                   edgecolor=INK, linewidth=1.2, zorder=3)
    dx, dn = p["easting"] - centre_e, p["northing"] - centre_n
    if fixed:
        offset, ha, va = (0, -10), "center", "top"
    elif dx < 0:
        offset, ha, va = (-11, 0), "right", "center"
    else:
        offset, ha, va = (11, 0), "left", "center"
    ax_map.annotate(pid, (p["easting"], p["northing"]), xytext=offset,
                    textcoords="offset points", ha=ha, va=va, fontsize=8,
                    color=INK, fontweight="bold")
    if "ellipse_mm" in p:
        ax_map.annotate("{:.1f} × {:.1f} mm".format(*p["ellipse_mm"]),
                        (p["easting"], p["northing"]),
                        xytext=(offset[0], offset[1] + (10 if dn > abs(dx) else -10)),
                        textcoords="offset points", ha=ha, va=va, fontsize=6.5,
                        color=INK_2)

ax_map.set_aspect("equal")
ax_map.set_xlim(840, 1660)
ax_map.set_ylim(1910, 2340)
ax_map.set_xlabel("Easting (m)")
ax_map.set_ylabel("Northing (m)")
ax_map.grid(color=GRID, lw=0.6)
ax_map.set_axisbelow(True)
ax_map.set_title("(a) Adjusted network, 95 % error ellipses",
                 loc="left", fontsize=9, color=INK)
ax_map.legend(
    handles=[
        Line2D([], [], marker="^", ls="", color=INK, markersize=7,
               label="Fixed control point"),
        Line2D([], [], marker="o", ls="", markerfacecolor="white",
               markeredgecolor=INK, markersize=6, label="Adjusted point"),
        Line2D([], [], color=ELLIPSE, lw=1.5,
               label=f"Error ellipse (× {ELLIPSE_SCALE:,})"),
    ],
    loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, fontsize=6.5,
    frameon=False, handletextpad=0.4, columnspacing=1.0)

# (b) Standardized residuals against the data-snooping threshold ------------
x = 0
ticks = []
for obs_type, (colour, marker) in TYPE_STYLE.items():
    group = [r for r in residuals if r["obs_type"] == obs_type]
    xs = np.arange(x, x + len(group))
    ws = [abs(r["standardized_residual"]) for r in group]
    ax_res.vlines(xs, 0, ws, color=colour, lw=2, zorder=2)
    ax_res.scatter(xs, ws, color=colour, marker=marker, s=34, zorder=3,
                   edgecolor="white", linewidth=1.2,
                   label=f"{obs_type.capitalize()} (n = {len(group)})")
    ticks.append((xs.mean(), obs_type))
    x += len(group) + 2

ax_res.axhline(threshold, color=INK, lw=1, ls=(0, (4, 3)), zorder=1)
ax_res.annotate(f"Outlier threshold |w| = {threshold:g}", (x - 2.5, threshold),
                xytext=(0, 4), textcoords="offset points", ha="right",
                fontsize=7, color=INK)
ax_res.set_xticks([t for t, _ in ticks], [n.capitalize() for _, n in ticks])
ax_res.tick_params(axis="x", length=0)
ax_res.set_ylim(0, threshold * 1.25)
ax_res.set_ylabel("|Standardized residual|  $|w_i|$")
ax_res.grid(axis="y", color=GRID, lw=0.6)
ax_res.set_axisbelow(True)
ax_res.set_title(f"(b) Data snooping, {len(residuals)} observations",
                 loc="left", fontsize=9, color=INK)
ax_res.legend(loc="upper left", fontsize=7, frameon=False)

fig.tight_layout(w_pad=2.5)
fig.savefig(OUTPUT, dpi=300, facecolor="white")
print(f"wrote {OUTPUT.relative_to(ROOT)}")
