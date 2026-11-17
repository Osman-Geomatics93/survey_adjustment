"""Regenerate the example observations from their design geometry.

Every observation is recomputed from the coordinates in the example's
``points.csv`` (taken as the true geometry) and perturbed by a random error
drawn from the observation's own stated precision: ``sigma`` for scalar
observations and the full 3x3 covariance for GNSS baselines. The observation
IDs, sigmas, covariances, column order and decimal places are kept, so only
the observed values change.

Conventions (those of the original data):
- distances are horizontal;
- directions are azimuths, clockwise from north (set orientation zero);
- angles are clockwise from the backsight to the foresight:
  az(at -> to) - az(at -> from);
- angular values are in degrees, angular sigmas in radians.

Usage (from the repository root; requires NumPy):
    python examples/generate_observations.py [--seed 2026]

The output is deterministic for a given seed. Each example draws from its own
random stream, so editing one example does not change the others.
"""

import argparse
import csv
import math
import zlib
from pathlib import Path

import numpy as np

EXAMPLES_DIR = Path(__file__).resolve().parent
DEFAULT_SEED = 2026


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def decimals(rows, column):
    """Largest number of decimal places used in a column."""
    return max(len(r[column].split(".")[1]) if "." in r[column] else 0 for r in rows)


def azimuth_deg(points, from_id, to_id):
    (e1, n1, _), (e2, n2, _) = points[from_id], points[to_id]
    return math.degrees(math.atan2(e2 - e1, n2 - n1)) % 360.0


def load_points(example):
    _, rows = read_csv(example / "points.csv")
    return {r["point_id"]: (float(r["easting"]), float(r["northing"]), float(r["height"]))
            for r in rows}


def perturb_scalar(path, rng, column, model, angular=False):
    """Replace ``column`` by model(row) + N(0, sigma) noise."""
    fieldnames, rows = read_csv(path)
    places = decimals(rows, column)
    for row in rows:
        sigma = float(row["sigma"])
        if angular:
            sigma = math.degrees(sigma)
        value = model(row) + rng.normal(0.0, sigma)
        if angular:
            value %= 360.0
        row[column] = f"{value:.{places}f}"
    write_csv(path, fieldnames, rows)
    return len(rows)


def perturb_baselines(path, rng, points):
    fieldnames, rows = read_csv(path)
    places = {c: decimals(rows, c) for c in ("dE", "dN", "dH")}
    for row in rows:
        c = {k: float(row[f"cov_{k}"]) for k in ("EE", "EN", "EH", "NN", "NH", "HH")}
        cov = np.array([[c["EE"], c["EN"], c["EH"]],
                        [c["EN"], c["NN"], c["NH"]],
                        [c["EH"], c["NH"], c["HH"]]])
        truth = np.subtract(points[row["to_point"]], points[row["from_point"]])
        value = truth + rng.multivariate_normal(np.zeros(3), cov)
        for k, v in zip(("dE", "dN", "dH"), value):
            row[k] = f"{v:.{places[k]}f}"
    write_csv(path, fieldnames, rows)
    return len(rows)


def generate(example, seed):
    rng = np.random.default_rng([seed, zlib.crc32(example.name.encode())])
    points = load_points(example)
    counts = {}

    def horizontal_distance(row):
        (e1, n1, _), (e2, n2, _) = points[row["from_point"]], points[row["to_point"]]
        return math.hypot(e2 - e1, n2 - n1)

    def direction(row):
        return azimuth_deg(points, row["from_point"], row["to_point"])

    def angle(row):
        return (azimuth_deg(points, row["at_point"], row["to_point"])
                - azimuth_deg(points, row["at_point"], row["from_point"])) % 360.0

    def height_difference(row):
        return points[row["to_point"]][2] - points[row["from_point"]][2]

    # Fixed file order keeps the random stream, and so the output, reproducible.
    jobs = [
        ("distances.csv", lambda p: perturb_scalar(p, rng, "distance", horizontal_distance)),
        ("directions.csv", lambda p: perturb_scalar(p, rng, "direction", direction, angular=True)),
        ("angles.csv", lambda p: perturb_scalar(p, rng, "angle", angle, angular=True)),
        ("height_differences.csv", lambda p: perturb_scalar(p, rng, "dh", height_difference)),
        ("baselines.csv", lambda p: perturb_baselines(p, rng, points)),
    ]
    for name, job in jobs:
        path = example / name
        if path.exists():
            counts[name] = job(path)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    for example in sorted(p for p in EXAMPLES_DIR.iterdir() if (p / "points.csv").exists()):
        counts = generate(example, args.seed)
        summary = ", ".join(f"{name}: {n}" for name, n in counts.items())
        print(f"{example.name}: {summary}")


if __name__ == "__main__":
    main()
