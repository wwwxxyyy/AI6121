#!/usr/bin/env python3
"""Render a colored PLY point cloud to three headless PCA-aligned views."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ply_io import read_ply, write_ply, fuse_points


def equalize_axes(axis, x: np.ndarray, y: np.ndarray) -> None:
    x_min, x_max = np.percentile(x, [1, 99])
    y_min, y_max = np.percentile(y, [1, 99])
    half_span = max(x_max - x_min, y_max - y_min) / 2
    axis.set_xlim((x_min + x_max) / 2 - half_span, (x_min + x_max) / 2 + half_span)
    axis.set_ylim((y_min + y_max) / 2 - half_span, (y_min + y_max) / 2 + half_span)
    axis.set_aspect("equal", adjustable="box")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Input colored PLY point cloud")
    parser.add_argument("output", type=Path, help="Output PNG preview")
    args = parser.parse_args()
    render_preview(args.input, args.output)


def render_preview(input_path: Path, output_path: Path) -> None:
    points, colors = read_ply(input_path)
    if len(points) == 0:
        raise ValueError(f"Point cloud is empty: {input_path}")

    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors = colors[finite] if len(colors) == len(finite) else np.ones_like(points)
    if not len(points):
        raise ValueError("Point cloud contains no finite points")

    center = np.median(points, axis=0)
    centered = points - center
    distances = np.linalg.norm(centered, axis=1)
    keep = distances <= np.percentile(distances, 99)
    centered = centered[keep]
    colors = colors[keep]

    # A 3x3 eigendecomposition also supports very small clouds without an NxN allocation.
    _, eigenvectors = np.linalg.eigh(centered.T @ centered)
    axes = eigenvectors[:, ::-1].T
    aligned = centered @ axes.T

    views = ((0, 1, "Front / principal plane"), (0, 2, "Top / depth"), (1, 2, "Side / depth"))
    figure, plot_axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#111111")
    for plot_axis, (x_idx, y_idx, title) in zip(plot_axes, views):
        plot_axis.set_facecolor("#111111")
        plot_axis.scatter(
            aligned[:, x_idx],
            aligned[:, y_idx],
            c=np.clip(colors, 0, 1),
            s=2.5,
            linewidths=0,
        )
        equalize_axes(plot_axis, aligned[:, x_idx], aligned[:, y_idx])
        plot_axis.set_title(title, color="white")
        plot_axis.tick_params(colors="#aaaaaa", labelsize=7)
        for spine in plot_axis.spines.values():
            spine.set_color("#555555")

    figure.suptitle(
        f"{input_path.name}: {len(centered):,} points (99% display subset)",
        color="white",
        y=0.98,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)
    print(f"Rendered {len(centered)} points to {output_path}")


if __name__ == "__main__":
    main()
