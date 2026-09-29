#!/usr/bin/env python3
"""Render the exported cloud through measured cameras with a point z-buffer.

No input photograph is composited into these images. Small pixel splats are
only a rendering choice; the PLY contains the actual reconstructed samples.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ply_io import read_ply, write_ply, fuse_points
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def project_cloud(xyz, rgb, K, camera, width, height, radius=1):
    X = xyz @ np.array(camera['R_world_to_camera']).T + camera['t_world_to_camera']
    keep = X[:, 2] > 0
    X, rgb = X[keep], rgb[keep]
    uv = X @ K.T
    uv = np.rint(uv[:, :2] / uv[:, 2:]).astype(int)
    keep = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    uv, z, rgb = uv[keep], X[keep, 2], rgb[keep]
    depth = np.full(width * height, np.inf)
    splats = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            x, y = uv[:, 0] + dx, uv[:, 1] + dy
            k = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            index = y[k] * width + x[k]
            np.minimum.at(depth, index, z[k])
            splats.append((index, k))
    pixels = np.full((width * height, 3), [16, 24, 32], np.uint8)
    for index, k in splats:
        near = z[k] <= depth[index]
        pixels[index[near]] = np.rint(rgb[k][near] * 255).astype(np.uint8)
    return pixels.reshape(height, width, 3)


def run(folder, title, stereo=False):
    cloud_name = 'stereo_cloud.ply' if stereo else 'reconstruction.ply'
    xyz, rgb = read_ply(folder / cloud_name)
    cameras = sorted(json.loads((folder / 'cameras.json').read_text()), key=lambda c: c['image_id'])
    info = json.loads((folder / 'intrinsics.json').read_text())
    K = np.loadtxt(folder / 'K.txt')
    w, h = info['width'], info['height']
    choices = [cameras[0], cameras[len(cameras)//2]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 8 if h > w else 4.8), facecolor='#101820')
    for ax, camera in zip(axes, choices):
        rendered = project_cloud(xyz, rgb, K, camera, w, h)
        ax.imshow(rendered)
        ax.set_title(f"Cloud view from frame {camera['image_id']:02d}", color='white', fontsize=12)
        ax.set_axis_off()
    method = 'multi-view checked stereo' if stereo else 'SIFT sparse'
    fig.suptitle(f'{title} | {method} | {len(xyz):,} points', color='white', fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .94))
    output = folder / ('stereo_views.png' if stereo else 'sparse_views.png')
    fig.savefig(output, dpi=160, facecolor=fig.get_facecolor()); plt.close(fig)
    assert cv2.imread(str(output)) is not None
    print(output)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('folder', type=Path)
    p.add_argument('--title', default='Reconstruction'); p.add_argument('--stereo', action='store_true')
    args = p.parse_args(); run(args.folder, args.title, args.stereo)
