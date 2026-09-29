#!/usr/bin/env python3
"""Optional RGB and explicit nearest-neighbor distance filtering of a PLY."""
import argparse
from pathlib import Path
import sys
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ply_io import read_ply, write_ply


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path); parser.add_argument('output', type=Path)
    parser.add_argument('--min-color', type=float, default=.06)
    parser.add_argument('--neighbors', type=int, default=30)
    parser.add_argument('--std-ratio', type=float, default=1.75)
    args = parser.parse_args()
    xyz,rgb = read_ply(args.input)
    keep = np.isfinite(xyz).all(axis=1) & (rgb.max(axis=1)>=args.min_color)
    xyz,rgb = xyz[keep],rgb[keep]
    if len(xyz)>2:
        d,_ = cKDTree(xyz).query(xyz,k=min(args.neighbors+1,len(xyz)),workers=1)
        score = d[:,1:].mean(axis=1); keep = score<=score.mean()+args.std_ratio*score.std()
        xyz,rgb = xyz[keep],rgb[keep]
    write_ply(args.output,xyz,rgb)
    print(f'Wrote {len(xyz)} points: {args.output}')


if __name__=='__main__': main()
