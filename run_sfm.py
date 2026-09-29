#!/usr/bin/env python3
"""Headless, reproducible entry point for the project's incremental SfM pipeline."""
import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import re
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")

import cv2
import numpy as np

from matching import FeatureMatcher, MatchConfig
from scripts.render_ply import render_preview

LOG = logging.getLogger("sfm_cli")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def estimate_intrinsics(width, height, focal_scale=1.2):
    if width <= 0 or height <= 0 or not np.isfinite(focal_scale) or focal_scale <= 0:
        raise ValueError("Image dimensions and focal scale must be positive and finite")
    f = focal_scale * max(width, height)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], dtype=float)


def sampled_frame_indices(count, source_fps, target_fps, max_frames=0):
    if count < 1 or not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("Video frame count / FPS is unavailable")
    step = source_fps / min(target_fps, source_fps)
    indices = np.unique(np.clip(np.rint(np.arange(0, count, step)).astype(int), 0, count - 1))
    if max_frames and len(indices) > max_frames:
        indices = indices[np.rint(np.linspace(0, len(indices) - 1, max_frames)).astype(int)]
    return indices


def prepare_inputs(args, output):
    frame_dir = output / "frames"
    frame_dir.mkdir()
    images, records = [], []
    source = (args.video or args.images).resolve()
    if not source.exists():
        raise FileNotFoundError(source)

    def store_frame(img, record):
        if img is None:
            raise ValueError(f"Could not decode {record}")
        if images and img.shape != images[0].shape:
            raise ValueError("All frames must have the same resolution")
        name = f"{len(images):04d}.png"
        if not cv2.imwrite(str(frame_dir / name), img):
            raise IOError(f"Could not write {name}")
        records.append({"image_id": len(images), "file": f"frames/{name}", **record})
        images.append(img)

    metadata = {"source": str(source), "kind": "video" if args.video else "images"}
    if args.video:
        capture = cv2.VideoCapture(str(source))
        try:
            if not capture.isOpened():
                raise ValueError(f"Cannot open video: {source}")
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = capture.get(cv2.CAP_PROP_FPS)
            indices = sampled_frame_indices(count, fps, args.fps, args.max_frames)
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, img = capture.read()
                if not ok:
                    raise ValueError(f"Cannot decode video frame {index}")
                store_frame(img, {"source_frame": int(index), "timestamp_seconds": float(index / fps)})
            metadata.update(source_fps=fps, source_frames=count, duration_seconds=count / fps,
                            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        finally:
            capture.release()
    else:
        if not source.is_dir():
            raise ValueError("--images must name a directory")
        files = [p for p in source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}]
        files.sort(key=lambda p: [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", p.name)])
        for p in files[:args.max_frames or None]:
            store_frame(cv2.imread(str(p)), {"source_file": str(p), "source_sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    if len(images) < 3:
        raise ValueError("At least three readable frames are required")
    height, width = images[0].shape[:2]
    calibration = args.calibration
    if calibration is None and args.images and (source / "K.txt").is_file():
        calibration = source / "K.txt"
    if calibration:
        K = np.loadtxt(calibration)
        if (K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0
                or not np.allclose(K[2], [0, 0, 1])):
            raise ValueError("Calibration must be a finite 3x3 pinhole matrix with positive focal lengths")
        intrinsic_info = {"source": "provided", "path": str(calibration.resolve())}
    else:
        K = estimate_intrinsics(width, height, args.focal_scale)
        intrinsic_info = {"source": "estimated", "focal_scale": args.focal_scale,
                          "formula": "fx=fy=focal_scale*max(width,height); cx=width/2; cy=height/2"}
    intrinsic_info.update(K=K.tolist(), width=width, height=height, distortion=[0.0] * 5,
                          scale="arbitrary; no metric scale reference")
    np.savetxt(output / "K.txt", K, fmt="%.8f")
    write_json(output / "intrinsics.json", intrinsic_info)
    write_json(output / "frames.json", records)
    write_json(output / "input.json", metadata)
    return images, K, intrinsic_info


def frame_usage(recon, final_ba, camera_ids, errors):
    """A frame is used only if its real pose has observations in final BA."""
    normalized = {cam: i for i, cam in enumerate(camera_ids)}
    usage = []
    for image_id in range(len(recon.images)):
        index = normalized.get(image_id)
        selected = final_ba.cam_idx == index if index is not None else np.zeros(len(errors), dtype=bool)
        count = int(np.count_nonzero(selected))
        usage.append({"image_id": image_id, "registered": index is not None,
                      "registration_method": recon.registration_methods.get(image_id),
                      "final_ba_observations": count,
                      "used_in_final_ba": index is not None and count >= recon.cfg.min_pnp_correspondences,
                      "mean_reprojection_error_px": float(errors[selected].mean()) if count else None,
                      "registration_attempt_failures": recon.registration_failures.get(image_id, 0)})
    return usage


def run_pipeline(args, output):
    from basic_backend import reconstruct
    from feature_cache import extract_and_match
    cv2.setRNGSeed(args.seed); cv2.setNumThreads(1); np.random.seed(args.seed)
    images, K, intrinsic_info = prepare_inputs(args, output)
    matcher = FeatureMatcher(len(images), MatchConfig(
        dataset_path=str(output / "frames"), img_pattern="{idx:04d}.png", ratio_thresh=.75,
        ransac_thresh=3., min_inliers=20, use_flann=False, nfeatures=args.nfeatures,
        num_threads=args.threads, mutual_check=True))
    matcher.images = images
    cache_info = extract_and_match(matcher, args.feature_cache)
    write_json(output / "feature_provenance.json", cache_info)
    return reconstruct(args, output, matcher, K, intrinsic_info)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path)
    source.add_argument("--images", type=Path)
    parser.add_argument("--calibration", type=Path, help="K.txt at the input resolution; images/K.txt is auto-detected")
    parser.add_argument("--output", type=Path, required=True, help="New or empty output directory")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=0,
                        help="0 (default): retain every sampled frame. Positive N explicitly caps video samples / image count.")
    parser.add_argument("--focal-scale", type=float, default=1.2)
    parser.add_argument("--nfeatures", type=int, default=3000)
    parser.add_argument("--engine", choices=["basic"], default="basic", help="Project-owned SfM using OpenCV/NumPy/SciPy primitives")
    parser.add_argument("--refine-focal", action="store_true", help="Estimate a common focal from image-pair F matrices, then refine in project-owned BA")
    parser.add_argument("--feature-cache", type=Path, help="Optional pixel/settings-bound cache of our SIFT/BF matches; never camera poses or 3D points")
    parser.add_argument("--ba-iterations", "--ba-max-nfev", dest="ba_max_nfev", type=int, default=100, help="Maximum residual evaluations (also iteration cap) of project-owned block LM per BA call")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if (not np.isfinite(args.fps) or args.fps <= 0 or not np.isfinite(args.focal_scale)
            or args.focal_scale <= 0 or args.max_frames < 0 or 0 < args.max_frames < 3 or args.ba_max_nfev < 1
            or args.threads < 1 or args.nfeatures < 0):
        parser.error("FPS/focal scale/BA budget/threads must be positive; max-frames is 0 or >= 3; nfeatures >= 0")
    return args


def main(argv=None):
    args = parse_args(argv)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output directory must be empty (existing results preserved): {output}")
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.FileHandler(output / "run.log"), logging.StreamHandler()], force=True)
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["versions"] = {p: importlib.metadata.version(p) for p in
                          ["numpy", "scipy", "opencv-python", "matplotlib"]}
    config["python"] = sys.version
    config["code_sha256"] = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                             for name in ["run_sfm.py", "matching.py", "basic_backend.py", "track_graph.py", "incremental_sfm.py", "sparse_ba.py", "feature_cache.py", "quality_checks.py", "ply_io.py", "scripts/render_ply.py"]}
    write_json(output / "config.json", config)
    start = time.perf_counter()
    try:
        result = run_pipeline(args, output)
    except Exception as exc:
        LOG.exception("Pipeline failed")
        result = {"status": "failed", "accepted": False, "error_type": type(exc).__name__, "error": str(exc)}
    result["elapsed_seconds"] = time.perf_counter() - start
    write_json(output / "metrics.json", result)
    LOG.info("Run result: %s", json.dumps(result, ensure_ascii=False))
    return 0 if result["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
