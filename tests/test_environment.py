import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import cv2
import numpy as np
import scipy

from bundle_adjustment import BAConfig, BundleAdjuster
from matching import FeatureMatcher, MatchConfig
from reconstruction import Reconstruction, ReconstructionCfg
from utilis import load_calibration_matrix


ROOT = Path(__file__).resolve().parents[1]


def test_core_imports_and_sift() -> None:
    assert tuple(map(int, cv2.__version__.split(".")[:2])) >= (4, 10)
    assert scipy.__version__
    assert cv2.SIFT_create().descriptorSize() == 128
    assert BAConfig and BundleAdjuster and Reconstruction and ReconstructionCfg


def test_bundled_calibration_and_dataset_layout() -> None:
    dataset = ROOT / "datasets" / "templeRing"
    K = load_calibration_matrix(dataset / "K.txt")
    assert K.shape == (3, 3)
    assert np.all(np.diag(K) > 0)
    assert (dataset / "00.png").is_file()
    assert (dataset / "01.png").is_file()


def test_two_view_feature_matching_smoke() -> None:
    dataset = ROOT / "datasets" / "templeRing"
    cfg = MatchConfig(
        dataset_path=str(dataset),
        img_pattern="{idx:02d}.png",
        ratio_thresh=0.8,
        ransac_thresh=3.0,
        min_inliers=8,
        use_flann=False,
    )
    matcher = FeatureMatcher(n_imgs=2, cfg=cfg)
    connected_pairs = matcher.run()
    assert len(matcher.kps) == 2
    assert all(len(keypoints) > 0 for keypoints in matcher.kps)
    assert matcher.adjacency.shape == (2, 2)
    assert matcher.adjacency[0, 1] == 1
    assert (0, 1) in connected_pairs
