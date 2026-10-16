import os

os.environ.setdefault("MPLBACKEND", "Agg")

import cv2
import numpy as np
import scipy

from matching import FeatureMatcher, MatchConfig
from incremental_sfm import IncrementalSfM
from sparse_ba import SparseBundleAdjuster


def test_core_imports_and_sift():
    assert tuple(map(int, cv2.__version__.split(".")[:2])) >= (4, 10)
    assert scipy.__version__
    assert cv2.SIFT_create().descriptorSize() == 128
    assert IncrementalSfM and SparseBundleAdjuster


def test_two_view_feature_matching_on_synthetic_texture(tmp_path):
    """Exercise disk loading, SIFT, matching and geometry without a bundled dataset."""
    rng = np.random.default_rng(23)
    image = rng.integers(0, 256, (240, 320), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), .7)
    transform = np.array([[1., .015, 9.], [-.01, 1., 6.], [.00003, -.00002, 1.]])
    warped = cv2.warpPerspective(image, transform, (320, 240))
    for index, pixels in enumerate((image, warped)):
        assert cv2.imwrite(str(tmp_path / f"{index:02d}.png"), pixels)
    cfg = MatchConfig(
        dataset_path=str(tmp_path), img_pattern="{idx:02d}.png", ratio_thresh=.8,
        ransac_thresh=2., min_inliers=20, use_flann=False, mutual_check=True,
        nfeatures=500, num_threads=1,
    )
    cv2.setRNGSeed(0)
    matcher = FeatureMatcher(n_imgs=2, cfg=cfg)
    assert matcher.run() == [(0, 1)]
    assert all(len(keypoints) >= 20 for keypoints in matcher.kps)
    assert len(matcher.matches[(0, 1)]) >= 20
    np.testing.assert_array_equal(matcher.adjacency, [[0, 1], [1, 0]])
