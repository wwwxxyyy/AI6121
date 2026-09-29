import cv2
import numpy as np
import pytest

from scripts.stereo_cloud import stereo_depth


@pytest.mark.parametrize('vertical', [False, True])
@pytest.mark.parametrize('rectify_alpha', [0., -1.])
def test_stereo_known_plane_recovers_metric_depth(vertical, rectify_alpha):
    # A textured plane at Z=4: disparity=f*baseline/Z=12 pixels.
    rng = np.random.default_rng(8)
    left = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    left = cv2.GaussianBlur(left, (3, 3), .7)
    right = np.zeros_like(left)
    if vertical:
        right[:-12] = left[12:]
        t = np.array([0., -.2, 0.])
    else:
        right[:, :-12] = left[:, 12:]
        t = np.array([-.2, 0., 0.])
    K = np.array([[240., 0., 160.], [0., 240., 120.], [0., 0., 1.]])
    depth = stereo_depth(left, right, K, np.eye(3), t, (1., 10.), rectify_alpha, 80)
    z = depth[depth > 0]
    assert len(z) > 10000
    assert np.median(np.abs(z - 4.)) < .03
    assert np.percentile(np.abs(z - 4.), 95) < .08
