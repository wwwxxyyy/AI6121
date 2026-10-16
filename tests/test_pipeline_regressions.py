from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from matching import FeatureMatcher, MatchConfig
from run_sfm import estimate_intrinsics, main, parse_args, prepare_inputs, sampled_frame_indices


def test_matcher_handles_missing_and_single_neighbor_descriptors():
    matcher = FeatureMatcher(2, MatchConfig("", "", use_flann=False))
    matcher.des = [None, np.zeros((3, 128), dtype=np.float32)]
    assert matcher._match_pair(0, 1)[1] == []
    matcher.des = [np.zeros((2, 128), dtype=np.float32), np.zeros((1, 128), dtype=np.float32)]
    assert matcher._match_pair(0, 1)[1] == []
    matcher.des[1] = np.zeros((2, 128), dtype=np.float32)
    matcher.matcher = SimpleNamespace(knnMatch=lambda *a, **kw: [[cv2.DMatch(0, 0, 0.)]])
    assert matcher._match_pair(0, 1)[1] == []


def test_adjacency_is_symmetric_and_discards_stale_or_weak_edges():
    matcher = FeatureMatcher(3, MatchConfig("", "", min_inliers=3, use_flann=False))
    matcher.adjacency.fill(1)
    matcher.matches = {
        (0, 1): [cv2.DMatch(i, i, 0.) for i in range(3)],
        (1, 2): [cv2.DMatch(i, i, 0.) for i in range(2)],
    }
    assert matcher.build_adjacency() == [(0, 1)]
    np.testing.assert_array_equal(matcher.adjacency, [[0, 1, 0], [1, 0, 0], [0, 0, 0]])


@pytest.mark.parametrize("width,height,f", [(544, 960, 1152), (1280, 720, 1536)])
def test_estimated_intrinsics(width, height, f):
    K = estimate_intrinsics(width, height)
    np.testing.assert_allclose(K, [[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]])


def test_explicit_frame_cap_spans_video_and_preserves_order():
    indices = sampled_frame_indices(903, 903 / 30.1333333333, 2, 40)
    assert len(indices) == 40 and indices[0] == 0 and indices[-1] > 890
    assert np.all(np.diff(indices) > 0)


def test_two_fps_default_retains_every_sample():
    hydrant = sampled_frame_indices(491, 30., 2.)
    garden = sampled_frame_indices(903, 903 / 30.1333333333, 2.)
    assert len(hydrant) == 33 and len(garden) == 61
    np.testing.assert_array_equal(hydrant, np.arange(0, 491, 15))
    assert np.all(np.isin(np.diff(garden), [14, 15]))
    args = parse_args(["--video", "input.mp4", "--output", "new-output"])
    assert args.fps == 2 and args.max_frames == 0


@pytest.mark.parametrize("option,value", [("--fps", "0"), ("--fps", "nan"),
                                         ("--max-frames", "2"), ("--focal-scale", "-1"),
                                         ("--ba-max-nfev", "0")])
def test_cli_rejects_invalid_sampling_and_camera_parameters(option, value):
    with pytest.raises(SystemExit):
        parse_args(["--video", "input.mp4", "--output", "new-output", option, value])


def test_image_preparation_copies_first_n_without_renaming_source(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    for i in [1, 2, 10, 11]:
        cv2.imwrite(str(source / f"photo{i}.png"), np.full((20, 30, 3), i, dtype=np.uint8))
    original = sorted(p.name for p in source.iterdir())
    args = SimpleNamespace(video=None, images=source, max_frames=3, calibration=None, focal_scale=1.2)
    images, K, metadata = prepare_inputs(args, output)
    assert [int(img[0, 0, 0]) for img in images] == [1, 2, 10]
    assert sorted(p.name for p in source.iterdir()) == original
    assert metadata["source"] == "estimated"
    np.testing.assert_allclose(K, estimate_intrinsics(30, 20))


def test_cli_preserves_existing_output(tmp_path):
    marker = tmp_path / "keep.txt"
    marker.write_text("original")
    with pytest.raises(SystemExit, match="existing results preserved"):
        main(["--video", "does-not-exist.mp4", "--output", str(tmp_path)])
    assert marker.read_text() == "original"
