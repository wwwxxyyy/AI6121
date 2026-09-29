from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from bundle_adjustment import BAConfig, BundleAdjuster
from matching import FeatureMatcher, MatchConfig
from reconstruction import Point3DView, Reconstruction, ReconstructionCfg
from run_sfm import estimate_intrinsics, frame_usage, main, parse_args, prepare_inputs, sampled_frame_indices


def scene(second_rotation=None, n=4):
    K = np.array([[100., 0, 100], [0, 100, 100], [0, 0, 1]])
    points = np.array([[0., 0., 4.], [.2, .1, 5.], [-.3, .2, 6.], [.4, -.2, 4.5]])[:n]
    poses = {0: (np.eye(3), np.zeros((3, 1))),
             1: (np.eye(3) if second_rotation is None else second_rotation, np.array([[-.5], [0.], [0.]]))}
    keypoints = []
    for rotation, translation in poses.values():
        pixels = cv2.projectPoints(points, cv2.Rodrigues(rotation)[0], translation, K, None)[0].reshape(-1, 2)
        keypoints.append([cv2.KeyPoint(float(x), float(y), 1.) for x, y in pixels])
    matches = {(0, 1): [cv2.DMatch(i, i, 0.) for i in range(n)]}
    recon = Reconstruction(keypoints, matches, np.ones((2, 2), dtype=np.uint8),
                           ReconstructionCfg(K=K, min_pnp_correspondences=4),
                           [np.full((200, 200, 3), [200, 220, 240], dtype=np.uint8)] * 2)
    recon.poses = poses
    return recon, points


def test_matcher_handles_missing_and_single_neighbor_descriptors():
    fm = FeatureMatcher(2, MatchConfig("", "", use_flann=False))
    fm.des = [None, np.zeros((3, 128), dtype=np.float32)]
    assert fm._match_pair(0, 1)[1] == []
    fm.des = [np.zeros((2, 128), dtype=np.float32), np.zeros((1, 128), dtype=np.float32)]
    assert fm._match_pair(0, 1)[1] == []
    fm.des[1] = np.zeros((2, 128), dtype=np.float32)
    fm.matcher = SimpleNamespace(knnMatch=lambda *a, **kw: [[cv2.DMatch(0, 0, 0.)]])
    assert fm._match_pair(0, 1)[1] == []


def test_triangulation_recovers_geometry_and_does_not_overflow_rgb():
    recon, points = scene()
    recovered = recon._triangulate(0, 1, list(range(4)), list(range(4)))
    np.testing.assert_allclose([p.coords for p in recovered], points, atol=1e-5)
    assert all(p.rgb == (240, 220, 200) for p in recovered)


def test_triangulation_rejects_points_behind_second_camera():
    recon, _ = scene(np.diag([1., -1., -1.]))
    assert recon._triangulate(0, 1, list(range(4)), list(range(4))) == []


def test_baseline_triangulation_does_not_reuse_a_keypoint():
    recon, _ = scene()
    recovered = recon._triangulate(0, 1, [0, 0, 1], [0, 0, 1])
    assert len(recovered) == 2
    observations = [(cam, key) for p in recovered for cam, key in p.observations.items()]
    assert len(observations) == len(set(observations))


def test_triangulation_handles_single_point_and_nonfinite_homogeneous(monkeypatch):
    recon, _ = scene(n=1)
    assert len(recon._triangulate(0, 1, [0], [0])) == 1
    monkeypatch.setattr(cv2, "triangulatePoints", lambda *a: np.array([[0.], [0.], [1.], [0.]]))
    assert recon._triangulate(0, 1, [0], [0]) == []


def test_reversed_pair_uses_correct_keypoints():
    recon, _ = scene()
    a, b, ai, bi = recon._aligned_points(0, 1, True)
    rb, ra, rbi, rai = recon._aligned_points(1, 0, True)
    np.testing.assert_array_equal(a, ra)
    np.testing.assert_array_equal(b, rb)


def test_grow_tries_other_views_and_bounds_retries(monkeypatch):
    recon, _ = scene()
    recon.placed, recon.unplaced = [0, 1], [2, 3]
    monkeypatch.setattr(recon, "_count_correspondences", lambda _: 10)
    calls = []
    def register(i):
        calls.append(i)
        if i == 2:
            raise RuntimeError("synthetic PnP failure")
        recon.poses[i] = recon.poses[0]
        recon.placed.append(i)
        recon.unplaced.remove(i)
    monkeypatch.setattr(recon, "_add_image", register)
    recon.grow()
    assert 3 in recon.poses and recon.unplaced == [2]
    # One failure before view 3 grows the map, then bounded retries on the new map.
    assert calls.count(2) == 1 + recon.cfg.max_failed_attempts
    assert recon.registration_failures[2] == calls.count(2)


def test_exhausted_frame_is_reconsidered_after_map_growth(monkeypatch):
    recon, _ = scene()
    recon.placed, recon.unplaced = [0, 1], [2, 3]
    recon.failed_attempts[2] = recon.cfg.max_failed_attempts
    monkeypatch.setattr(recon, "_count_correspondences", lambda _: 10)
    def register(i):
        if i == 2 and 3 not in recon.poses:
            raise RuntimeError("requires new map support")
        recon.poses[i] = recon.poses[0]
        recon.placed.append(i)
        recon.unplaced.remove(i)
    monkeypatch.setattr(recon, "_add_image", register)
    recon.grow()
    assert set(recon.poses) == {0, 1, 2, 3}
    assert not recon.unplaced


def test_registration_rolls_back_after_triangulation_failure(monkeypatch):
    recon, points = scene()
    recon.points3d = [Point3DView(p, (1, 2, 3), {0: i, 1: i}) for i, p in enumerate(points)]
    recon.keypoints.append(recon.keypoints[1])
    recon.placed, recon.unplaced = [0, 1], [2]
    monkeypatch.setattr(recon, "_correspondences", lambda _: list(zip(range(4), range(4))))
    monkeypatch.setattr(cv2, "solvePnPRansac", lambda *a, **kw: (True, np.zeros(3), np.zeros(3), np.arange(4).reshape(-1, 1)))
    def fail(_):
        recon.points3d.append(Point3DView(np.ones(3)))
        raise RuntimeError("synthetic triangulation failure")
    monkeypatch.setattr(recon, "_triangulate_new_matches", fail)
    with pytest.raises(RuntimeError):
        recon._add_image(2)
    assert len(recon.points3d) == 4 and 2 not in recon.poses
    assert recon.placed == [0, 1] and recon.unplaced == [2]
    assert all(2 not in p.observations for p in recon.points3d)


def test_ba_failure_is_not_misclassified_as_registration_failure(monkeypatch):
    recon, _ = scene()
    recon.placed, recon.unplaced = [0, 1], [2]
    recon.cfg.bundle_every = 1
    monkeypatch.setattr(recon, "_count_correspondences", lambda _: 10)
    def register(i):
        recon.poses[i] = recon.poses[0]
        recon.placed.append(i)
        recon.unplaced.remove(i)
    def fail():
        raise RuntimeError("synthetic BA failure")
    monkeypatch.setattr(recon, "_add_image", register)
    with pytest.raises(RuntimeError, match="BA failure"):
        recon.grow(bundle_adjust_fn=fail)
    assert 2 in recon.poses and not recon.failed_attempts


def test_correspondences_are_unique_across_registered_views():
    recon, points = scene()
    recon.placed = [0, 1]
    recon.points3d = [Point3DView(p, (1, 2, 3), {0: i, 1: i}) for i, p in enumerate(points)]
    recon.keypoints.append(recon.keypoints[1])
    recon.matches[(0, 2)] = recon.matches[(0, 1)]
    recon.matches[(1, 2)] = recon.matches[(0, 1)]
    assert len(recon._correspondences(2)) == len(points)


def test_batched_ba_projection_matches_independent_scalar_projection():
    recon, points = scene()
    cam = np.array([0, 1, 0, 1, 1, 0])
    ids = np.array([0, 1, 2, 3, 0, 1])
    r = {i: cv2.Rodrigues(R)[0].ravel() for i, (R, t) in recon.poses.items()}
    t = {i: t for i, (R, t) in recon.poses.items()}
    expected = np.array([cv2.projectPoints(points[p:p+1], r[c], t[c], recon.cfg.K, None)[0].ravel()
                         for c, p in zip(cam, ids)])
    ba = BundleAdjuster(recon.cfg.K, cam, ids, expected, 2, 4, r, t, points, BAConfig(verbose=0))
    np.testing.assert_allclose(ba._project(ba.x0), expected)
    assert ba.compute_average_reprojection_error() < 1e-10


@pytest.mark.parametrize("width,height,f", [(544, 960, 1152), (1280, 720, 1536)])
def test_estimated_intrinsics(width, height, f):
    K = estimate_intrinsics(width, height)
    np.testing.assert_allclose(K, [[f, 0, width/2], [0, f, height/2], [0, 0, 1]])


def test_frame_cap_spans_video_and_preserves_order():
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


def test_loading_a_frame_or_assigning_a_pose_does_not_count_as_ba_use():
    recon, _ = scene()
    recon.images.append(recon.images[0])
    # Camera 1 has a pose but only one observation; camera 2 is not registered.
    ba = SimpleNamespace(cam_idx=np.array([0, 0, 0, 0, 1]))
    usage = frame_usage(recon, ba, [0, 1], np.ones(5))
    assert usage[0]["used_in_final_ba"]
    assert not usage[1]["used_in_final_ba"] and usage[1]["registered"]
    assert not usage[2]["used_in_final_ba"] and not usage[2]["registered"]


def test_flow_tracks_real_translated_pixels_without_assigning_pose():
    rng = np.random.default_rng(8)
    source = rng.integers(0, 256, (160, 160), dtype=np.uint8)
    target = cv2.warpAffine(source, np.array([[1., 0., 2.], [0., 1., 1.]]), (160, 160))
    pixels = [(50., 50.), (100., 50.), (50., 100.), (100., 100.)]
    keypoints = [[cv2.KeyPoint(x, y, 1.) for x, y in pixels], []]
    recon = Reconstruction(keypoints, {}, np.zeros((2, 2)), ReconstructionCfg(np.eye(3)), [source, target])
    recon.placed = [0]
    recon.points3d = [Point3DView(np.array([0., 0., 5.]), observations={0: i}) for i in range(4)]
    result = dict(recon._flow_correspondences(1))
    assert len(result) == 4
    for point_id, xy in result.items():
        np.testing.assert_allclose(xy, np.array(pixels[point_id]) + [2, 1], atol=.1)
    assert 1 not in recon.poses and not recon.keypoints[1]


def test_flow_recovery_requires_real_pnp_and_adds_ba_observations(monkeypatch):
    original, points = scene()
    recon = Reconstruction(tuple(tuple(view) for view in original.keypoints), original.matches,
                           original.adjacency, original.cfg, original.images)
    recon.poses = original.poses
    recon.cfg.optical_flow_fallback = True
    recon.placed, recon.unplaced = [0, 1], [2]
    recon.images.append(recon.images[0])
    recon.keypoints.append([])
    recon.points3d = [Point3DView(p, (1, 2, 3), {0: i, 1: i}) for i, p in enumerate(points)]
    monkeypatch.setattr(recon, "_flow_correspondences", lambda _: [(i, recon.keypoints[1][i].pt) for i in range(4)])
    monkeypatch.setattr(cv2, "solvePnPRansac", lambda *a, **kw: (True, np.zeros(3), np.array([-.5, 0., 0.]), np.arange(4).reshape(-1, 1)))
    recon._add_image(2)
    assert recon.registration_methods[2] == "pnp_optical_flow"
    assert len(recon.keypoints[2]) == 4 and not recon.unplaced
    assert all(2 in p.observations for p in recon.points3d)


def test_failed_flow_pnp_leaves_no_pose_or_synthetic_observations(monkeypatch):
    recon, points = scene()
    recon.cfg.optical_flow_fallback = True
    recon.placed, recon.unplaced = [0, 1], [2]
    recon.images.append(recon.images[0]); recon.keypoints.append([])
    recon.points3d = [Point3DView(p, (1, 2, 3), {0: i, 1: i}) for i, p in enumerate(points)]
    monkeypatch.setattr(recon, "_flow_correspondences", lambda _: [(i, recon.keypoints[1][i].pt) for i in range(4)])
    monkeypatch.setattr(cv2, "solvePnPRansac", lambda *a, **kw: (False, None, None, None))
    with pytest.raises(RuntimeError, match="PnP failed"):
        recon._add_image(2)
    assert 2 not in recon.poses and recon.unplaced == [2]
    assert not recon.keypoints[2] and all(2 not in p.observations for p in recon.points3d)


def test_image_preparation_copies_first_n_without_renaming_source(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir(); output.mkdir()
    for i in [1, 2, 10, 11]:
        cv2.imwrite(str(source / f"photo{i}.png"), np.full((20, 30, 3), i, dtype=np.uint8))
    original = sorted(p.name for p in source.iterdir())
    args = SimpleNamespace(video=None, images=source, max_frames=3, calibration=None, focal_scale=1.2)
    images, K, metadata = prepare_inputs(args, output)
    assert [int(img[0, 0, 0]) for img in images] == [1, 2, 10]
    assert sorted(p.name for p in source.iterdir()) == original
    assert metadata["source"] == "estimated"


def test_cli_preserves_existing_output(tmp_path):
    marker = tmp_path / "keep.txt"
    marker.write_text("original")
    with pytest.raises(SystemExit, match="existing results preserved"):
        main(["--video", "does-not-exist.mp4", "--output", str(tmp_path)])
    assert marker.read_text() == "original"
