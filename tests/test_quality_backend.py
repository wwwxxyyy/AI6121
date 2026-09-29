from types import SimpleNamespace
import importlib.abc
import subprocess
import sys
import cv2
import numpy as np

from matching import FeatureMatcher, MatchConfig
from quality_checks import sampson_errors, summarize_geometry
from sparse_ba import SparseBundleAdjuster
from incremental_sfm import IncrementalSfM, triangulate_dlt, point_errors
from track_graph import build_tracks
from ply_io import read_ply, write_ply, fuse_points


def test_mutual_ratio_rejects_a_one_way_match():
    matcher = FeatureMatcher(2, MatchConfig('', '', use_flann=False, mutual_check=True))
    matcher.des = [np.zeros((2, 128), np.float32)] * 2
    calls = iter([
        [[cv2.DMatch(0, 0, 1.), cv2.DMatch(0, 1, 10.)],
         [cv2.DMatch(1, 1, 1.), cv2.DMatch(1, 0, 10.)]],
        [[cv2.DMatch(0, 0, 1.), cv2.DMatch(0, 1, 10.)],
         [cv2.DMatch(1, 0, 1.), cv2.DMatch(1, 1, 10.)]],
    ])
    matcher.matcher = SimpleNamespace(knnMatch=lambda *a, **kw: next(calls))
    _, matches = matcher._match_pair(0, 1)
    assert [(m.queryIdx, m.trainIdx) for m in matches] == [(0, 0)]


def synthetic_scene():
    rng = np.random.default_rng(4)
    X = rng.uniform([-1., -1., 4.], [1., 1., 7.], size=(80, 3))
    K = np.array([[800., 0., 320.], [0., 800., 240.], [0., 0., 1.]])
    poses = [(cv2.Rodrigues(np.array([0., i*.04, i*.003]))[0], np.array([-i*.3, i*.03, i*.02])) for i in range(5)]
    uv = np.array([cv2.projectPoints(X,cv2.Rodrigues(R)[0],t,K,None)[0].reshape(-1,2) for R,t in poses])
    return rng, X, K, poses, uv


def test_geometry_check_catches_inconsistent_neighbor_pose():
    _, X, K, p, uv = synthetic_scene()
    poses = {0:p[0],1:p[1]}
    obs = [{0:a,1:b} for a,b in zip(uv[0],uv[1])]
    good = summarize_geometry(X,poses,obs,K,[(0,1,uv[0],uv[1])])
    assert good['summary']['all_adjacent_pairs_passed']
    bad_pose = (cv2.Rodrigues(np.array([.3,0.,0.]))[0],poses[1][1])
    bad = summarize_geometry(X,{0:poses[0],1:bad_pose},obs,K,[(0,1,uv[0],uv[1])])
    assert not bad['summary']['all_adjacent_pairs_passed']
    assert np.median(sampson_errors(K,poses[0],bad_pose,uv[0],uv[1])) > 20


def test_self_written_ba_recovers_noisy_scene_and_preserves_gauge():
    rng,X,K,poses,uv = synthetic_scene()
    noisy = [(R,t.copy()) if i==0 else (cv2.Rodrigues(cv2.Rodrigues(R)[0].ravel()+rng.normal(0,.004,3))[0],t+rng.normal(0,.008,3)) for i,(R,t) in enumerate(poses)]
    ba = SparseBundleAdjuster(K,noisy,X+rng.normal(0,.03,X.shape),np.repeat(np.arange(5),80),np.tile(np.arange(80),5),uv.reshape(-1,2))
    fixed = ba.cameras.ravel()[ba.fixed].copy()
    result,xyz,k,stats = ba.optimize(100)
    assert stats['cost'] < stats['initial_cost'] * .001
    assert stats['mean_error_after_px'] < .01
    assert stats['converged'] and stats['successful_steps'] > 2
    assert stats['nfev'] <= 100
    np.testing.assert_array_equal(ba.cameras.ravel()[ba.fixed],fixed)


def test_ba_projection_jacobian_matches_central_difference():
    rng,X,K,poses,uv = synthetic_scene()
    ba = SparseBundleAdjuster(K,poses,X,np.repeat(np.arange(5),80),np.tile(np.arange(80),5),uv.reshape(-1,2),True)
    r,Jc,Jp,_ = ba.evaluate(jacobian=True)
    d = rng.normal(0,.01,len(ba.free)); dp = rng.normal(0,.01,X.shape)
    full = np.zeros(31); full[ba.free] = d; dc = full[:30].reshape(5,6)
    eps=1e-5
    numeric=(ba.evaluate(ba.cameras+eps*dc,X+eps*dp,eps*full[-1])-ba.evaluate(ba.cameras-eps*dc,X-eps*dp,-eps*full[-1]))/(2*eps)
    analytic=Jc@d+Jp@dp.ravel()
    np.testing.assert_allclose(analytic,numeric.ravel(),rtol=1e-5,atol=1e-6)


def test_track_union_merges_transitively_without_same_image_conflict():
    keypoints=[[None]*2 for _ in range(3)]
    pair=lambda a,b,d:(np.array(a),np.array(b),np.array(d))
    pairs={(0,1):pair([0],[0],[1.]),(1,2):pair([0],[0],[2.]),(0,2):pair([1],[0],[3.])}
    tracks,lookup,stats=build_tracks(keypoints,pairs)
    assert tracks==[{0:0,1:0,2:0}]
    assert lookup[0]=={0:0}
    assert stats['image_conflicts_rejected']==1


def test_multiview_dlt_and_cheirality():
    _,X,K,poses,uv=synthetic_scene()
    reconstructed=triangulate_dlt(poses,uv[:,0],K)
    np.testing.assert_allclose(reconstructed,X[0],atol=1e-10)
    errors,z=point_errors(reconstructed,poses,uv[:,0],K)
    assert errors.max()<1e-8 and (z>0).all()
    inverted=(np.diag([1.,-1.,-1.]),np.zeros(3))
    assert point_errors(X[0],[inverted],uv[:1,0],K)[1][0]<0


def test_basic_registration_failure_does_not_assign_pose(monkeypatch):
    _,X,K,poses,uv=synthetic_scene()
    tracks=[{0:i,1:i,2:i} for i in range(80)]
    recon=IncrementalSfM(list(uv[:3]),{},tracks,[{i:i for i in range(80)}]*3,K,[None]*3)
    recon.poses={0:poses[0],1:poses[1]};recon.xyz=dict(enumerate(X));recon.observations={i:{0:i,1:i} for i in range(80)}
    monkeypatch.setattr(cv2,'solvePnPRansac',lambda *a,**kw:(False,None,None,None))
    assert not recon.register(2)
    assert 2 not in recon.poses and all(2 not in obs for obs in recon.observations.values())


def test_binary_ply_round_trip_and_small_voxel_cloud(tmp_path):
    xyz=np.array([[1.,2.,3.],[1.01,2.,3.]])
    rgb=np.array([[200.,220.,240.],[240.,220.,200.]])/255.
    path=tmp_path/'points.ply';write_ply(path,xyz,rgb)
    a,b=read_ply(path);np.testing.assert_array_equal(a,xyz);np.testing.assert_allclose(b,rgb)
    p,c=fuse_points(a,b,.1)
    assert len(p)==1
    np.testing.assert_allclose(c[0],[220,220,220]/np.array(255.))


def test_current_cli_imports_without_sfm_or_point_cloud_wrappers():
    code="""import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, *a):
  if fullname.split('.')[0] in {'pycolmap','open3d','gtsam','pyceres'}: raise RuntimeError('Forbidden import: '+fullname)
sys.meta_path.insert(0,Block())
import run_sfm, basic_backend, scripts.stereo_cloud
"""
    subprocess.run([sys.executable,'-c',code],check=True)


def test_basic_registration_rolls_back_when_triangulation_raises(monkeypatch):
    _,X,K,poses,uv=synthetic_scene()
    tracks=[{0:i,1:i,2:i} for i in range(80)]
    recon=IncrementalSfM(list(uv[:3]),{},tracks,[{i:i for i in range(80)}]*3,K,[None]*3)
    recon.poses={0:poses[0],1:poses[1]};recon.xyz=dict(enumerate(X));recon.observations={i:{0:i,1:i} for i in range(80)}
    def fail(_):
        recon.xyz.pop(0); raise RuntimeError('synthetic triangulation failure')
    monkeypatch.setattr(recon,'retriangulate',fail)
    import pytest
    with pytest.raises(RuntimeError,match='synthetic triangulation'):
        recon.register(2)
    assert 2 not in recon.poses and len(recon.xyz)==80
    assert all(2 not in obs for obs in recon.observations.values())


def test_basic_export_averages_color_in_float():
    im=np.full((10,10,3),[200,220,240],np.uint8)
    recon=IncrementalSfM([np.array([[5.,5.]])]*3,{},[{0:0,1:0,2:0}],[{0:0}]*3,np.eye(3),[im]*3)
    recon.xyz={0:np.array([0.,0.,5.])};recon.observations={0:{0:0,1:0,2:0}}
    xyz,rgb,obs=recon.export()
    np.testing.assert_allclose(rgb[0],[240/255,220/255,200/255])


def test_final_ba_repeats_after_filter_changes_observations(monkeypatch):
    recon=IncrementalSfM([],{},[],[],np.eye(3),[])
    recon.poses={i:None for i in range(4)}
    recon.xyz={0:np.array([0.,0.,5.])};recon.observations={0:{0:0,1:0,2:0,3:0}}
    optimized=[]
    monkeypatch.setattr(recon,'bundle',lambda label:optimized.append(recon.observations[0].copy()))
    def remove_boundary_observation(min_views):
        recon.observations[0].pop(3,None)
    monkeypatch.setattr(recon,'filter_points',remove_boundary_observation)
    recon.finalize()
    assert len(optimized)==2
    assert set(optimized[0])=={0,1,2,3}
    assert optimized[-1]==recon.observations[0]
