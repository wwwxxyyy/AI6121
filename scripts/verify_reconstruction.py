#!/usr/bin/env python3
"""Independent PLY/projection and fresh adjacent-image verification.

Does not reuse the mapper's match database or quality-check implementation.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ply_io import read_ply, write_ply, fuse_points


def verify(folder):
    load=lambda f:json.loads((folder/f).read_text())
    frames=load('frames.json');meta=load('input.json');cfg=load('config.json');metrics=load('metrics.json')
    root=Path(__file__).resolve().parents[1]
    assert cfg['fps']==2 and cfg['max_frames']==0
    assert len(frames)==int(np.ceil(meta['duration_seconds']*2))
    assert hashlib.sha256(Path(meta['source']).read_bytes()).hexdigest()==meta['source_sha256']
    assert all(hashlib.sha256((root/f).read_bytes()).hexdigest()==sha for f,sha in cfg['code_sha256'].items())
    assert all(abs(f['timestamp_seconds']-i*.5)<=1/meta['source_fps'] for i,f in enumerate(frames))
    xyz,rgb=read_ply(folder/'reconstruction.ply')
    assert xyz.shape==rgb.shape and len(xyz)==metrics['points'] and np.isfinite(xyz).all()
    K=np.loadtxt(folder/'K.txt');inv=np.linalg.inv(K)
    cams={c['image_id']:c for c in load('cameras.json')}
    assert set(cams)==set(range(len(frames)))
    R={c:np.array(v['R_world_to_camera']) for c,v in cams.items()}
    t={c:np.array(v['t_world_to_camera']) for c,v in cams.items()}
    center={c:-R[c].T@t[c] for c in cams}
    for c in cams:np.testing.assert_allclose(R[c].T@R[c],np.eye(3),atol=1e-7)
    shared={c:set() for c in cams};counts=Counter();errs=[];seen=set();angles=[];lengths=[]
    for p in load('observations.json'):
        X=xyz[p['point_id']];rays=[];lengths.append(len(p['views']))
        assert len(p['views'])>=2
        for cs,key in p['views'].items():
            c=int(cs);assert (c,key) not in seen;seen.add((c,key))
            Y=R[c]@X+t[c];assert Y[2]>0
            uv=K@Y;err=np.linalg.norm(uv[:2]/uv[2]-p['pixels'][cs]);errs.append(float(err))
            counts[c]+=1;shared[c].add(p['point_id'])
            ray=X-center[c];rays.append(ray/np.linalg.norm(ray))
        rays=np.array(rays);theta=np.degrees(np.arccos(np.clip(rays@rays.T,-1,1)))
        angles.append(float(np.minimum(theta,180-theta).max()))
    assert min(counts.values())>=10
    assert min(angles)>=1.5-1e-3
    assert abs(np.mean(errs)-metrics['final_reprojection_error_px']['mean'])<.005
    usage=load('frame_usage.json')
    assert all(u['used_in_final_ba'] and u['final_ba_observations']==counts[u['image_id']] for u in usage)
    cv2.setNumThreads(1);sift=cv2.SIFT_create(nfeatures=3000);matcher=cv2.BFMatcher(cv2.NORM_L2)
    feature=[sift.detectAndCompute(cv2.imread(str(folder/f['file'])),None) for f in frames]
    pairs=[]
    for i in range(len(frames)-1):
        j=i+1;ki,di=feature[i];kj,dj=feature[j]
        ms=[m for m,n in matcher.knnMatch(di,dj,k=2) if m.distance<.75*n.distance]
        p=np.float32([ki[m.queryIdx].pt for m in ms]);q=np.float32([kj[m.trainIdx].pt for m in ms])
        cv2.setRNGSeed(0);_,mask=cv2.findFundamentalMat(p,q,cv2.FM_RANSAC,3.)
        good=mask.ravel().astype(bool);p=p[good];q=q[good]
        relR=R[j]@R[i].T;relt=t[j]-relR@t[i];a,b,c=relt
        skew=np.array([[0,-c,b],[c,0,-a],[-b,a,0]])
        F=inv.T@skew@relR@inv
        ph=np.c_[p,np.ones(len(p))];qh=np.c_[q,np.ones(len(q))]
        fp=ph@F.T;fq=qh@F
        dist=np.abs(np.sum(qh*fp,axis=1))/np.sqrt(np.sum(fp[:,:2]**2+fq[:,:2]**2,axis=1))
        median=float(np.median(dist));nshared=len(shared[i]&shared[j])
        pairs.append({'images':[i,j],'fresh_F_inliers':len(p),'shared_points':nshared,
                      'sampson_median_px':median,'passed':len(p)>=20 and median<=3 and nshared>=15})
    result={'all_frames_verified':True,'sampled_frames':len(frames),'registered_frames':len(cams),
            'points':len(xyz),'observations':len(errs),'min_final_ba_observations':min(counts.values()),
            'mean_reprojection_error_px':float(np.mean(errs)),'mean_track_length':float(np.mean(lengths)),
            'two_view_fraction':float(np.mean(np.array(lengths)==2)),
            'min_triangulation_angle_deg':min(angles),'nonpositive_depth_observations':0,
            'independent_adjacent_pairs':pairs,'all_adjacent_pairs_passed':all(p['passed'] for p in pairs),
            'source_unchanged':True,'current_code_matches_run':True,'preview_readable':cv2.imread(str(folder/'preview.png')) is not None}
    (folder/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='independent_adjacent_pairs'},indent=2))
    assert result['all_adjacent_pairs_passed'] and metrics['accepted']
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);args=p.parse_args();verify(args.folder)
