"""Explicit feature tracks, F-RANSAC verification and scalar focal initialization."""
import logging
import cv2
import numpy as np
from scipy.optimize import minimize_scalar

LOG = logging.getLogger(__name__)


def verified_pairs(keypoints, matches, seed=0):
    points = [np.asarray([k.pt for k in view], float).reshape(-1, 2) for view in keypoints]
    result = {}; matrices = {}; rows = []
    for (i, j), ms in sorted(matches.items()):
        if len(ms) < 20: continue
        a = np.array([m.queryIdx for m in ms]); b = np.array([m.trainIdx for m in ms])
        p, q = points[i][a], points[j][b]
        cv2.setRNGSeed(seed)
        F, mask = cv2.findFundamentalMat(p, q, cv2.USAC_MAGSAC, 3., .999, 5000)
        if F is None or F.shape != (3,3) or mask is None: continue
        keep = mask.ravel().astype(bool)
        if keep.sum() < 20: continue
        result[i, j] = (a[keep], b[keep], np.array([m.distance for m in ms])[keep])
        matrices[i, j] = F
        rows.append({'images':[i,j],'ratio_matches':len(ms),'F_inliers':int(keep.sum())})
    LOG.info('Verified %d F pairs', len(result))
    return points, result, matrices, rows


def build_tracks(keypoints, pairs):
    """Greedy union by descriptor distance, rejecting duplicate-image conflicts.

    Every accepted union explicitly merges two observation dictionaries. A
    component can contain at most one keypoint from each input frame.
    """
    offsets = np.r_[0, np.cumsum([len(k) for k in keypoints])]
    parent = np.arange(offsets[-1]); views = {}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return int(x)
    edges = []
    for (i, j), (a, b, distance) in pairs.items():
        edges.extend(zip(distance.tolist(), (a+offsets[i]).tolist(), (b+offsets[j]).tolist(), [i]*len(a), [j]*len(a), a.tolist(), b.tolist()))
    conflicts = 0; merged = 0
    for _, a, b, i, j, ki, kj in sorted(edges):
        ra, rb = find(a), find(b)
        if ra == rb: continue
        va = views.get(ra, {i:ki}); vb = views.get(rb, {j:kj})
        if va.keys() & vb.keys():
            conflicts += 1; continue
        if len(va) < len(vb): ra, rb, va, vb = rb, ra, vb, va
        parent[rb] = ra; va.update(vb); views[ra] = va; views.pop(rb, None); merged += 1
    tracks = [dict(sorted(v.items())) for _,v in sorted(views.items()) if len(v)>=2]
    feature_to_track = [{ } for _ in keypoints]
    for t, obs in enumerate(tracks):
        for i, k in obs.items(): feature_to_track[i][k] = t
    stats = {'tracks':len(tracks),'edges':len(edges),'accepted_unions':merged,'image_conflicts_rejected':conflicts,
             'track_lengths':{str(n):sum(len(t)==n for t in tracks) for n in sorted(set(map(len,tracks)))}}
    LOG.info('Built %d tracks; rejected %d image conflicts', len(tracks), conflicts)
    return tracks, feature_to_track, stats


def estimate_focal(K, matrices, pairs, image_shape):
    """Fit equal nonzero singular values of K^T F K, using many image pairs.

    This is an initialization under centered principal point/equal focal/zero
    distortion assumptions. It is recorded as an estimate, then refined in BA.
    No prior reconstruction or calibration-library result is read.
    """
    size = max(image_shape[:2]); lo, hi = .3*size, 2.*size
    selected = sorted((p for p in matrices if len(pairs[p][0])>=60 and 1<=p[1]-p[0]<=6),
                      key=lambda p:len(pairs[p][0]), reverse=True)[:100]
    if len(selected)<3: return K.copy(), {'method':'insufficient pair support; retain input K','selected_pairs':len(selected)}
    Fs = np.array([matrices[p] for p in selected])
    def errors(logf):
        k = K.copy(); k[0,0] = k[1,1] = np.exp(logf)
        E = k.T @ Fs @ k
        s = np.linalg.svd(E, compute_uv=False)
        return (s[:,0]-s[:,1])/(s[:,0]+s[:,1]+1e-15)
    grid = np.linspace(np.log(lo), np.log(hi), 81)
    scores = np.array([np.median(errors(v)**2) for v in grid]); idx = int(np.argmin(scores))
    opt = minimize_scalar(lambda f:np.median(errors(f)**2), bounds=(grid[max(0,idx-2)],grid[min(len(grid)-1,idx+2)]),method='bounded')
    f = float(np.exp(opt.x)); result = K.copy(); result[0,0] = result[1,1] = f
    stats = {'method':'median equal-singular-value residual of K.T@F@K; OpenCV F + NumPy SVD + scalar search',
             'input_focal':float(K[0,0]),'estimated_focal':f,'bounds_px':[lo,hi],'pairs':[list(p) for p in selected],
             'objective':float(opt.fun),'at_search_boundary':bool(f<lo*1.01 or f>hi*.99),
             'grid_focal_px':np.exp(grid).tolist(),'grid_objective':scores.tolist()}
    LOG.info('Basic F/SVD focal initialization %.2f -> %.2f px',K[0,0],f)
    return result, stats
