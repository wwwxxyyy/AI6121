"""Incremental SfM implemented explicitly from OpenCV geometric primitives."""
import logging
import time
from itertools import combinations
import cv2
import numpy as np
from sparse_ba import SparseBundleAdjuster

LOG = logging.getLogger(__name__)


def triangulate_dlt(poses, pixels, K):
    xy = cv2.undistortPoints(np.asarray(pixels,float).reshape(-1,1,2),K,None).reshape(-1,2)
    A=[]
    for (R,t),(u,v) in zip(poses,xy):
        P=np.column_stack([R,np.asarray(t).ravel()]);A.extend([u*P[2]-P[0],v*P[2]-P[1]])
    A=np.asarray(A);A/=np.maximum(np.linalg.norm(A[:,:3],axis=1,keepdims=True),1e-15)
    _,_,V=np.linalg.svd(A,full_matrices=False);h=V[-1]
    return h[:3]/h[3] if abs(h[3])>1e-12 else np.full(3,np.nan)


def point_errors(X, poses, pixels, K):
    Y=np.array([R@X+np.asarray(t).ravel() for R,t in poses]);uv=Y@K.T
    with np.errstate(divide='ignore',invalid='ignore'):
        error=np.linalg.norm(uv[:,:2]/uv[:,2:]-pixels,axis=1)
    return error,Y[:,2]


class IncrementalSfM:
    def __init__(self, pixels, pairs, tracks, feature_tracks, K, images, refine_focal=False, max_ba_iterations=100):
        self.pixels=pixels;self.pairs=pairs;self.tracks=tracks;self.feature_tracks=feature_tracks
        self.K=K.copy();self.images=images;self.refine_focal=refine_focal;self.max_ba_iterations=max_ba_iterations
        self.poses={};self.xyz={};self.observations={};self.history=[];self.registration=[];self.baseline=None
        self.by_image=[sorted(set(f.values())) for f in feature_tracks]

    def initialize(self):
        ranked=sorted(self.pairs,key=lambda p:len(self.pairs[p][0]),reverse=True)
        candidates=[]
        for i,j in ranked[:60]:
            a,b,_=self.pairs[i,j];p,q=self.pixels[i][a],self.pixels[j][b]
            E,mask=cv2.findEssentialMat(p,q,self.K,method=cv2.RANSAC,prob=.999,threshold=1.5)
            if E is None or E.shape!=(3,3) or mask is None:continue
            _,R,t,inliers=cv2.recoverPose(E,p,q,self.K,mask=mask.copy())
            idx=np.flatnonzero(inliers.ravel())
            if len(idx)<50:continue
            P0=self.K@np.c_[np.eye(3),np.zeros(3)];P1=self.K@np.c_[R,t]
            h=cv2.triangulatePoints(P0,P1,p[idx].T,q[idx].T)
            with np.errstate(divide='ignore',invalid='ignore'):X=(h[:3]/h[3]).T
            Y=X@R.T+t.ravel();c=-R.T@t.ravel()
            ray1=X/np.linalg.norm(X,axis=1,keepdims=True);ray2=(X-c)/np.linalg.norm(X-c,axis=1,keepdims=True)
            angle=np.degrees(np.arccos(np.clip(np.sum(ray1*ray2,axis=1),-1,1)))
            good=np.isfinite(X).all(axis=1)&(X[:,2]>0)&(Y[:,2]>0)&(angle>=1.5)&(angle<=178.5)
            if good.sum()<50 or np.median(angle[good])<4:continue
            H,hm=cv2.findHomography(p,q,cv2.RANSAC,3.)
            hratio=float(hm.mean()) if hm is not None else 0.
            connected=sum(len(self.tracks[self.feature_tracks[i][int(k)]])>=3
                          for k in a[idx[good]] if int(k) in self.feature_tracks[i])
            score=float((connected+.1*good.sum())*(1-.5*hratio))
            candidates.append((score,i,j,R,t.ravel(),int(good.sum()),float(np.median(angle[good])),hratio))
        if not candidates:raise RuntimeError('No reliable basic essential-matrix baseline')
        _,i,j,R,t,n,angle,hr=max(candidates,key=lambda c:c[0]);self.baseline=[i,j]
        self.poses={i:(np.eye(3),np.zeros(3)),j:(R,t)}
        self.retriangulate()
        LOG.info('Basic baseline %d-%d: %d points; angle %.2f deg; H support %.2f',i,j,len(self.xyz),angle,hr)
        self.registration.extend([{'image_id':i,'method':'essential'},{'image_id':j,'method':'essential'}])
        return [{'images':[c[1],c[2]],'score':c[0],'valid_points':c[5],'median_angle_deg':c[6],'homography_ratio':c[7]} for c in sorted(candidates,key=lambda c:-c[0])]

    def triangulate_track(self, tid):
        obs={i:k for i,k in self.tracks[tid].items() if i in self.poses}
        if len(obs)<2:return
        ids=list(obs);ps=[self.poses[i] for i in ids];uv=np.array([self.pixels[i][obs[i]] for i in ids])
        rays=cv2.undistortPoints(uv.reshape(-1,1,2),self.K,None).reshape(-1,2)
        rays=np.c_[rays,np.ones(len(rays))]
        rays=np.array([r@R for r,(R,t) in zip(rays,ps)]);rays/=np.linalg.norm(rays,axis=1,keepdims=True)
        choices=[]
        for a,b in combinations(range(len(ids)),2):
            angle=np.degrees(np.arccos(np.clip(rays[a]@rays[b],-1,1)))
            if 1.5<=angle<=178.5:choices.append((min(angle,180-angle),a,b))
        if not choices:
            self.xyz.pop(tid,None);self.observations.pop(tid,None);return
        hypotheses=[]
        if tid in self.xyz:hypotheses.append(self.xyz[tid])
        for _,a,b in sorted(choices,reverse=True)[:3]:
            hypotheses.append(triangulate_dlt([ps[a],ps[b]],uv[[a,b]],self.K))
        best=None
        for X in hypotheses:
            if not np.isfinite(X).all():continue
            error,z=point_errors(X,ps,uv,self.K);keep=(error<=3.)&(z>0)
            if keep.sum()<2:continue
            rank=(int(keep.sum()),-float(np.median(error[keep])))
            if best is None or rank>best[0]:best=(rank,X,keep)
        if best is None:
            self.xyz.pop(tid,None);self.observations.pop(tid,None);return
        _,X,keep=best
        refined=triangulate_dlt([ps[k] for k in np.flatnonzero(keep)],uv[keep],self.K)
        if np.isfinite(refined).all():
            er,z=point_errors(refined,ps,uv,self.K);new=(er<=3.)&(z>0)
            if new.sum()>=keep.sum():X,keep=refined,new
        centers=np.array([-ps[k][0].T@ps[k][1] for k in np.flatnonzero(keep)])
        rr=X-centers;rr/=np.linalg.norm(rr,axis=1,keepdims=True)
        angle=np.degrees(np.arccos(np.clip(rr@rr.T,-1,1)));angle=np.minimum(angle,180-angle).max()
        if angle<1.5:
            self.xyz.pop(tid,None);self.observations.pop(tid,None);return
        self.xyz[tid]=X;self.observations[tid]={ids[k]:obs[ids[k]] for k in np.flatnonzero(keep)}

    def retriangulate(self, image_id=None):
        tids=range(len(self.tracks)) if image_id is None else self.by_image[image_id]
        for tid in tids:self.triangulate_track(tid)

    def correspondences(self, image_id):
        tids=[t for t in self.by_image[image_id] if t in self.xyz]
        X=np.array([self.xyz[t] for t in tids]).reshape(-1,3)
        uv=np.array([self.pixels[image_id][self.tracks[t][image_id]] for t in tids]).reshape(-1,2)
        return tids,X,uv

    def register(self, image_id):
        tids,X,uv=self.correspondences(image_id)
        if len(X)<12:return False
        ok,r,t,inside=cv2.solvePnPRansac(X,uv,self.K,None,iterationsCount=2000,reprojectionError=4.,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
        if not ok or inside is None or len(inside)<12 or len(inside)<.2*len(X):return False
        idx=inside.ravel();r,t=cv2.solvePnPRefineLM(X[idx],uv[idx],self.K,None,r,t)
        R=cv2.Rodrigues(r)[0];pose=(R,t.ravel())
        pred=cv2.projectPoints(X,r,t,self.K,None)[0].reshape(-1,2);err=np.linalg.norm(pred-uv,axis=1)
        keep=(err<=3.)&((X@R.T+t.ravel())[:,2]>0)
        if keep.sum()<12:return False
        previous_xyz=self.xyz.copy()
        previous_obs={tid:obs.copy() for tid,obs in self.observations.items()}
        try:
            self.poses[image_id]=pose
            for k in np.flatnonzero(keep):self.observations[tids[k]][image_id]=self.tracks[tids[k]][image_id]
            self.retriangulate(image_id)
        except Exception:
            self.poses.pop(image_id,None)
            self.xyz=previous_xyz;self.observations=previous_obs
            raise
        self.registration.append({'image_id':image_id,'method':'opencv_pnp_ransac','correspondences':len(X),'inliers':int(keep.sum()),'mean_inlier_error_px':float(err[keep].mean())})
        LOG.info('Registered %d: %d/%d PnP inliers; cameras=%d points=%d',image_id,keep.sum(),len(X),len(self.poses),len(self.xyz))
        return True

    def bundle(self, label='global'):
        if not self.xyz:raise RuntimeError('No valid tracks remain for bundle adjustment')
        ids=sorted(self.poses);tids=sorted(self.xyz);ci=[];pi=[];uv=[]
        mapping={c:i for i,c in enumerate(ids)}
        for p,tid in enumerate(tids):
            for c,key in self.observations[tid].items():ci.append(mapping[c]);pi.append(p);uv.append(self.pixels[c][key])
        LOG.info('Basic BA %s: cameras=%d points=%d observations=%d',label,len(ids),len(tids),len(ci))
        ba=SparseBundleAdjuster(self.K,[self.poses[c] for c in ids],[self.xyz[t] for t in tids],ci,pi,uv,self.refine_focal)
        poses,xyz,K,stats=ba.optimize(self.max_ba_iterations)
        self.K=K;self.poses.update(zip(ids,poses));self.xyz.update(zip(tids,xyz))
        stats.update(label=label,cameras=len(ids),points=len(tids),observations=len(ci),focal_px=float(K[0,0]))
        self.history.append(stats)
        LOG.info('Basic BA %.3f -> %.3f px; steps=%d; stop=%s; focal=%.2f',stats['mean_error_before_px'],stats['mean_error_after_px'],stats['successful_steps'],stats['termination'],K[0,0])
        return stats

    def filter_points(self, min_views=2):
        for tid in list(self.xyz):
            obs=self.observations[tid];ids=list(obs);ps=[self.poses[c] for c in ids];uv=np.array([self.pixels[c][obs[c]] for c in ids])
            e,z=point_errors(self.xyz[tid],ps,uv,self.K);keep=(e<=3.)&(z>0)
            valid={ids[k]:obs[ids[k]] for k in np.flatnonzero(keep)}
            if len(valid)<min_views:
                self.xyz.pop(tid);self.observations.pop(tid);continue
            rr=np.array([self.xyz[tid]+R.T@t for R,t in [self.poses[c] for c in valid]])
            rr/=np.linalg.norm(rr,axis=1,keepdims=True);angle=np.degrees(np.arccos(np.clip(rr@rr.T,-1,1)))
            if np.minimum(angle,180-angle).max()<1.5:
                self.xyz.pop(tid);self.observations.pop(tid);continue
            self.observations[tid]=valid

    def grow(self, save_history=None):
        pending=set(range(len(self.images)))-self.poses.keys();since=0
        while pending:
            order=sorted(pending,key=lambda i:-len(self.correspondences(i)[0]));added=False
            for i in order:
                if self.register(i):
                    pending.remove(i);since+=1;added=True;break
            if not added:
                LOG.warning('No registration progress for frames %s; retry after global BA/track completion',sorted(pending))
                self.bundle('registration_retry');self.filter_points();self.retriangulate()
                if save_history:save_history()
                for i in order:
                    if self.register(i):pending.remove(i);since+=1;added=True;break
                if not added:break
            if since>=5 or len(self.poses)==3:
                self.bundle();self.filter_points();self.retriangulate();since=0
                if save_history:save_history()
        self.bundle('final_before_track_filter');self.filter_points();self.retriangulate()
        self.filter_points(min_views=3 if len(self.poses)>=3 else 2)
        self.finalize()
        if save_history:save_history()

    def finalize(self):
        """Stop only when BA and filtering retain exactly the same observations.

        Otherwise the last BA can move a boundary point below the angle/depth
        gate, or the exported observations would differ from those optimized.
        """
        def signature():
            return tuple((t,tuple(sorted(self.observations[t].items()))) for t in sorted(self.xyz))
        for i in range(10):
            before=signature()
            self.bundle('final_filtered_%d'%i)
            self.filter_points(min_views=3 if len(self.poses)>=3 else 2)
            if signature()==before:return
        raise RuntimeError('Final BA/filter observations did not stabilize in 10 passes')

    def export(self):
        points=[];colors=[];rows=[]
        for tid in sorted(self.xyz):
            obs=self.observations[tid];rgb=[]
            for c,key in obs.items():
                u,v=np.rint(self.pixels[c][key]).astype(int);im=self.images[c]
                rgb.append(im[np.clip(v,0,im.shape[0]-1),np.clip(u,0,im.shape[1]-1),::-1].astype(float))
            rows.append({'point_id':len(points),'track_id':tid,'views':obs,'pixels':{c:self.pixels[c][key].tolist() for c,key in obs.items()}})
            points.append(self.xyz[tid]);colors.append(np.mean(rgb,axis=0)/255.)
        return np.asarray(points),np.asarray(colors),rows
