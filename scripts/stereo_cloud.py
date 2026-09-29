#!/usr/bin/env python3
"""CPU stereo depth + >=3-view depth consistency; supplements the sparse model.

Uses the saved SfM cameras. It neither estimates new poses nor fills missing
surfaces. The sparse PLY and all input frames remain unchanged.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import cv2
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ply_io import read_ply, write_ply, fuse_points


def stereo_depth(left, right, K, R, t, z_range, rectify_alpha=0., disparities=None):
    h,w=left.shape[:2]
    R1,R2,P1,P2,Q,_,_=cv2.stereoRectify(K,None,K,None,(w,h),R,t,flags=cv2.CALIB_ZERO_DISPARITY,alpha=rectify_alpha)
    maps=[cv2.initUndistortRectifyMap(K,None,r,p,(w,h),cv2.CV_32FC1) for r,p in [(R1,P1),(R2,P2)]]
    images=[cv2.remap(im,*m,cv2.INTER_LINEAR) for im,m in zip([left,right],maps)]
    masks=[(mx>=0)&(mx<w-1)&(my>=0)&(my<h-1) for mx,my in maps]
    vertical=abs(P2[1,3])>abs(P2[0,3])
    if vertical:
        images=[cv2.transpose(im) for im in images]
        masks=[mask.T for mask in masks]
    gray=[cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) for im in images]
    hh,ww=gray[0].shape
    nd=max(16,min(192,((ww//3)//16)*16))
    if disparities is not None:
        nd=min(disparities,((ww-16)//16)*16)
    sign=P2[1 if vertical else 0,3]
    mind=0 if sign<0 else -nd
    def compute(a,b,start):
        s=cv2.StereoSGBM_create(minDisparity=start,numDisparities=nd,blockSize=5,
             P1=8*25,P2=32*25,disp12MaxDiff=1,uniquenessRatio=12,
             speckleWindowSize=100,speckleRange=2,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        return s.compute(a,b).astype(np.float32)/16
    dl=compute(gray[0],gray[1],mind);dr=compute(gray[1],gray[0],-mind-nd)
    yy,xx=np.indices(dl.shape);xr=np.rint(xx-dl).astype(int)
    ok=(dl>mind)&(np.abs(dl)>1)&(xr>=0)&(xr<ww)&masks[0]
    indices=np.flatnonzero(ok);flat_dr=dr.ravel();flat_x=xr.ravel();flat_y=yy.ravel()
    valid=np.zeros(dl.size,bool)
    right_indices=flat_y[indices]*ww+flat_x[indices]
    valid[indices]=(np.abs(dl.ravel()[indices]+flat_dr[right_indices])<=1.)&masks[1].ravel()[right_indices]
    valid=valid.reshape(dl.shape)
    if vertical:dl=dl.T;valid=valid.T
    rect=cv2.reprojectImageTo3D(dl,Q)
    with np.errstate(invalid='ignore'):
        camera=rect@R1
    valid&=np.isfinite(camera).all(axis=2)&(camera[:,:,2]>z_range[0])&(camera[:,:,2]<z_range[1])
    X=camera[valid].astype(float)
    uv=X@K.T;uv=np.rint(uv[:,:2]/uv[:,2:]).astype(int)
    inside=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    X=X[inside];uv=uv[inside]
    depth=np.full((h,w),np.inf,np.float32)
    np.minimum.at(depth,(uv[:,1],uv[:,0]),X[:,2])
    depth[~np.isfinite(depth)]=0
    return depth


def run(folder,max_side=800,rectify_alpha=0.,disparities=None):
    start=time.perf_counter();cv2.setNumThreads(1)
    load=lambda p:json.loads((folder/p).read_text())
    assert load('metrics.json')['accepted'],'Only densify a model that passed geometric checks'
    check=load('independent_verification.json')
    assert check['all_frames_verified'] and check['all_adjacent_pairs_passed']
    frames=load('frames.json');cams=sorted(load('cameras.json'),key=lambda c:c['image_id'])
    assert [c['image_id'] for c in cams]==list(range(len(frames)))
    images=[cv2.imread(str(folder/f['file'])) for f in frames]
    original_h,original_w=images[0].shape[:2]
    scale=min(1.,max_side/max(original_h,original_w))
    w,h=int(round(original_w*scale)),int(round(original_h*scale))
    sx,sy=w/original_w,h/original_h
    images=[cv2.resize(im,(w,h),interpolation=cv2.INTER_AREA) for im in images]
    K=np.loadtxt(folder/'K.txt');K[0,:]*=sx;K[1,:]*=sy
    # cv2.resize maps pixel centers with a half-pixel offset.
    K[0,2]+=(sx-1)/2;K[1,2]+=(sy-1)/2
    R=[np.array(c['R_world_to_camera']) for c in cams];t=[np.array(c['t_world_to_camera']) for c in cams]
    sparse,_=read_ply(folder/'reconstruction.ply')
    depth=[];pair_counts=[]
    for i in range(len(images)):
        j=i+1 if i+1<len(images) else i-1
        relR=R[j]@R[i].T;relt=t[j]-relR@t[i]
        z=(sparse@R[i].T+t[i])[:,2];z=z[z>0]
        zrange=(float(np.percentile(z,1)*.5),float(np.percentile(z,99)*1.5))
        d=stereo_depth(images[i],images[j],K,relR,relt,zrange,rectify_alpha,disparities)
        depth.append(d);pair_counts.append(int(np.count_nonzero(d)))
        print(f'depth {i+1}/{len(images)}: {pair_counts[-1]} consistent stereo pixels',flush=True)
    np.savez_compressed(folder/'stereo_depths.npz',depth=np.array(depth),K=K)
    xyz_all=[];rgb_all=[];accepted=[];supports=[]
    for i,d in enumerate(depth):
        v,u=np.nonzero(d);z=d[v,u]
        Xcam=np.c_[(u-K[0,2])*z/K[0,0],(v-K[1,2])*z/K[1,1],z]
        X=(Xcam-t[i])@R[i]
        count=np.zeros(len(X),int)
        for j in range(max(0,i-2),min(len(images),i+3)):
            if i==j:continue
            Y=X@R[j].T+t[j];p=Y@K.T
            uv=np.rint(p[:,:2]/p[:,2:]).astype(int)
            ok=(Y[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
            idx=np.flatnonzero(ok);sample=depth[j][uv[idx,1],uv[idx,0]]
            count[idx]+=(sample>0)&(np.abs(sample-Y[idx,2])/Y[idx,2]<.02)
        keep=count>=2
        xyz_all.append(X[keep]);rgb_all.append(images[i][v[keep],u[keep],::-1]/255.)
        accepted.append(int(keep.sum()));supports.extend((count[keep]+1).tolist())
    xyz=np.vstack(xyz_all);rgb=np.vstack(rgb_all)
    if len(xyz)<100:raise RuntimeError('Insufficient 3-view consistent stereo points')
    centers=np.array([-a.T@b for a,b in zip(R,t)])
    voxel=float(np.median(np.linalg.norm(np.diff(centers,axis=0),axis=1))*.01)
    fused,colors=fuse_points(xyz,rgb,voxel)
    path=folder/'stereo_cloud.ply';write_ply(path,fused,colors)
    reread,reread_colors=read_ply(path)
    assert len(reread)>100 and np.isfinite(reread).all()
    assert len(reread_colors)==len(reread)
    stats={'method':'OpenCV SGBM + left/right check <=1px + >=3-view depth consistency at 2%',
           'rectification_alpha':rectify_alpha,'requested_disparities':disparities,
           'input_frames':len(images),'processing_resolution':[w,h],'stereo_pixels_per_frame':pair_counts,
           'three_view_points_per_frame':accepted,'consistent_samples_before_fusion':len(xyz),
           'minimum_view_support_before_fusion':min(supports),'points':len(reread),'voxel_size':voxel,
           'source':'Measured image stereo depth; no interpolation or generated geometry',
           'source_sha256':{name:hashlib.sha256((folder/name).read_bytes()).hexdigest()
                            for name in ['K.txt','cameras.json','reconstruction.ply','frames.json','config.json']},
           'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'limitations':['Approximate camera model', 'Textureless/specular surfaces can have holes', 'No metric scale or ground truth'],
           'elapsed_seconds':time.perf_counter()-start}
    (folder/'stereo_metrics.json').write_text(json.dumps(stats,indent=2)+'\n');print(json.dumps(stats),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);parser.add_argument('--max-side',type=int,default=800)
    parser.add_argument('--rectify-alpha',type=float,default=0.,help='OpenCV stereoRectify alpha; -1 preserves the default focal scale without zoom-to-crop')
    parser.add_argument('--disparities',type=int,default=None,help='Optional SGBM disparity count, positive multiple of16')
    args=parser.parse_args()
    if args.max_side<64 or not -1<=args.rectify_alpha<=1 or (args.disparities is not None and (args.disparities<16 or args.disparities%16)):
        parser.error('max-side>=64, -1<=rectify-alpha<=1, disparities must be a positive multiple of16')
    run(args.folder,args.max_side,args.rectify_alpha,args.disparities)
