#!/usr/bin/env python3
"""Render camera-oriented point clouds and an offline interactive old/new gallery."""
import argparse
import json
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ply_io import read_ply, write_ply, fuse_points
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load(folder, cloud_name='reconstruction.ply'):
    points,colors=read_ply(folder/cloud_name)
    cameras=json.loads((folder/'cameras.json').read_text())
    first=min(cameras,key=lambda c:c['image_id'])
    R=np.array(first['R_world_to_camera']);t=np.array(first['t_world_to_camera'])
    xyz=points@R.T+t
    center=np.median(xyz,axis=0)
    distance=np.linalg.norm(xyz-center,axis=1)
    keep=distance<=np.percentile(distance,99)
    # Display normalization only: PLY and quantitative coordinates remain unchanged.
    scale=max(np.median(xyz[:,2]),1e-6)
    xyz=xyz/scale
    xyz[:,1]*=-1
    centers=np.array([np.array(c['camera_center'])@R.T+t for c in sorted(cameras,key=lambda c:c['image_id'])])/scale
    centers[:,1]*=-1
    xyz,colors=xyz[keep],colors[keep]
    if len(xyz)>200000:
        selected=np.random.default_rng(0).choice(len(xyz),200000,replace=False)
        xyz,colors=xyz[selected],colors[selected]
    return xyz,colors,centers


def gallery(old,new,title,stereo=False):
    data=[load(old),load(new,'stereo_cloud.ply' if stereo else 'reconstruction.ply')]
    prefix='stereo_' if stereo else ''
    kind='stereo' if stereo else 'sparse'
    views=[(0,1,'Front / first camera axes'),(0,2,'Top / depth'),(2,1,'Side / depth')]
    fig,axes=plt.subplots(2,3,figsize=(15,9),facecolor='#101820')
    for row,(xyz,rgb,centers) in enumerate(data):
        for ax,(a,b,label) in zip(axes[row],views):
            ax.set_facecolor('#101820')
            ax.scatter(xyz[:,a],xyz[:,b],c=rgb,s=2.8,linewidths=0)
            lo=np.percentile(xyz[:,[a,b]],1,axis=0);hi=np.percentile(xyz[:,[a,b]],99,axis=0)
            mid=(lo+hi)/2;span=max(hi-lo)*.55
            ax.set_xlim(mid[0]-span,mid[0]+span);ax.set_ylim(mid[1]-span,mid[1]+span)
            ax.set_aspect('equal');ax.tick_params(colors='#ADB7C5',labelsize=7)
            ax.set_title(('Before' if row==0 else 'After')+' — '+label,color='white')
    fig.suptitle(title+' | before: sparse / after: '+kind+', 99% subset; max200k display points',color='white',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(new/(prefix+'comparison.png'),dpi=150,facecolor=fig.get_facecolor());plt.close(fig)
    # Standalone self-contained HTML: no CDN or internet needed by the viewer.
    interactive=make_subplots(rows=1,cols=2,specs=[[{'type':'scene'},{'type':'scene'}]],subplot_titles=['Before','After'])
    for col,(xyz,rgb,centers) in enumerate(data,1):
        colors=['rgb(%d,%d,%d)'%tuple(c) for c in np.rint(rgb*255).astype(int)]
        interactive.add_trace(go.Scatter3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],mode='markers',
            marker=dict(size=2,color=colors),name='Scene',showlegend=False),row=1,col=col)
        interactive.add_trace(go.Scatter3d(x=centers[:,0],y=centers[:,1],z=centers[:,2],mode='lines+markers',
            marker=dict(size=3,color='#22C9F5'),line=dict(width=2,color='#22C9F5'),name='Cameras '+('before' if col==1 else 'after'),showlegend=True,
            visible='legendonly'),row=1,col=col)
    cam=dict(eye=dict(x=0,y=0,z=-2.2),up=dict(x=0,y=1,z=0))
    interactive.update_layout(title=title+' — 99% subset / max200k points; drag to rotate, scroll to zoom',template='plotly_dark',height=780,
                              scene=dict(aspectmode='data',camera=cam),scene2=dict(aspectmode='data',camera=cam),
                              margin=dict(l=10,r=10,t=65,b=10))
    interactive.write_html(str(new/(prefix+'viewer.html')),include_plotlyjs=True,full_html=True)
    # Large after-only 3D view, oriented from the first camera with a small oblique offset.
    xyz,rgb,_=data[1]
    fig=plt.figure(figsize=(10,8),facecolor='#101820');ax=fig.add_subplot(111,projection='3d')
    ax.set_facecolor('#101820');ax.scatter(xyz[:,0],xyz[:,2],xyz[:,1],c=rgb,s=3,depthshade=False)
    lo=np.percentile(xyz,2,axis=0);hi=np.percentile(xyz,98,axis=0);mid=(lo+hi)/2;half=max(hi-lo)*.55
    ax.set_xlim(mid[0]-half,mid[0]+half);ax.set_ylim(mid[2]-half,mid[2]+half);ax.set_zlim(mid[1]-half,mid[1]+half)
    ax.set_box_aspect((1,1,1));ax.view_init(elev=12,azim=-75);ax.set_axis_off()
    fig.suptitle(title+' — optimized '+kind+' point cloud (99% / max200k display)',color='white',fontsize=14)
    fig.tight_layout();fig.savefig(new/(prefix+'point_cloud.png'),dpi=170,facecolor=fig.get_facecolor());plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('old',type=Path);p.add_argument('new',type=Path);p.add_argument('--title',default='Reconstruction');p.add_argument('--stereo',action='store_true')
    a=p.parse_args();gallery(a.old,a.new,a.title,a.stereo)
