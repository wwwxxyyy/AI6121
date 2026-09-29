"""Orchestrate project-owned SfM, keeping all intermediate evidence inspectable."""
import json
from pathlib import Path
import cv2
import numpy as np
from track_graph import verified_pairs, build_tracks, estimate_focal
from incremental_sfm import IncrementalSfM
from quality_checks import summarize_geometry
from ply_io import write_ply, read_ply
from scripts.render_ply import render_preview


def write_json(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')


def reconstruct(args,output,matcher,K,intrinsic_info):
    pixels,pairs,matrices,match_stats=verified_pairs(matcher.kps,matcher.matches,args.seed)
    write_json(output/'matching.json',{'keypoints':[len(p) for p in pixels],'mutual_check':True,'verified_pairs':match_stats})
    tracks,feature_tracks,track_stats=build_tracks(matcher.kps,pairs)
    write_json(output/'tracks.json',tracks);write_json(output/'track_statistics.json',track_stats)
    initial_K=K.copy()
    if args.refine_focal:
        K,focal_stats=estimate_focal(K,matrices,pairs,matcher.images[0].shape)
        write_json(output/'focal_initialization.json',focal_stats)
    recon=IncrementalSfM(pixels,pairs,tracks,feature_tracks,K,matcher.images,args.refine_focal,args.ba_max_nfev)
    candidates=recon.initialize();write_json(output/'baseline_candidates.json',candidates)
    recon.grow(lambda:write_json(output/'ba_history.json',recon.history))
    xyz,rgb,observations=recon.export()
    if not len(xyz) or not np.isfinite(xyz).all():raise RuntimeError('Empty/nonfinite basic reconstruction')
    adjacent=[]
    for i in range(len(pixels)-1):
        a,b,_=pairs.get((i,i+1),(np.array([],int),np.array([],int),None))
        adjacent.append((i,i+1,pixels[i][a],pixels[i+1][b]))
    q=summarize_geometry(xyz,recon.poses,[{int(c):v for c,v in p['pixels'].items()} for p in observations],recon.K,adjacent)
    intrinsic_info.update(initial_K=initial_K.tolist(),K=recon.K.tolist(),refined_focal=args.refine_focal,
                          refinement='project-owned shared-focal BA' if args.refine_focal else 'fixed provided/estimated K')
    np.savetxt(output/'K.txt',recon.K,fmt='%.12g');write_json(output/'intrinsics.json',intrinsic_info)
    write_json(output/'observations.json',observations);write_json(output/'registration.json',recon.registration)
    write_json(output/'quality.json',q['summary'])
    cams=[{'image_id':i,'R_world_to_camera':R.tolist(),'t_world_to_camera':t.tolist(),'camera_center':(-R.T@t).tolist()} for i,(R,t) in sorted(recon.poses.items())]
    write_json(output/'cameras.json',cams)
    usage=[]
    for i in range(len(pixels)):
        local=q['per_camera'].get(i,[])
        usage.append({'image_id':i,'registered':i in recon.poses,'registration_method':'basic_essential_or_pnp',
                      'final_ba_observations':len(local),'used_in_final_ba':len(local)>=10,
                      'mean_reprojection_error_px':float(np.mean(local)) if local else None})
    write_json(output/'frame_usage.json',usage)
    write_ply(output/'reconstruction.ply',xyz,rgb);render_preview(output/'reconstruction.ply',output/'preview.png')
    reread,colors=read_ply(output/'reconstruction.ply');ply_valid=len(reread)==len(xyz) and np.isfinite(reread).all()
    errors=q['errors'];all_used=all(x['used_in_final_ba'] for x in usage)
    geometry=(q['summary']['all_adjacent_pairs_passed'] and q['summary']['nonpositive_depth_observations']==0
              and q['summary']['points_below_1_5deg']==0 and errors.mean()<2.)
    return {'engine':'project-owned OpenCV/NumPy/SciPy SfM','status':'passed' if all_used and geometry and ply_valid else 'failed_acceptance',
            'accepted':bool(all_used and geometry and ply_valid),'all_sampled_frames_used':all_used,'geometry_quality_passed':bool(geometry),
            'sampled_frames':len(pixels),'registered_frames':len(recon.poses),'unregistered_image_ids':sorted(set(range(len(pixels)))-recon.poses.keys()),
            'points':len(xyz),'observations':len(errors),'baseline':recon.baseline,
            'final_reprojection_error_px':{'mean':float(errors.mean()),'median':float(np.median(errors)),'rmse':float(np.sqrt(np.mean(errors**2))),'p95':float(np.percentile(errors,95))},
            'ba_runs':len(recon.history),'final_ba_converged':recon.history[-1]['converged'],'ply_readback_valid':bool(ply_valid),
            'preview_valid':cv2.imread(str(output/'preview.png')) is not None,'intrinsics':intrinsic_info,'quality':q['summary'],
            'limitations':['Estimated camera model unless provided; zero distortion','Arbitrary monocular scale','Sparse reconstruction; no geometry ground truth']}
