"""Cache only project-computed SIFT/BF data, bound to pixels and matcher source."""
import hashlib
import json
import logging
from pathlib import Path
import cv2
import numpy as np


def extract_and_match(matcher, path=None):
    spec={'images':[hashlib.sha256(im.tobytes()).hexdigest() for im in matcher.images],
          'cv_version':cv2.__version__,'nfeatures':matcher.cfg.nfeatures,'ratio':matcher.cfg.ratio_thresh,
          'mutual':matcher.cfg.mutual_check,'matcher_sha256':hashlib.sha256(Path(__file__).with_name('matching.py').read_bytes()).hexdigest()}
    fingerprint=json.dumps(spec,sort_keys=True)
    if path and Path(path).exists():
        with np.load(path,allow_pickle=False) as data:
            if str(data['fingerprint'])!=fingerprint:raise ValueError('Feature cache does not match current pixels/settings/matcher')
            for i in range(len(matcher.images)):
                matcher.kps.append([cv2.KeyPoint(float(k[0]),float(k[1]),float(k[2]),float(k[3]),float(k[4]),int(k[5]),int(k[6])) for k in data[f'keypoints_{i}']])
                d=data[f'descriptors_{i}'];matcher.des.append(d if len(d) else None)
            for i,j in data['pairs']:
                matcher.matches[int(i),int(j)]=[cv2.DMatch(int(a),int(b),float(d)) for a,b,d in data[f'matches_{i}_{j}']]
        logging.info('Loaded verified raw SIFT/BF cache %s',path)
        return {'cache':'loaded','fingerprint':spec}
    matcher.extract_features();matcher.match_pairs()
    if path:
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        values={'fingerprint':np.array(fingerprint),'pairs':np.array(sorted(matcher.matches))}
        for i,(ks,d) in enumerate(zip(matcher.kps,matcher.des)):
            values[f'keypoints_{i}']=np.array([[*k.pt,k.size,k.angle,k.response,k.octave,k.class_id] for k in ks]).reshape(-1,7)
            values[f'descriptors_{i}']=d if d is not None else np.empty((0,128),np.float32)
        for (i,j),ms in matcher.matches.items():values[f'matches_{i}_{j}']=np.array([[m.queryIdx,m.trainIdx,m.distance] for m in ms]).reshape(-1,3)
        np.savez_compressed(path,**values)
    return {'cache':'computed','fingerprint':spec}
