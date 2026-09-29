"""Small colored-vertex PLY I/O and explicit point processing using NumPy."""
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree


def write_ply(path, xyz, rgb):
    xyz = np.asarray(xyz, dtype=np.float64)
    rgb = np.asarray(rgb)
    if rgb.dtype.kind == 'f':
        rgb = np.rint(np.clip(rgb, 0, 1) * 255)
    assert xyz.shape == rgb.shape and xyz.ndim == 2 and xyz.shape[1] == 3
    assert np.isfinite(xyz).all()
    dtype = np.dtype([(n, '<f8') for n in ['x', 'y', 'z']] + [(n, 'u1') for n in ['red', 'green', 'blue']])
    values = np.empty(len(xyz), dtype=dtype)
    for i, n in enumerate(['x', 'y', 'z']): values[n] = xyz[:, i]
    for i, n in enumerate(['red', 'green', 'blue']): values[n] = np.clip(rgb[:, i], 0, 255)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        header = 'ply\nformat binary_little_endian 1.0\nelement vertex %d\n' % len(xyz)
        header += ''.join('property double %s\n' % n for n in ['x', 'y', 'z'])
        header += ''.join('property uchar %s\n' % n for n in ['red', 'green', 'blue'])
        f.write((header + 'end_header\n').encode()); values.tofile(f)


def read_ply(path):
    types = {'float': 'f4', 'double': 'f8', 'uchar': 'u1', 'uint8': 'u1', 'int': 'i4'}
    with Path(path).open('rb') as f:
        assert f.readline().strip() == b'ply'
        properties = []; count = None; vertex = False; fmt = None
        while True:
            raw = f.readline()
            if not raw: raise ValueError('Incomplete PLY header')
            s = raw.decode().strip().split()
            if not s: continue
            if s[0] == 'format': fmt = s[1]
            if s[0] == 'element':
                vertex = s[1] == 'vertex'
                if vertex: count = int(s[2])
            if s[0] == 'property' and vertex: properties.append((s[2], types[s[1]]))
            if s[0] == 'end_header': break
        if fmt == 'ascii':
            a = np.array([[float(v) for v in f.readline().split()] for _ in range(count)])
            fields = {n: a[:, i] for i, (n, _) in enumerate(properties)}
        elif fmt == 'binary_little_endian':
            fields = np.fromfile(f, dtype=np.dtype([(n, '<'+t) for n, t in properties]), count=count)
        else: raise ValueError('Unsupported PLY format')
        xyz = np.column_stack([fields[n] for n in ['x', 'y', 'z']]).astype(float)
        names = [n for n, _ in properties]
        rgb = np.column_stack([fields[n] for n in ['red', 'green', 'blue']]) / 255. if 'red' in names else np.ones_like(xyz)
        return xyz, rgb


def fuse_points(xyz, rgb, voxel, neighbors=20, std_ratio=2.):
    """Voxel averaging followed by an explicitly computed neighbor-distance gate."""
    if voxel <= 0: raise ValueError('Voxel size must be positive')
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, inverse, count = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    points = np.column_stack([np.bincount(inverse, weights=xyz[:, k]) / count for k in range(3)])
    colors = np.column_stack([np.bincount(inverse, weights=rgb[:, k]) / count for k in range(3)])
    if len(points) < 3: return points, colors
    tree = cKDTree(points); distances = []
    for start in range(0, len(points), 50000):
        d, _ = tree.query(points[start:start+50000], k=min(neighbors+1, len(points)), workers=1)
        distances.append(d[:, 1:].mean(axis=1))
    mean = np.concatenate(distances); keep = mean <= mean.mean() + std_ratio * mean.std()
    return points[keep], colors[keep]
