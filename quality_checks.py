"""Geometric checks independent of solver success and frame registration count."""
import numpy as np


def sampson_errors(K, pose_i, pose_j, pixels_i, pixels_j):
    Ri, ti = pose_i
    Rj, tj = pose_j
    R = Rj @ Ri.T
    t = np.asarray(tj).ravel() - R @ np.asarray(ti).ravel()
    a, b, c = t
    cross = np.array([[0., -c, b], [c, 0., -a], [-b, a, 0.]])
    inv = np.linalg.inv(K)
    F = inv.T @ cross @ R @ inv
    p = np.c_[pixels_i, np.ones(len(pixels_i))]
    q = np.c_[pixels_j, np.ones(len(pixels_j))]
    fp, fq = p @ F.T, q @ F
    denominator = np.sqrt(np.sum(fp[:, :2] ** 2 + fq[:, :2] ** 2, axis=1))
    return np.abs(np.sum(q * fp, axis=1)) / np.maximum(denominator, 1e-15)


def summarize_geometry(points, poses, observations, K, adjacent_pairs):
    """Observations use the exported OpenCV pixel-center convention throughout."""
    errors, per_camera, negative, angles = [], {i: [] for i in poses}, 0, []
    centers = {i: -R.T @ np.asarray(t).ravel() for i, (R, t) in poses.items()}
    point_sets = {i: set() for i in poses}
    for point_id, (X, observation) in enumerate(zip(points, observations)):
        rays = []
        for cam, pixel in observation.items():
            R, t = poses[cam]
            Y = R @ X + np.asarray(t).ravel()
            negative += int(Y[2] <= 0)
            projected = K @ Y
            error = float(np.linalg.norm(projected[:2] / projected[2] - pixel))
            errors.append(error)
            per_camera[cam].append(error)
            point_sets[cam].add(point_id)
            ray = X - centers[cam]
            rays.append(ray / np.linalg.norm(ray))
        rays = np.asarray(rays)
        pair_angles = np.degrees(np.arccos(np.clip(rays @ rays.T, -1., 1.)))
        angles.append(float(np.minimum(pair_angles, 180 - pair_angles).max()))
    errors, angles = np.asarray(errors), np.asarray(angles)
    track_lengths = np.array([len(o) for o in observations])
    pair_checks = []
    for i, j, p, q in adjacent_pairs:
        if i not in poses or j not in poses or len(p) < 20:
            pair_checks.append({'images': [i, j], 'matches': len(p), 'checked': False, 'passed': False})
            continue
        distance = sampson_errors(K, poses[i], poses[j], p, q)
        median = float(np.median(distance))
        shared = len(point_sets[i] & point_sets[j])
        pair_checks.append({'images': [i, j], 'matches': len(p), 'checked': True,
                            'shared_points': shared, 'sampson_median_px': median,
                            'sampson_p90_px': float(np.percentile(distance, 90)),
                            'passed': shared >= 15 and median <= 3.})
    return {
        'errors': errors, 'per_camera': per_camera,
        'summary': {
            'nonpositive_depth_observations': negative,
            'mean_track_length': float(track_lengths.mean()),
            'two_view_fraction': float(np.mean(track_lengths == 2)),
            'track_length_histogram': {str(n): int(np.sum(track_lengths == n)) for n in np.unique(track_lengths)},
            'triangulation_angle_min_deg': float(angles.min()),
            'triangulation_angle_median_deg': float(np.median(angles)),
            'points_below_1_5deg': int(np.sum(angles < 1.5 - 1e-6)),
            'adjacent_pairs': pair_checks,
            'all_adjacent_pairs_passed': bool(pair_checks) and all(p['passed'] for p in pair_checks),
        },
    }
