import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@dataclass
class Point3DView:
    coords: np.ndarray
    rgb: Optional[Tuple[int, int, int]] = None  # RGB color of the point
    observations: Dict[int, int] = field(default_factory=dict)

@dataclass
class ReconstructionCfg:
    K: np.ndarray
    min_inliers_baseline: int = 10
    essential_ransac_thresh: float = 2.0
    pnp_reproj_thresh: float = 4.0
    pnp_iterations: int = 100  # Reduced from 1000
    pnp_method: int = cv2.SOLVEPNP_EPNP  # Faster than P3P
    min_pnp_correspondences: int = 10
    max_failed_attempts: int = 3
    bundle_every: int = 5  # Less frequent bundle adjustment
    verbose: bool = True
    num_threads: int = 12  # For multithreading
    optical_flow_fallback: bool = False

class Reconstruction:
    def __init__(
        self,
        keypoints: List[List[cv2.KeyPoint]],
        matches: Dict[Tuple[int, int], List[cv2.DMatch]],
        img_adjacency: np.ndarray,
        cfg: ReconstructionCfg,
        images: List[np.ndarray]
    ):
        # OpenCV returns tuples in some builds; recovered optical-flow observations
        # need append/rollback without mutating the caller's original SIFT collection.
        self.keypoints = [list(view) for view in keypoints]
        self.matches = matches
        self.adjacency = img_adjacency
        self.cfg = cfg
        self.images = images
        self.poses: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.points3d: List[Point3DView] = []
        self.placed: List[int] = []
        self.unplaced: List[int] = list(range(img_adjacency.shape[0]))
        self.failed_attempts: Dict[int, int] = {}
        self.registration_failures: Dict[int, int] = {}
        self.registration_methods: Dict[int, str] = {}
        self.add_count = 0

    def select_baseline(self, top_percent: float = 0.3,
                    min_parallax_deg: float = 2.0) -> Tuple[int, int]:
        """
        Select a baseline image pair for initialization.
    
        Args:
            top_percent: optional, unused here but could be used if you want
                         to keep the top-k% pairs.
            min_parallax_deg: minimum median parallax (in degrees)
                              required to accept a pair.
        Returns:
            (i, j): indices of selected baseline pair
        """
    
        def compute_parallax(pts_i, pts_j, K, R, t, mask):
            # OpenCV commonly encodes recoverPose inliers as 255 rather than 1.
            # Treat every non-zero entry as an inlier so the parallax set is not
            # accidentally emptied.
            inliers = mask.ravel().astype(bool)
            pts_i = pts_i[inliers]
            pts_j = pts_j[inliers]
            if len(pts_i) == 0:
                return 0.0
            # undistort & normalize to bearing vectors
            v1 = cv2.undistortPoints(pts_i.reshape(-1,1,2), K, None).reshape(-1,2)
            v2 = cv2.undistortPoints(pts_j.reshape(-1,1,2), K, None).reshape(-1,2)
            v1 = np.hstack([v1, np.ones((v1.shape[0],1))])
            v2 = np.hstack([v2, np.ones((v2.shape[0],1))])
            # recoverPose maps camera-i rays into camera-j coordinates. Rotate
            # camera-j rays back into camera-i coordinates before comparing them.
            v2_rot = (R.T @ v2.T).T
            cos_angle = np.sum(v1 * v2_rot, axis=1) / (
                np.linalg.norm(v1, axis=1) * np.linalg.norm(v2_rot, axis=1)
            )
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            return np.median(np.arccos(cos_angle))  # radians
    
        scores = []
        for (i, j), mlist in self.matches.items():
            if len(mlist) < self.cfg.min_inliers_baseline:
                continue
            pts_i, pts_j = self._aligned_points(i, j)
            E, mask = cv2.findEssentialMat(
                pts_i, pts_j, self.cfg.K,
                method=cv2.FM_RANSAC,
                threshold=self.cfg.essential_ransac_thresh
            )
            if E is None or E.shape != (3, 3) or mask is None or np.count_nonzero(mask) < self.cfg.min_inliers_baseline:
                continue
            _, R, t, out_mask = cv2.recoverPose(E, pts_i, pts_j, self.cfg.K, mask=mask.copy())
            parallax = compute_parallax(pts_i, pts_j, self.cfg.K, R, t, out_mask)
            parallax_deg = np.degrees(parallax)
    
            # filter by parallax
            if parallax_deg < min_parallax_deg:
                logger.debug(f"Rejected pair {(i, j)}: parallax {parallax_deg:.2f}° < {min_parallax_deg}°")
                continue
    
            inlier_count = int(np.count_nonzero(out_mask))
            if inlier_count < self.cfg.min_inliers_baseline:
                continue
            scores.append(((i, j), len(mlist), inlier_count, parallax_deg))
    
        if not scores:
            raise RuntimeError("No valid baseline pair found (all had too little parallax).")
    
        # sort by inliers, then by parallax
        scores.sort(key=lambda x: (x[2], x[3]), reverse=True)
        best = scores[0]
        logger.info(f"Baseline pair: {best[0]} with {best[1]} matches, "
                    f"{best[2]} inliers, median parallax {best[3]:.2f}°")
        return best[0]


    def initialize(self, baseline: Tuple[int, int]) -> None:
        i, j = baseline
        pts_i, pts_j, idxs_i, idxs_j = self._aligned_points(i, j, return_idxs=True)
        E, mask = cv2.findEssentialMat(
            pts_i, pts_j, self.cfg.K, method=cv2.RANSAC,
            threshold=self.cfg.essential_ransac_thresh,
        )
        minimum = max(8, self.cfg.min_inliers_baseline)
        if E is None or E.shape != (3, 3) or mask is None or np.count_nonzero(mask) < minimum:
            raise ValueError("Baseline failed due to insufficient essential-matrix inliers")
        _, rotation, translation, pose_mask = cv2.recoverPose(
            E, pts_i, pts_j, self.cfg.K, mask=mask.copy()
        )
        inliers = pose_mask.ravel().astype(bool)
        if inliers.sum() < minimum:
            raise ValueError("Pose recovery failed with low inliers")
        self.poses[i] = (np.eye(3), np.zeros((3, 1)))
        self.poses[j] = (rotation, translation)
        try:
            self._triangulate_and_add(i, j, idxs_i[inliers], idxs_j[inliers])
        except Exception:
            self.poses.pop(i, None)
            self.poses.pop(j, None)
            raise
        self.placed = [i, j]
        self.registration_methods.update({i: "essential", j: "essential"})
        self.unplaced = [k for k in self.unplaced if k not in self.placed]
        logger.info("Initialized baseline %s with %d points", baseline, len(self.points3d))

    def grow(self, bundle_adjust_fn=None, pbar=None):
        # A failed candidate must not prevent other views from adding new tracks.
        # Low-correspondence views remain pending and are reconsidered after growth.
        while self.unplaced:
            candidates = sorted(
                ((i, self._count_correspondences(i)) for i in self.unplaced
                 if self.failed_attempts.get(i, 0) < self.cfg.max_failed_attempts),
                key=lambda item: (-item[1], item[0]),
            )
            eligible = [(i, count) for i, count in candidates
                        if count >= self.cfg.min_pnp_correspondences
                        or (self.cfg.optical_flow_fallback and any(abs(p-i) <= 2 for p in self.placed))]
            if not eligible:
                logger.info("No remaining view has enough unique 2D/3D correspondences")
                break
            added = False
            for img_idx, count in eligible:
                try:
                    self._add_image(img_idx)
                except (ValueError, RuntimeError, cv2.error) as exc:
                    self.failed_attempts[img_idx] = self.failed_attempts.get(img_idx, 0) + 1
                    self.registration_failures[img_idx] = self.registration_failures.get(img_idx, 0) + 1
                    logger.warning("Registration %d failed (%d/%d): %s", img_idx,
                                   self.failed_attempts[img_idx], self.cfg.max_failed_attempts, exc)
                    continue
                added = True
                self.add_count += 1
                # Newly registered views add tracks and change the PnP problem.
                # A failure against an older map must not permanently exclude a frame.
                self.failed_attempts.clear()
                if pbar is not None:
                    pbar.update(1)
                # BA errors are pipeline errors, not failed camera registrations.
                if bundle_adjust_fn and self.add_count % self.cfg.bundle_every == 0:
                    bundle_adjust_fn()
                break
            if not added:
                logger.info("No view registered in this pass; retrying within the configured limit")

    def _correspondences(self, img_idx):
        # Index observations instead of scanning every match for every 3D point.
        observed = {(cam, key): point_id for point_id, pt in enumerate(self.points3d)
                    for cam, key in pt.observations.items()}
        candidates = []
        for placed in self.placed:
            for m in self.matches.get(tuple(sorted((placed, img_idx))), []):
                q, t = (m.queryIdx, m.trainIdx) if placed < img_idx else (m.trainIdx, m.queryIdx)
                point_id = observed.get((placed, q))
                if point_id is not None and 0 <= t < len(self.keypoints[img_idx]):
                    candidates.append((m.distance, point_id, t))
        # A point or keypoint contributes at most once to PnP.
        seen_points, seen_keys, result = set(), set(), []
        for _, point_id, key in sorted(candidates):
            if point_id not in seen_points and key not in seen_keys:
                result.append((point_id, key))
                seen_points.add(point_id)
                seen_keys.add(key)
        return result

    def _count_correspondences(self, img_idx: int) -> int:
        return len(self._correspondences(img_idx))

    def _flow_correspondences(self, img_idx):
        """Track existing map observations into a weak SIFT frame; do not assign a pose."""
        neighbors = sorted((p for p in self.placed if abs(p-img_idx) <= 2),
                           key=lambda p: (abs(p-img_idx), p))[:2]
        target = self.images[img_idx]
        target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY) if target.ndim == 3 else target
        height, width = target.shape
        options = dict(winSize=(21, 21), maxLevel=3,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
        candidates = []
        for source_id in neighbors:
            observed = [(i, pt.observations[source_id]) for i, pt in enumerate(self.points3d)
                        if source_id in pt.observations]
            if not observed:
                continue
            source = self.images[source_id]
            source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY) if source.ndim == 3 else source
            original = np.asarray([self.keypoints[source_id][key].pt for _, key in observed],
                                  dtype=np.float32).reshape(-1, 1, 2)
            tracked, status, error = cv2.calcOpticalFlowPyrLK(source, target, original, None, **options)
            if tracked is None:
                continue
            xy = tracked.reshape(-1, 2)
            valid = (status.ravel().astype(bool) & np.isfinite(xy).all(axis=1)
                     & (error.ravel() < 20) & (xy[:, 0] >= 0) & (xy[:, 0] < width)
                     & (xy[:, 1] >= 0) & (xy[:, 1] < height))
            selected = np.flatnonzero(valid)
            if not len(selected):
                continue
            back, back_status, _ = cv2.calcOpticalFlowPyrLK(target, source, tracked[selected], None, **options)
            if back is None:
                continue
            fb_error = np.linalg.norm(back.reshape(-1, 2) - original[selected, 0], axis=1)
            for local in np.flatnonzero(back_status.ravel().astype(bool) & (fb_error < 1.0)):
                index = selected[local]
                candidates.append((float(fb_error[local]), observed[index][0], xy[index]))
        used_points, used_pixels, result = set(), set(), []
        for _, point_id, pixel in sorted(candidates, key=lambda c: c[0]):
            cell = tuple(np.rint(pixel).astype(int))
            if point_id not in used_points and cell not in used_pixels:
                result.append((point_id, pixel.astype(np.float64)))
                used_points.add(point_id)
                used_pixels.add(cell)
        return result

    def _add_image(self, img_idx: int, pbar=None) -> None:
        correspondences = self._correspondences(img_idx)
        minimum = max(4, self.cfg.min_pnp_correspondences)
        pixels = np.asarray([self.keypoints[img_idx][k].pt for _, k in correspondences], dtype=np.float64).reshape(-1, 2)

        def estimate(correspondences, pixels):
            if len(correspondences) < minimum:
                return False, None, None, None
            points = np.asarray([self.points3d[p].coords for p, _ in correspondences], dtype=np.float64)
            return cv2.solvePnPRansac(
                points, pixels, self.cfg.K, distCoeffs=None,
                iterationsCount=self.cfg.pnp_iterations,
                reprojectionError=self.cfg.pnp_reproj_thresh, flags=self.cfg.pnp_method,
            )

        success, rvec, tvec, inliers = estimate(correspondences, pixels)
        use_flow = False
        if (not success or inliers is None or len(inliers) < minimum) and self.cfg.optical_flow_fallback:
            tracked = self._flow_correspondences(img_idx)
            correspondences = [(point_id, None) for point_id, _ in tracked]
            pixels = np.asarray([pixel for _, pixel in tracked], dtype=np.float64).reshape(-1, 2)
            success, rvec, tvec, inliers = estimate(correspondences, pixels)
            use_flow = True
            if success and inliers is not None:
                xyz = np.asarray([self.points3d[p].coords for p, _ in correspondences])
                projection = cv2.projectPoints(xyz, rvec, tvec, self.cfg.K, None)[0].reshape(-1, 2)
                errors = np.linalg.norm(projection - pixels, axis=1)
                depth = (xyz @ cv2.Rodrigues(rvec)[0].T + tvec.ravel())[:, 2]
                ids = inliers.ravel()
                inliers = ids[(errors[ids] <= self.cfg.pnp_reproj_thresh) & (depth[ids] > 0)].reshape(-1, 1)
        if not success or inliers is None or len(inliers) < minimum:
            raise RuntimeError(f"PnP failed: {len(correspondences)} correspondences, "
                               f"{0 if inliers is None else len(inliers)} inliers, required {minimum}")
        rotation, _ = cv2.Rodrigues(rvec)
        if not np.isfinite(rotation).all() or not np.isfinite(tvec).all():
            raise RuntimeError("PnP returned a non-finite pose")
        # Registration is transactional: roll back tracks and points on triangulation failure.
        point_count = len(self.points3d)
        keypoint_count = len(self.keypoints[img_idx])
        self.poses[img_idx] = (rotation, tvec.reshape(3, 1))
        updated = []
        try:
            for idx in inliers.ravel():
                point_id, key = correspondences[idx]
                if use_flow:
                    x, y = pixels[idx]
                    key = len(self.keypoints[img_idx])
                    self.keypoints[img_idx].append(cv2.KeyPoint(float(x), float(y), 1.0))
                pt = self.points3d[point_id]
                updated.append((pt, pt.observations.get(img_idx)))
                pt.observations[img_idx] = key
            self._triangulate_new_matches(img_idx)
        except Exception:
            del self.points3d[point_count:]
            del self.keypoints[img_idx][keypoint_count:]
            self.poses.pop(img_idx, None)
            for pt, previous in updated:
                if previous is None:
                    pt.observations.pop(img_idx, None)
                else:
                    pt.observations[img_idx] = previous
            raise
        self.placed.append(img_idx)
        self.registration_methods[img_idx] = "pnp_optical_flow" if use_flow else "pnp_sift"
        self.unplaced.remove(img_idx)
        logger.info("Added image %d with %d PnP inliers; %d total points",
                    img_idx, len(inliers), len(self.points3d))
        if use_flow:
            logger.info("Image %d recovered using forward/backward optical flow and validated PnP", img_idx)
        if pbar is not None:
            pbar.update(1)

    def _aligned_points(self, i: int, j: int, return_idxs: bool = False):
        matches = self.matches.get(tuple(sorted((i, j))), [])
        idx_i = np.asarray([m.queryIdx if i < j else m.trainIdx for m in matches], dtype=int)
        idx_j = np.asarray([m.trainIdx if i < j else m.queryIdx for m in matches], dtype=int)
        pts_i = np.asarray([self.keypoints[i][k].pt for k in idx_i], dtype=np.float32).reshape(-1, 2)
        pts_j = np.asarray([self.keypoints[j][k].pt for k in idx_j], dtype=np.float32).reshape(-1, 2)
        return (pts_i, pts_j, idx_i, idx_j) if return_idxs else (pts_i, pts_j)

    def _triangulate(self, i, j, idx_i, idx_j):
        if not len(idx_i):
            return []
        Ri, ti = self.poses[i]
        Rj, tj = self.poses[j]
        Pi = self.cfg.K @ np.hstack((Ri, ti))
        Pj = self.cfg.K @ np.hstack((Rj, tj))
        pixels_i = np.asarray([self.keypoints[i][k].pt for k in idx_i], dtype=np.float64)
        pixels_j = np.asarray([self.keypoints[j][k].pt for k in idx_j], dtype=np.float64)
        homogeneous = cv2.triangulatePoints(Pi, Pj, pixels_i.T, pixels_j.T).T
        valid = np.isfinite(homogeneous).all(axis=1) & (np.abs(homogeneous[:, 3]) > 1e-10)
        points = np.full((len(idx_i), 3), np.nan)
        points[valid] = homogeneous[valid, :3] / homogeneous[valid, 3:4]
        camera_i = points @ Ri.T + ti.ravel()
        camera_j = points @ Rj.T + tj.ravel()
        valid &= np.isfinite(points).all(axis=1) & (camera_i[:, 2] > 1e-6) & (camera_j[:, 2] > 1e-6)
        for camera, pixels in ((camera_i, pixels_i), (camera_j, pixels_j)):
            projected = camera @ self.cfg.K.T
            with np.errstate(divide='ignore', invalid='ignore'):
                errors = np.linalg.norm(projected[:, :2] / projected[:, 2:3] - pixels, axis=1)
            valid &= np.isfinite(errors) & (errors <= self.cfg.pnp_reproj_thresh)
        result = []
        used_i, used_j = set(), set()
        for k in np.flatnonzero(valid):
            if idx_i[k] in used_i or idx_j[k] in used_j:
                continue
            colors = []
            for cam, key in ((i, idx_i[k]), (j, idx_j[k])):
                img = self.images[cam]
                x, y = np.rint(self.keypoints[cam][key].pt).astype(int)
                x, y = np.clip(x, 0, img.shape[1]-1), np.clip(y, 0, img.shape[0]-1)
                color = img[y, x]
                colors.append(np.repeat(color, 3) if np.ndim(color) == 0 else color[:3][::-1])
            # Convert before adding: uint8 + uint8 wraps around at 255.
            rgb = tuple(np.rint(np.mean(np.asarray(colors, dtype=np.float64), axis=0)).astype(int))
            result.append(Point3DView(points[k], rgb, {i: int(idx_i[k]), j: int(idx_j[k])}))
            used_i.add(idx_i[k])
            used_j.add(idx_j[k])
        return result

    def _process_pair(self, p, img_idx, P_i=None, img_n=None):
        observed = {(cam, key) for pt in self.points3d for cam, key in pt.observations.items()}
        idx_p, idx_n = [], []
        for m in self.matches.get(tuple(sorted((p, img_idx))), []):
            q, t = (m.queryIdx, m.trainIdx) if p < img_idx else (m.trainIdx, m.queryIdx)
            if ((p, q) not in observed and (img_idx, t) not in observed
                    and 0 <= q < len(self.keypoints[p]) and 0 <= t < len(self.keypoints[img_idx])):
                idx_p.append(q)
                idx_n.append(t)
                observed.update(((p, q), (img_idx, t)))
        return self._triangulate(p, img_idx, idx_p, idx_n)

    def _triangulate_new_matches(self, img_idx):
        # Commit each pair before the next one, so observations cannot be duplicated.
        for placed in self.placed:
            if placed != img_idx:
                self.points3d.extend(self._process_pair(placed, img_idx))

    def _triangulate_and_add(self, i, j, idxs_i, idxs_j):
        points = self._triangulate(i, j, idxs_i, idxs_j)
        if len(points) < 2:
            raise ValueError("Not enough valid 3D points after triangulation")
        self.points3d.extend(points)
