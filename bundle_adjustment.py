from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@dataclass
class BAConfig:
    max_nfev: int = 1000
    ftol: float = 1e-6
    xtol: float = 1e-6
    gtol: float = 1e-6
    loss: str = 'huber'
    f_scale: float = 2.0
    method: str = 'trf'
    verbose: int = 2      # 0=silent, 2=lots of output

class BundleAdjuster:
    def __init__(
        self,
        K: np.ndarray,
        camera_indices: np.ndarray,
        point_indices: np.ndarray,
        points_2d: np.ndarray,
        n_cameras: int,
        n_points: int,
        rvecs: Dict[int, np.ndarray],
        tvecs: Dict[int, np.ndarray],
        points3d: List[np.ndarray],
        config: BAConfig = BAConfig()
    ):
        self.K = K
        self.cam_idx = camera_indices
        self.pt_idx = point_indices
        self.pts2d = points_2d
        self.n_cams = n_cameras
        self.n_pts = n_points
        self.config = config
        self.result = None
        self._observation_groups = [np.flatnonzero(self.cam_idx == i) for i in range(n_cameras)]

        # initialize parameter vector: [rvecs,tvecs,pts3d]
        cam_params = []
        for i in range(n_cameras):
            r = rvecs[i].ravel()
            t = tvecs[i].ravel()
            cam_params.append(np.hstack((r, t)))  # 6 params per cam
        self.x0 = np.hstack((
            np.array(cam_params).ravel(),
            np.array(points3d).reshape(-1)
        ))

    def _sparsity(self) -> lil_matrix:
        m = self.cam_idx.size * 2
        n = self.n_cams * 6 + self.n_pts * 3
        A = lil_matrix((m, n), dtype=int)
        i = np.arange(self.cam_idx.size)
        for s in range(6):
            A[2*i,     self.cam_idx*6 + s] = 1
            A[2*i+1,   self.cam_idx*6 + s] = 1
        base = self.n_cams * 6
        for s in range(3):
            A[2*i,     base + self.pt_idx*3 + s] = 1
            A[2*i+1,   base + self.pt_idx*3 + s] = 1
        return A

    def _project(self, params: np.ndarray) -> np.ndarray:
        cam_params = params[:self.n_cams*6].reshape(self.n_cams, 6)
        pts3d = params[self.n_cams*6:].reshape(self.n_pts, 3)
        proj = np.zeros((self.cam_idx.size, 2))
        # One OpenCV call per camera, rather than one call per observation.
        for camera_id, indices in enumerate(self._observation_groups):
            if not len(indices):
                continue
            cam = cam_params[camera_id]
            rvec = cam[:3]
            tvec = cam[3:].reshape(3,1)
            X = pts3d[self.pt_idx[indices]]
            p, _ = cv2.projectPoints(X, rvec, tvec, self.K, distCoeffs=None)
            proj[indices] = p.reshape(-1, 2)
        return proj

    def residuals(self, params: np.ndarray) -> np.ndarray:
        proj = self._project(params)
        return (proj - self.pts2d).ravel()

    def optimize(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        A = self._sparsity()
        res = least_squares(
            fun=self.residuals,
            x0=self.x0,
            jac_sparsity=A,
            verbose=self.config.verbose,
            ftol=self.config.ftol,
            xtol=self.config.xtol,
            gtol=self.config.gtol,
            loss=self.config.loss,
            f_scale=self.config.f_scale,
            max_nfev=self.config.max_nfev,
            method=self.config.method
        )
        self.result = res
        p = res.x[:self.n_cams*6].reshape(self.n_cams, 6)
        pts = res.x[self.n_cams*6:].reshape(self.n_pts, 3)
        rvecs_opt = {i: p[i,:3] for i in range(self.n_cams)}
        tvecs_opt = {i: p[i,3:].reshape(3,1) for i in range(self.n_cams)}
        return rvecs_opt, tvecs_opt, pts

    def compute_average_reprojection_error(self, params: np.ndarray = None) -> float:
        """
        Computes the average reprojection error in pixels.

        If no params are provided, uses the optimized parameters.
        """
        if params is None:
            params = self.result.x if self.result is not None else self.x0

        proj = self._project(params)
        errors = np.linalg.norm(proj - self.pts2d, axis=1)
        return np.mean(errors)
