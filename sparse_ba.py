"""Project-owned block Levenberg-Marquardt BA; no SfM or optimization wrapper.

OpenCV supplies projection derivatives. We assemble the camera/point normal
blocks, eliminate points using a Schur complement, apply damping, and accept
steps by their actual robust cost. NumPy/SciPy only solve linear systems.
"""
import time
import cv2
import numpy as np
from scipy import sparse
from scipy.linalg import solve


class SparseBundleAdjuster:
    def __init__(self, K, poses, xyz, camera_indices, point_indices, pixels, refine_focal=False):
        self.K0 = np.asarray(K, float).copy()
        self.cameras = np.array([np.r_[cv2.Rodrigues(R)[0].ravel(), np.asarray(t).ravel()] for R, t in poses])
        self.xyz = np.asarray(xyz, float).copy()
        self.ci = np.asarray(camera_indices, int); self.pi = np.asarray(point_indices, int)
        self.pixels = np.asarray(pixels, float); self.refine_focal = refine_focal
        self.log_scale = 0.; self.ncam = len(poses); self.npoint = len(xyz)
        first_R, first_t = poses[0]
        relative = np.array([np.asarray(t).ravel() - R @ first_R.T @ np.asarray(first_t).ravel() for R, t in poses])
        second = int(np.argmax(np.linalg.norm(relative, axis=1)))
        axis = int(np.argmax(np.abs(relative[second])))
        self.fixed = np.r_[np.arange(6), second*6+3+axis]
        self.free = np.setdiff1d(np.arange(self.ncam*6+int(refine_focal)), self.fixed)
        self.groups = [np.flatnonzero(self.ci == i) for i in range(self.ncam)]
        self.gauge = {'fixed_pose_index': 0, 'scale_camera_index': second, 'fixed_translation_axis': axis}

    def evaluate(self, cameras=None, xyz=None, log_scale=None, jacobian=False):
        cameras = self.cameras if cameras is None else cameras
        xyz = self.xyz if xyz is None else xyz
        log_scale = self.log_scale if log_scale is None else log_scale
        K = self.K0.copy(); K[0, 0] *= np.exp(log_scale); K[1, 1] *= np.exp(log_scale)
        prediction = np.empty_like(self.pixels)
        cam_derivatives = np.zeros((len(self.ci), 2, 6+int(self.refine_focal)))
        point_derivatives = np.zeros((len(self.ci), 2, 3))
        for c, idx in enumerate(self.groups):
            if not len(idx): continue
            pose = cameras[c]; X = xyz[self.pi[idx]]
            uv, J = cv2.projectPoints(X, pose[:3], pose[3:], K, None)
            prediction[idx] = uv.reshape(-1, 2)
            if jacobian:
                J = J.reshape(-1, 2, J.shape[1]); cam_derivatives[idx, :, :6] = J[:, :, :6]
                point_derivatives[idx] = J[:, :, 3:6] @ cv2.Rodrigues(pose[:3])[0]
                if self.refine_focal:
                    cam_derivatives[idx, :, 6] = J[:, :, 6]*K[0, 0] + J[:, :, 7]*K[1, 1]
        residual = prediction - self.pixels
        if not jacobian: return residual
        rows = np.arange(2*len(self.ci)).reshape(-1, 2, 1)
        cc = np.broadcast_to(self.ci[:, None, None]*6+np.arange(6), (len(self.ci), 2, 6))
        if self.refine_focal:
            cc = np.concatenate([cc, np.full((len(self.ci), 2, 1), self.ncam*6)], axis=2)
        cr = np.broadcast_to(rows, cc.shape)
        Jc = sparse.coo_matrix((cam_derivatives.ravel(), (cr.ravel(), cc.ravel())), shape=(2*len(self.ci), self.ncam*6+int(self.refine_focal))).tocsr()[:, self.free]
        pc = np.broadcast_to(self.pi[:, None, None]*3+np.arange(3), point_derivatives.shape)
        pr = np.broadcast_to(rows, pc.shape)
        Jp = sparse.coo_matrix((point_derivatives.ravel(), (pr.ravel(), pc.ravel())), shape=(2*len(self.ci), self.npoint*3)).tocsr()
        return residual, Jc, Jp, point_derivatives

    @staticmethod
    def cost(residual):
        return float(np.sum(np.sqrt(1. + np.sum(residual**2, axis=1)) - 1.))

    def optimize(self, max_iterations=100, focal_bounds=(.2, 5.)):
        start = time.perf_counter(); damping = 1e-3; steps = []; evaluations = 1
        r = self.evaluate(); initial_cost = cost = self.cost(r); initial_error = float(np.linalg.norm(r, axis=1).mean())
        converged = False; reason = 'iteration_budget'; rejected = 0
        for iteration in range(max_iterations):
            r, Jc, Jp, dp = self.evaluate(jacobian=True)
            weight = 1. / np.sqrt(1. + np.sum(r*r, axis=1)); sqrtw = np.repeat(np.sqrt(weight), 2)
            A = Jc.multiply(sqrtw[:, None]).tocsr(); B = Jp.multiply(sqrtw[:, None]).tocsr()
            U = (A.T @ A).toarray(); W = (A.T @ B).tocsr()
            V = np.zeros((self.npoint, 3, 3))
            np.add.at(V, self.pi, np.einsum('nij,nik,n->njk', dp, dp, weight))
            wr = r.ravel()*sqrtw
            gc = np.asarray(A.T @ wr).ravel(); gp = np.asarray(B.T @ wr).ravel()
            diagc = np.maximum(np.diag(U), 1e-8); diagp = np.maximum(np.diagonal(V, axis1=1, axis2=2), 1e-8)
            scaled_gradient = max(np.max(np.abs(gc)/np.sqrt(diagc)), np.max(np.abs(gp)/np.sqrt(diagp.ravel())))
            if scaled_gradient < 1e-6:
                converged = True; reason = 'scaled_gradient'; break
            if evaluations >= max_iterations:
                reason = 'evaluation_budget'; break
            accepted = False
            for trial in range(10):
                if evaluations >= max_iterations: break
                Vd = V.copy(); Vd[:, np.arange(3), np.arange(3)] += damping*diagp
                inverse = np.linalg.inv(Vd)
                Vinv = sparse.bsr_matrix((inverse, np.arange(self.npoint), np.arange(self.npoint+1)), shape=(3*self.npoint, 3*self.npoint)).tocsr()
                WVi = W @ Vinv
                S = U - (WVi @ W.T).toarray(); S.flat[::len(S)+1] += damping*diagc
                try:
                    dc = solve(S, -gc + WVi @ gp, assume_a='pos', check_finite=False)
                except np.linalg.LinAlgError:
                    damping *= 10; rejected += 1; continue
                dx = -(Vinv @ (gp + W.T @ dc)).reshape(-1, 3)
                full = np.zeros(self.ncam*6+int(self.refine_focal)); full[self.free] = dc
                candidate = self.cameras + full[:self.ncam*6].reshape(-1, 6)
                candidate_f = self.log_scale + (full[-1] if self.refine_focal else 0.)
                if not np.log(focal_bounds[0]) <= candidate_f <= np.log(focal_bounds[1]):
                    damping *= 10; rejected += 1; continue
                nr = self.evaluate(candidate, self.xyz+dx, candidate_f); evaluations += 1
                nc = self.cost(nr)
                if np.isfinite(nc) and nc < cost:
                    improvement = (cost-nc)/max(cost, 1.)
                    self.cameras = candidate; self.xyz += dx; self.log_scale = candidate_f
                    steps.append({'iteration':iteration+1,'cost_before':cost,'cost_after':nc,'damping':damping,'scaled_gradient':float(scaled_gradient)})
                    cost = nc; damping = max(damping/3., 1e-10); accepted = True
                    if improvement < 1e-8:
                        converged = True; reason = 'relative_cost'
                    break
                damping *= 10.; rejected += 1
            if converged: break
            if not accepted:
                reason = 'evaluation_budget' if evaluations >= max_iterations else 'no_acceptable_step'; break
        K = self.K0.copy(); K[0,0] *= np.exp(self.log_scale); K[1,1] *= np.exp(self.log_scale)
        poses = [(cv2.Rodrigues(c[:3])[0], c[3:].copy()) for c in self.cameras]
        stats = {'engine':'project-owned Schur LM; OpenCV derivatives + NumPy/SciPy linear algebra',
                 'initial_cost':initial_cost,'cost':cost,'successful_steps':len(steps),'rejected_steps':rejected,'nfev':evaluations,
                 'converged':converged,'termination':reason,'gauge':self.gauge,'iterations':steps,
                 'mean_error_before_px':initial_error,'mean_error_after_px':float(np.linalg.norm(self.evaluate(),axis=1).mean()),
                 'elapsed_seconds':time.perf_counter()-start}
        return poses, self.xyz.copy(), K, stats
