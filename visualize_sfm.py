"""Display legacy reconstruction with basic Matplotlib plotting."""
import numpy as np
import matplotlib.pyplot as plt


def visualize_reconstruction(poses, points3d, cam_size=0.1, max_dist=50.0):
    xyz = np.asarray(points3d).reshape(-1, 3)
    centers = np.array([-R.T @ np.asarray(t).ravel() for R, t in poses.values()])
    fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
    if len(xyz): ax.scatter(*xyz.T, c=xyz[:, 2], s=1, cmap='viridis')
    if len(centers): ax.scatter(*centers.T, c='orange', s=20)
    for R, t in poses.values():
        center = -R.T @ np.asarray(t).ravel()
        corners = np.array([[0,0,0],[-.6,-.4,1],[.6,-.4,1],[.6,.4,1],[-.6,.4,1]]) * cam_size
        corners = corners @ R + center
        for a,b in [(0,1),(0,2),(0,3),(0,4),(1,2),(2,3),(3,4),(4,1)]:
            ax.plot(*corners[[a,b]].T, c='orange', linewidth=.7)
    if len(xyz): ax.set_box_aspect(np.maximum(np.ptp(xyz,axis=0),1e-9))
    plt.show()
