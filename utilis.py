import logging
import os
import numpy as np
import cv2
from pathlib import Path
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import time

from visualize_sfm import visualize_reconstruction
from reconstruction import *

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def pack_params(rvecs_opt, tvecs_opt, pts3d_opt):
    n_cams = len(rvecs_opt)
    cam_params = []
    for i in range(n_cams):
        r = rvecs_opt[i].ravel()
        t = tvecs_opt[i].ravel()
        cam_params.append(np.hstack((r, t)))  # 6 params per cam
    params = np.hstack((
        np.array(cam_params).ravel(),
        np.array(pts3d_opt).reshape(-1)
    ))
    return params
def load_calibration_matrix(path):
    """Load camera calibration matrix from a text file."""
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
        mat = [list(map(float, line.strip().split())) for line in lines]
        K = np.array(mat)
        if K.shape != (3, 3):
            raise ValueError(f"Expected 3x3 matrix but got shape {K.shape}")
        return K
    except Exception as e:
        logger.error(f"Failed to load calibration matrix from {path}: {e}")
        raise

def display_saved_reconstruction(ply_path):
    from ply_io import read_ply
    xyz, rgb = read_ply(ply_path)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*xyz.T, c=rgb, s=1)
    ax.set_box_aspect(np.maximum(np.ptp(xyz, axis=0), 1e-9))
    plt.show()


def visualize_current_state(recon, title="Current State", pause=False):
    pts3d_final = [p.coords for p in recon.points3d]
    logger.info(f"{title} | Points: {len(pts3d_final)}, Cameras: {len(recon.poses)}")
    visualize_reconstruction(recon.poses, pts3d_final, cam_size=0.5)
    if pause:
        input("Press Enter to continue...")

def pack_params(rvecs_opt, tvecs_opt, pts3d_opt):
    n_cams = len(rvecs_opt)
    cam_params = []
    for i in range(n_cams):
        r = rvecs_opt[i].ravel()
        t = tvecs_opt[i].ravel()
        cam_params.append(np.hstack((r, t)))  # 6 params per cam
    params = np.hstack((
        np.array(cam_params).ravel(),
        np.array(pts3d_opt).reshape(-1)
    ))
    return params


def load_calibration_matrix(path):
    """Load camera calibration matrix from a text file."""
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
        mat = [list(map(float, line.strip().split())) for line in lines]
        K = np.array(mat)
        if K.shape != (3, 3):
            raise ValueError(f"Expected 3x3 matrix but got shape {K.shape}")
        return K
    except Exception as e:
        logger.error(f"Failed to load calibration matrix from {path}: {e}")
        raise


def save_reconstruction_ply(
    points3d: List['Point3DView'],  
    output_path: str,
    camera_poses: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
    max_dist: float = 50.0  # Filter points beyond this distance from any camera
) -> None:
    """
    Save 3D reconstruction to PLY format with filtered points and RGB colors.
    
    Args:
        points3d: List of 3D points with RGB values
        output_path: Path to save the .ply file
        camera_poses: Optional dictionary of camera poses {img_idx: (R, t)}
        max_dist: Maximum distance from camera centers to keep points (meters)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Filter points within max_dist from any camera center
    if camera_poses and max_dist > 0:
        cam_centers = [(-R.T @ t).flatten() for R, t in camera_poses.values()]
        cam_centers = np.array(cam_centers)
        filtered_points = []
        
        for p in points3d:
            pt = np.array(p.coords)
            if cam_centers.shape[0] > 0:
                distances = np.linalg.norm(cam_centers - pt, axis=1)
                if np.min(distances) < max_dist:
                    filtered_points.append(p)
            else:
                filtered_points.append(p)
        points3d = filtered_points
    
    # Write PLY file
    try:
        with open(output_path, 'w') as f:
            # Header
            total_vertices = len(points3d)
            if camera_poses:
                total_vertices += len(camera_poses)
                
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {total_vertices}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            
            # Write 3D points with RGB
            for p in points3d:
                x, y, z = p.coords
                r, g, b = getattr(p, 'rgb', (255, 255, 255))  # Default to white
                # Ensure RGB values are integers
                r = int(min(max(r, 0), 255))
                g = int(min(max(g, 0), 255))
                b = int(min(max(b, 0), 255))
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
            
            # Write camera centers as red points
            if camera_poses:
                for R, t in camera_poses.values():
                    cam_center = (-R.T @ t).flatten()
                    x, y, z = cam_center
                    f.write(f"{x:.6f} {y:.6f} {z:.6f} 255 0 0\n")
                    
        logger.info(f"📁 Saved filtered reconstruction to: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save PLY file: {str(e)}")
        raise



def display_saved_reconstruction(ply_path):
    pcd = o3d.io.read_point_cloud(str(ply_path))
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Reconstruction', width=1280, height=720)
    
    render_option = vis.get_render_option()
    render_option.background_color = np.array([0.1, 0.1, 0.1])  # Match live view
    render_option.point_size = 3.0  # Match live view
    
    vis.add_geometry(pcd)
    
    # Set viewpoint to match live view
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(0.8)
    
    vis.run()
    vis.destroy_window()


def rename_images_sequentially(image_folder, valid_extensions=None):
    """
    Renames images in the folder sequentially (e.g., 00.jpg, 01.png)
    and ensures all file extensions are lowercase.

    Returns:
        str: The file extension of the first renamed image (lowercase).
    """
    folder = Path(image_folder)
    if valid_extensions is None:
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    # Step 1: Convert all valid image files to lowercase extensions
    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in valid_extensions:
            if file.suffix != file.suffix.lower():
                new_path = file.with_suffix(file.suffix.lower())
                if not new_path.exists():
                    os.rename(file, new_path)
                    print(f"Renamed extension to lowercase: {file.name} -> {new_path.name}")
    
    # Step 2: Collect valid image files (now with lowercase extensions)
    images = []
    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in valid_extensions:
            images.append(file)

    # Sort by name
    images.sort()

    # Step 3: Rename sequentially, preserving lowercase extensions
    for idx, old_path in enumerate(images):
        new_name = f"{idx:02d}{old_path.suffix}"  # Keep now-lowercase extension
        new_path = folder / new_name

        if new_path.exists():
            continue

        os.rename(old_path, new_path)
        print(f"Renamed: {old_path.name} -> {new_name}")

    print("✅ Done renaming images.")

    # Step 4: Return the lowercase extension of the first image
    if images:
        return images[0].suffix.lower()
    else:
        return ""
    


    
def get_image_files(folder_path, ext):
    folder = Path(folder_path)
    ext = f".{ext}" if not ext.startswith(".") else ext
    return sorted([f.name for f in folder.iterdir() if f.suffix == ext])
