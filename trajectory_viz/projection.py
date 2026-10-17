"""
projection.py: Core logic for 3D->2D projection and polygon creation.
"""

import cv2
import numpy as np

from .config import CameraConfig


def create_3d_trajectory_polygon(
    centerline_3d: np.ndarray, width_m: float
) -> np.ndarray:
    """
    Creates a 3D polygon mesh from a 3D centerline.

    Assumes the VEHICLE coordinate system:
    - Centerline (N, 3) array is [X, Y, Z]
    - +X is lateral (right)
    - +Y is vertical (up)
    - +Z is longitudinal (forward)

    Args:
        centerline_3d: (N, 3) numpy array of 3D points.
        width_m: Width of the vehicle/polygon in meters.

    Returns:
        (2*N, 3) numpy array of 3D points defining the polygon.
    """
    if centerline_3d.ndim != 2 or centerline_3d.shape[1] != 3:
        raise ValueError(f"centerline_3d must be (N, 3), but got {centerline_3d.shape}")

    # Create left and right boundaries
    # Assumes X is the lateral coordinate
    points_left_3d = centerline_3d.copy()
    points_left_3d[:, 0] = centerline_3d[:, 0] - width_m / 2

    points_right_3d = centerline_3d.copy()
    points_right_3d[:, 0] = centerline_3d[:, 0] + width_m / 2

    # Combine to form a closed polygon: (left_points) + (reversed right_points)
    # The flip ensures the points are in order to draw the polygon correctly
    polygon_3d = np.concatenate(
        (points_left_3d, np.flip(points_right_3d, axis=0)), axis=0
    ).astype(np.float32)

    return polygon_3d


def project_points_3d_to_2d(points_3d: np.ndarray, config: CameraConfig) -> np.ndarray:
    """
    Projects 3D points from VEHICLE/WORLD space to 2D image space.

    Args:
        points_3d: (N, 3) numpy array of 3D points.
        config: The CameraConfig object.

    Returns:
        (N, 2) numpy array of 2D integer pixel coordinates.
    """
    if points_3d.size == 0:
        return np.array([], dtype=np.int32).reshape(0, 2)

    points_2d, _ = cv2.projectPoints(
        points_3d.astype(np.float32),
        config.rvec,
        config.tvec,
        config.camera_matrix,
        config.dist_coeffs,
    )

    # Reshape to (N, 2) and convert to integer pixels
    return points_2d.reshape(-1, 2).astype(np.int32)
