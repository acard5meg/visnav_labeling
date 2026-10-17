"""
visualizer.py: The main, user-facing TrajectoryVisualizer class.
"""

import cv2
import numpy as np

from .config import CameraConfig, RobotConfig
from .projection import create_3d_trajectory_polygon, project_points_3d_to_2d
from typing import List, Tuple

class TrajectoryVisualizer:
    """
    A class to encapsulate trajectory projection and drawing.
    """

    def __init__(self, camera_config: CameraConfig, robot_config: RobotConfig):
        """
        Initializes the visualizer with camera and robot configurations.

        Args:
            camera_config: A CameraConfig dataclass instance.
            robot_config: A RobotConfig dataclass instance.
        """
        self.cam_config = camera_config
        self.robot_config = robot_config
        self._undistort_map_x, self._undistort_map_y = cv2.initUndistortRectifyMap(
            camera_config.camera_matrix,
            camera_config.dist_coeffs,
            None,
            camera_config.camera_matrix,
            (camera_config.image_size[1], camera_config.image_size[0]),
            cv2.CV_16SC2,
        )

    def mask_points(self, arr):
        width_0 = arr[:, 0] >= 0
        width_max = arr[:, 0] <= self.cam_config.image_size[1]
        height_0 = arr[:, 1] >= 0
        height_max = arr[:, 1] <= self.cam_config.image_size[0]
        width_mask = np.logical_and(width_0, width_max)
        height_mask = np.logical_and(height_0, height_max)

        combine_mask = np.logical_and(width_mask, height_mask).reshape(-1,1)

        final_mask = np.hstack((combine_mask, combine_mask))

        return arr[final_mask].reshape(-1, 2)



    def draw_trajectory_on_image(
        self,
        image: np.ndarray,
        centerline: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        alpha: float = 0.4,
        line_thickness: int = 2,
    ) -> np.ndarray:
        """
        Draws the projected 3D trajectory path and coverage area on an image.

        Args:
            image: The (H, W, 3) BGR image (from cv2.imread or camera).
            centerline: (N, 3) [X, Y, Z] or (N, 2) [X, Z] numpy array of
                        centerline points. If (N, 2) is provided,
                        robot_config.height_m will be used for the Y-coordinate.
            color: (B, G, R) tuple for the visualization color.
            alpha: Transparency of the filled polygon (0.0 to 1.0).
            line_thickness: Thickness of the centerline in pixels.

        Returns:
            A new (H, W, 3) BGR image with the trajectory drawn on it.
        """
        # --- Validation ---
        
        if image.shape[:2] != self.cam_config.image_size:
            raise ValueError(
                f"Image size {image.shape[:2]} does not match "
                f"config size {self.cam_config.image_size}"
            )

        if not (centerline.ndim == 2 and centerline.shape[1] in (2, 3)):
            raise ValueError(
                f"centerline must be (N, 2) [X_lat, Z_fwd] or (N, 3) [X_lat, Y_hgt, Z_fwd], "
                f"but got shape {centerline.shape}"
            )
        
        centerline_4 = np.ones((centerline.shape[0], 4))
        centerline_4[:, :3] = centerline
        centerline_4 = centerline_4[:,:,np.newaxis]
        batch_rotation_matrix = np.zeros((centerline.shape[0], 4, 4))
        batch_rotation_matrix[:, ] = self.cam_config.rotation_matrix
        points_in_camera_frame =  batch_rotation_matrix @  centerline_4 
        
        # Clip path to be in front of the camera (e.g., Z > 0.1m)
        # This prevents projection errors and "wrap-around" polygons
        # Assumes +Z is forward

        valid_indices = points_in_camera_frame[:, 2] > self.cam_config.z_project
        
        if not np.any(valid_indices):
            return image  # No valid points to draw

        valid_centerline_4 = points_in_camera_frame[valid_indices.squeeze(1)]
        valid_centerline_3d = valid_centerline_4[:, :3, 0]

        polygon_3d = create_3d_trajectory_polygon(
            valid_centerline_3d, self.robot_config.width_m
        )

        # --- 2D Projection ---
        polygon_2d_project = project_points_3d_to_2d(polygon_3d, self.cam_config)
        centerline_2d_project = project_points_3d_to_2d(valid_centerline_3d, self.cam_config)

        polygon_2d = polygon_2d_project
        centerline_2d = centerline_2d_project

        if polygon_2d.size == 0 or centerline_2d.size == 0:
            return image  # All points were outside the view

        image = cv2.remap(
            image,
            self._undistort_map_x,
            self._undistort_map_y,
            interpolation=cv2.INTER_LINEAR,
        )

        # Create a separate overlay for the transparent polygon
        overlay = image.copy()

        # Draw the filled polygon on the overlay
        cv2.fillPoly(overlay, [polygon_2d], color=color)

        # Blend the overlay with the output image
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        # Draw the centerline on top of the blended image
        cv2.polylines(
            image,
            [centerline_2d],
            isClosed=False,
            color=color,
            thickness=line_thickness,
        )

        return image
