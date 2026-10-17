"""
config.py: Dataclasses for holding camera and robot configuration.
"""

from dataclasses import dataclass

import numpy as np
from typing import List, Tuple

@dataclass
class CameraConfig:
    """
    Holds all camera calibration parameters.

    A note on coordinate systems: This library *assumes*
    - VEHICLE/WORLD Frame: +Z Forward, +X Right, +Y Up
    - CAMERA Frame: +Z Forward, +X Right, +Y Down

    The extrinsics (rvec, tvec) must transform points from the
    VEHICLE frame to the CAMERA frame.
    """

    camera_matrix: np.ndarray  # 3x3 Intrinsic matrix (K)
    dist_coeffs: np.ndarray  # 1xN Distortion coefficients
    rvec: np.ndarray  # 3x1 Rotation vector (extrinsic)
    tvec: np.ndarray  # 3x1 Translation vector (extrinsic)
    rotation_matrix: np.ndarray # 4x4 Rotation matrix 
    image_size: Tuple[int, int]  # (height, width)
    z_project: float # clip path in front by z_project meters


@dataclass
class RobotConfig:
    """
    Holds vehicle dimensions and properties.
    """

    width_m: float  # Vehicle width in meters
    
