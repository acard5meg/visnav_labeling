"""
Trajectory Visualization Toolkit
__init__.py: Makes 'trajectory_viz' a package and exports key classes.
"""

from .config import CameraConfig, RobotConfig
from .visualizer import TrajectoryVisualizer

__all__ = ["CameraConfig", "RobotConfig", "TrajectoryVisualizer"]
