"""
Streamlit app for visualizing vehicle trajectories using the trajectory_viz library.

This app allows:
- Browsing trajectory CSVs from a root directory.
- Displaying trajectory data in a table.
- Clicking a frame to see the camera image with the projected future trajectory.
- Animating the trajectory visualization from a selected start point.
"""

import ast
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import os
import pickle
import yaml

from utils.odom_6_dof import to_robot_np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    from trajectory_viz import CameraConfig, RobotConfig, TrajectoryVisualizer
except ImportError:
    st.error(
        "Could not import trajectory_viz. "
        "Please install the package first by running "
        "'pip install -e .' from the project's root directory."
    )
    st.stop()


# --- Configuration and Caching ---
FOLDER_NAME = "traj_viz"
CONFIG_PATH = os.path.join(PROJECT_ROOT, FOLDER_NAME, "config.yaml")

@st.cache_resource
def get_visualizer_for_robot(
    filepath: str,
    curr_dataset: str
) -> TrajectoryVisualizer:
    """
    Parses the config for a specific robot and returns a cached Visualizer instance.
    Using _all_configs (with an underscore) is a streamlit caching convention.
    """

    with open(os.path.join(filepath, 'traj_data.pkl'), 'rb') as f:
        all_data = pickle.load(f)

    with open(CONFIG_PATH, 'rb') as f:
        changes = yaml.safe_load(f)

    pkl_camera_matrix, pkl_distortion_coeff = all_data['camera matrix'], all_data['distortion coeff']
    pkl_rotation_matrix, pkl_image_res = all_data['rotation matrix'], all_data['image resolution']
    width, cam_project = all_data['width'], all_data['camera distance projection']

    # width, height
    img_size = tuple(pkl_image_res[::-1])

    camera_matrix = np.array(pkl_camera_matrix, dtype=np.float32)

    dist_coeffs = np.array(pkl_distortion_coeff, dtype=np.float32)

    df_rotation_matrix = np.array(pkl_rotation_matrix, dtype=np.float32)

    # use config file to change values for visualization
    if curr_dataset in changes:
        keys = changes[curr_dataset]

        for k in keys:
            row, col = k.split('-')
            val = keys[k] 
            df_rotation_matrix[int(row), int(col)] = val
    
    rvec = cv2.Rodrigues(df_rotation_matrix[:3,:3])[0].flatten()
    
    tvec = df_rotation_matrix[:3,3].flatten()

    # Create config dataclasses
    cam_config = CameraConfig(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        # We use 0's b/c we do rotations in visualizer script
        rvec=np.zeros((3,1), dtype=np.float32), 
        tvec=np.zeros((3,1), dtype=np.float32),
        rotation_matrix=df_rotation_matrix,
        image_size=img_size,
        z_project=cam_project
    )

    robot_config = RobotConfig(
        width_m=width
    )

    return TrajectoryVisualizer(cam_config, robot_config)

def get_relative_trajectory(
    filepath: str, start_index: int, num_points: int
) -> np.ndarray:
    """
    Calculates the future trajectory relative to the pose at start_index.
    Returns an (N, 2) array of [X_fwd, Y_lat_left].
    """
    end_index = min(start_index + num_points, len(df))
    # Need at least 2 points (current and one future) to get a relative pose
    if (end_index - start_index) < 2:
        return np.array([]).reshape(0, 2)

    with open(os.path.join(filepath, 'traj_data.pkl'), 'rb') as f:
        all_data = pickle.load(f)

    position, rotation = all_data['position'], all_data['rotations']

    pos_list = position[start_index:end_index]
    
    rpy_list = rotation[start_index:end_index]

    # Get current pose and its inverse
    if len(pos_list[0]) != 3:
        pos_current = np.array([pos_list[0][0], pos_list[0][1], 0.0])
    else:
        pos_current = np.array(pos_list[0])
    
    rpy_current = np.array(rpy_list[0])  # [0]  # 'yaw' is stored as a list [val]
    
    future_pos = np.array(pos_list[1:])
    future_rpy = np.array(rpy_list[1:])

    future_poses = np.hstack((future_pos, future_rpy))

    cur_poses = np.concatenate((pos_current, rpy_current))
    repeat_curr_pose = np.vstack((cur_poses,)* len(future_poses))
    future_poses_relative = np.array(to_robot_np(repeat_curr_pose, future_poses))

    odometry_in_frame_1 = future_poses_relative[:, :3]
    return odometry_in_frame_1

def make_dataframe(pkl):
    traj_data_i = {}

    traj_data_i['position'] = [f'[{x}, {y}, {z}]' for x,y,z in pkl['position']]
    
    traj_data_i['rotations'] = [f'[{r}, {p}, {y}]' for r,p,y in pkl['rotations']]

    refine_data = list(
                zip(pkl['timestep'], 
                traj_data_i['position'], 
                traj_data_i['rotations'])
                )
    pd_frame = pd.DataFrame(refine_data, columns=['frame', 'position', 'rotations'])

    return pd_frame

# --- Streamlit App UI ---

st.set_page_config(layout="wide")
st.title("Trajectory Visualization Tool")

# --- Sidebar ---
st.sidebar.markdown("# Control Panel")

# --- Trajectory Selection ---
root_dir = st.sidebar.text_input("Data Root Directory:", '/mnt/datasets/GNM/processed_data_6dof/processed_data_10hz')

root_path = Path(root_dir)

if not root_path.exists():
    st.sidebar.error("Root directory not found. Please check the path.")
    st.stop()

cleaned_datasets = sorted(os.listdir(str(root_path)))

cleaned_datasets.remove(FOLDER_NAME)

selected_dataset = st.sidebar.selectbox("Select Dataset:", cleaned_datasets)

dataset_traj = os.listdir(os.path.join(str(root_path), selected_dataset))

all_traj_and_length = [(len(os.listdir(os.path.join(str(root_path), selected_dataset, i)))-1, i) for i in dataset_traj if len(os.listdir(os.path.join(str(root_path), selected_dataset, i)))-1 >= 16]

choose_traj = sorted(all_traj_and_length, key = lambda x: ('_'.join(x[1].split('_')[:-1]), -x[0]))

with st.sidebar:
    with st.container(height=300):
        _, selected_traj_name = st.radio("Select Trajectory:\nLength : Name", choose_traj)
    traj_len = st.slider("Trajectory Length (steps):", 1, 50, 10)
    frame_rate = st.slider("Animation FPS:", 1, 30, 5)

    # Use 'key' to make the toggle's state persistent in st.session_state
    st.toggle("Animate", key="animate")


# --- Main Panel (Table and Visualization) ---
col1, col2 = st.columns([0.4, 0.6])

# --- Get the visualizer for the selected robot ---
selected_traj_path = os.path.join(str(root_path), selected_dataset, selected_traj_name)

with open(os.path.join(selected_traj_path, 'traj_data.pkl'), 'rb') as f:
    all_data = pickle.load(f)

df = make_dataframe(all_data)

st.sidebar.text_input("Current Robot", all_data['robot'])

try:
    visualizer = get_visualizer_for_robot(selected_traj_path, selected_dataset)
except Exception as e:
    st.error(f"Error initializing visualizer for '{selected_traj_path}': {e}")
    st.exception(e)
    st.stop()

with col1:
    st.subheader("Trajectory Data")
    # Use st.dataframe for selection
    selection_event = st.dataframe(
        df,
        on_select="rerun",
        selection_mode="single-row",
        use_container_width=True,
        hide_index=True,
        key="df_select",
    )

    # Get the selected row index
    selected_rows = selection_event.selection.rows
    frame_start = selected_rows[0] if selected_rows else 0

with col2:
    # Create a placeholder for the image/animation
    placeholder = st.empty()

    if st.session_state.animate:
        # --- Animation Logic ---
        st.sidebar.info(f"Animating from frame {frame_start}...")


        for i in range(frame_start, len(df)):

            img_path = os.path.join(selected_traj_path, f'{i}.jpg')

            frame_image = cv2.imread(str(img_path))

            if frame_image is None:
                st.error(f"Could not load image: {img_path}")
                time.sleep(0.1)
                continue

            # --- Draw Trajectory for *this* frame ---
            # 1. Get relative trajectory
            
            # this doesn't assume any specific coordinate frame
            # conversion from robot's specific frame to optical frame
            # contained in the camera config class
            rel_poses_xy = get_relative_trajectory(selected_traj_path, i, traj_len)
            
            if rel_poses_xy.size == 0:
                placeholder.image(frame_image, channels="BGR", use_container_width=True)
                time.sleep(1.0 / frame_rate)
                continue

            # 2. Draw on image
            try:
                vis_image = visualizer.draw_trajectory_on_image(
                    frame_image, rel_poses_xy
                )
                # 3. Display
                with placeholder.container():
                    col1, col2 = st.columns([0.9,0.1])

                    with col1:
                        col1.image(vis_image, channels="BGR", use_container_width=True)
                    with col2:
                        col2.write(f"Frame: {i}")
            except ValueError as e:
                # Catch image size mismatch errors
                st.error(f"Error drawing on frame {i}: {e}")
                placeholder.image(frame_image, channels="BGR", use_container_width=True)

            # 4. Sleep
            time.sleep(1.0 / frame_rate)

            # Check if user toggled 'Animate' off during loop
            if not st.session_state.animate:
                st.sidebar.info("Animation stopped.")

        # Reset toggle after animation finishes
        if st.session_state.animate:
            st.session_state.animate = False
            st.rerun()

    else:
        # --- Single Frame Logic ---
        img_path = os.path.join(selected_traj_path, f'{frame_start}.jpg')
        image = cv2.imread(str(img_path))

        if image is None:
            st.error(f"Could not load image: {img_path}")
        else:
            # 1. Get relative trajectory
            # this doesn't assume any specific coordinate frame
            # conversion from robot's specific frame to optical frame
            # contained in the camera config class
            rel_poses_xy = get_relative_trajectory(selected_traj_path, frame_start, traj_len)
            if rel_poses_xy.size > 0:

                # 2. Draw on image
                try:
                    vis_image = visualizer.draw_trajectory_on_image(image, rel_poses_xy)
                    placeholder.image(
                        vis_image, channels="BGR", use_container_width=True
                    )
                except ValueError as e:
                    # Catch image size mismatch errors (e.g., from config)
                    st.error(f"Error drawing on frame: {e}")
                    st.exception(e)
                    placeholder.image(image, channels="BGR", use_container_width=True)
            else:
                # No trajectory to draw, just show the image
                placeholder.image(image, channels="BGR", use_container_width=True)
