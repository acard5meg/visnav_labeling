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
from typing import Tuple

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
    pkl_image_res = [i // 2 for i in pkl_image_res]

    # width, height
    img_size = tuple(pkl_image_res[::-1])

    camera_matrix = np.array(pkl_camera_matrix, dtype=np.float32) / 2
    camera_matrix[2,2] = 1.

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

@st.cache_resource(ttl=30*60)
def read_trajectory_data(selected_traj_path):
    with open(os.path.join(selected_traj_path, 'traj_data.pkl'), 'rb') as f:
        all_data = pickle.load(f)
    
    return all_data

def clean_pickle_file(all_data) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Reads the pickle file and returns the position and rotation vectors
    these are cast to float32 to speed up processing/reduce memory footprint
    '''

    position = np.asarray(all_data['position'], dtype=np.float32)
    rotation = np.asarray(all_data['rotations'], dtype=np.float32)

    return position, rotation

@st.cache_resource(ttl=30 * 60, max_entries=128)
def load_frame_image(traj_path: str, frame_index: int):
    """Load and resize one frame, keeping only recently viewed frames cached."""
    img_path = os.path.join(traj_path, f'{frame_index}.jpg')
    frame_image = cv2.imread(img_path)
    if frame_image is None:
        return None
    return cv2.resize(
        frame_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR
    )


def get_relative_trajectory(
    position : np.ndarray, rotation : np.ndarray, start_index: int, num_points: int
) -> np.ndarray:
    """
    Calculates the future trajectory relative to the pose at start_index.
    Returns an (N, 2) array of [X_fwd, Y_lat_left].
    """
    
    end_index = min(start_index + num_points, len(df))
    # Need at least 2 points (current and one future) to get a relative pose
    if (end_index - start_index) < 2:
        return np.array([]).reshape(0, 2)

    pos_list = position[start_index:end_index]
    
    rpy_list = rotation[start_index:end_index]

    # Get current pose and its inverse
    if len(pos_list[0]) != 3:
        pos_current = np.array([pos_list[0][0], pos_list[0][1], 0.0])
    else:
        pos_current = pos_list[0]
    
    rpy_current = rpy_list[0]  # [0]  # 'yaw' is stored as a list [val]
    
    future_pos = pos_list[1:]
    future_rpy = rpy_list[1:]

    future_poses = np.hstack((future_pos, future_rpy))

    cur_poses = np.concatenate((pos_current, rpy_current))
    repeat_curr_pose = np.vstack((cur_poses,)* len(future_poses))
    future_poses_relative = np.array(to_robot_np(repeat_curr_pose, future_poses))

    odometry_in_frame_1 = future_poses_relative[:, :3]
    return odometry_in_frame_1


def make_labeling_dataframe(pkl):
    if 'policy' not in pkl:
        traj_data_i = {}
        steps = pkl['timestep']
        policy = np.zeros_like(steps, dtype=bool)
        dynamics = np.ones_like(steps, dtype=bool)
        interaction = np.zeros_like(steps, dtype=bool)
        notes = np.array(['']*len(steps), dtype=str)

    else:
        steps = pkl['timestep']
        policy = pkl['policy']
        dynamics = pkl['dynamics']
        interaction = pkl['interaction']
        notes = pkl['notes']

    refine_data = list(
                    zip(steps, policy, dynamics, interaction, notes))

    pd_frame = pd.DataFrame(refine_data, columns=['timestep', 'policy', 'dynamics', 'interaction', 'notes'])

    return pd_frame

def make_empty_coarse_labels():
    return {
        'Environment Labels': [],
        'Object Labels': [],
        'Additional Labels': []
    }

# --- Streamlit App UI ---

st.set_page_config(layout="wide")
st.title("Trajectory Visualization Tool")

# --- Sidebar ---
st.sidebar.markdown("# Control Panel")

# --- Trajectory Selection ---
root_dir = st.sidebar.text_input("Data Root Directory:", '.')

root_path = Path(root_dir)

if "animate" not in st.session_state:
    st.session_state.animate = False

if not root_path.exists():
    st.sidebar.error("Root directory not found. Please check the path.")
    st.stop()

cleaned_datasets = sorted(os.listdir(str(root_path)))

if FOLDER_NAME in cleaned_datasets:
    cleaned_datasets.remove(FOLDER_NAME)

selected_dataset = st.sidebar.selectbox("Select Dataset:", cleaned_datasets)

curr_path = os.path.join(str(root_path), selected_dataset)

dataset_traj = os.listdir(curr_path)

all_traj_and_length = []

for i in dataset_traj:
    name = i
    
    contains_policy_labels = 'traj_data_frame_labels.pkl' in os.listdir(os.path.join(curr_path, i))
    if not contains_policy_labels:
        display_name = f"***{name}***"
    else:
        display_name = name
    all_traj_and_length.append(display_name)
    
choose_traj = sorted(all_traj_and_length )

with st.sidebar:
    with st.container(height=300):
        selected_display_name = st.radio("Select Trajectory:\nLength : Name", choose_traj)
    traj_len = st.slider("Trajectory Length (steps):", 1, 50, 10)
    frame_rate = st.slider("Animation FPS:", 1, 30, 5)

    st.toggle("Animate", key='animate')


# --- Main Panel (Table and Visualization) ---

col1, col2 = st.columns([0.45, 0.55])

selected_traj_name = selected_display_name.replace('***', '')  # Remove asterisks if present

selected_traj_path = os.path.join(curr_path, selected_traj_name)

# --- Get the visualizer for the selected robot ---

all_data = read_trajectory_data(selected_traj_path)

position, rotation = clean_pickle_file(all_data)

df = None
coarse_labels = None
if 'traj_data_frame_labels.pkl' in os.listdir(selected_traj_path):
    with open(os.path.join(selected_traj_path, 'traj_data_frame_labels.pkl'), 'rb') as f:
        all_label_data = pickle.load(f)
    df = make_labeling_dataframe(all_label_data)

    with open(os.path.join(selected_traj_path, 'traj_data_bag_labels.yaml'), 'r') as f:
        coarse_labels = yaml.safe_load(f)

else:
    df = make_labeling_dataframe(all_data)
    coarse_labels = make_empty_coarse_labels()

st.sidebar.text_input("Current Robot", all_data['robot'])

try:
    visualizer = get_visualizer_for_robot(selected_traj_path, selected_dataset)
except Exception as e:
    st.error(f"Error initializing visualizer for '{selected_traj_path}': {e}")
    st.exception(e)
    st.stop()

with col1:
    st.subheader("Trajectory Data")

    number = st.number_input("Go to Frame:", min_value=0, max_value=len(df)-1, value=0, step=1)

    if number:
        frame_start = number
    else:
        frame_start = 0    

    with st.form(key='label_form'):
        editable_df = st.data_editor(
            df,
            width='content',
            disabled= ['timestep'],
            hide_index=True,
            column_order = ['timestep', 'dynamics', 'policy', 'interaction', 'notes'],
            column_config={
                'timestep': "Frame"
            }
        )

        if len(coarse_labels['Environment Labels']) == 0:
            defy = None
        else:
            defy = coarse_labels['Environment Labels']
        env_vars = st.multiselect(
            "Environment Labels",
            [i for i in ['indoor', 'outdoor', 'off-road', 'underground', 'social', 'rain', 'sun', 'low-light', 'snow']],
            default=None if len(coarse_labels['Environment Labels']) == 0 else coarse_labels['Environment Labels']
        )
        object_labels = st.multiselect(
            "Object Labels",
            [i for i in ['people', 'vehicles', 'stairs', 'sidewalks', 'crowd', "crosswalk"]],
            default=None if len(coarse_labels['Object Labels']) == 0 else coarse_labels['Object Labels']
        )

        old_label_edits = None
        if len(coarse_labels['Additional Labels']) > 0:
            old_label_edits = st.data_editor(
                pd.DataFrame(coarse_labels['Additional Labels'], columns = ["Current"]),
                hide_index=True)
        
        additional_labels = [st.text_input("Additional Labels (comma-separated):")]

        submit = st.form_submit_button("Save Labels")

    if submit:
        with open(os.path.join(selected_traj_path, 'traj_data_frame_labels.pkl'), 'wb') as f:
            pickle.dump(editable_df, f)
        
        coarse_labels['Environment Labels'] = env_vars
        coarse_labels['Object Labels'] = object_labels

        if old_label_edits is not None:
            old_labs = set(old_label_edits['Current'].to_list())
            entered_labs = set(additional_labels)
            all_labs = old_labs.union(entered_labs)
            coarse_labels['Additional Labels'] = list(all_labs)
        else:
            coarse_labels['Additional Labels'] = [additional_labels]

        label_dumper = {
            'Environment Labels': coarse_labels['Environment Labels'],
            'Object Labels': coarse_labels['Object Labels'],
            'Additional Labels': coarse_labels['Additional Labels']
        }

        with open(os.path.join(selected_traj_path, 'traj_data_bag_labels.yaml'), 'w') as f:
            yaml.dump(label_dumper, f, default_flow_style=False)

        coarse_labels = make_empty_coarse_labels()

        st.rerun()

with col2:
    # Create a placeholder for the image/animation
    placeholder = st.empty()

    if st.session_state.animate:
        # --- Animation Logic ---
        st.sidebar.info(f"Animating from frame {frame_start}...")

        frame_interval = 1.0 / frame_rate
        playback_started_at = time.monotonic()
        for i in range(frame_start, len(df)):
            frame_deadline = playback_started_at + (i - frame_start) * frame_interval
            frame_delay = frame_deadline - time.monotonic()
            if frame_delay < -frame_interval / 2:
                # Drop late frames rather than speeding through them to catch up.
                continue
            if frame_delay > 0:
                time.sleep(frame_delay)
            
            frame_image = load_frame_image(selected_traj_path, i)

            if frame_image is None:
                st.error(f"Could not load image: {selected_traj_path}/{i}.jpg")
                continue

            # --- Draw Trajectory for *this* frame ---
            # 1. Get relative trajectory
            
            # this doesn't assume any specific coordinate frame
            # conversion from robot's specific frame to optical frame
            # contained in the camera config class

            rel_poses_xy = get_relative_trajectory(position, rotation, i, traj_len)
            
            if rel_poses_xy.size == 0:
                placeholder.image(frame_image, channels="BGR", width='content',)
                continue

            # 2. Draw on image
            try:
                vis_image = visualizer.draw_trajectory_on_image(
                    frame_image, rel_poses_xy
                )
                # 3. Display
                with placeholder.container():
                    col1, col2 = st.columns([0.9,0.1])
                    col1.image(vis_image, channels="BGR", width='content',)
                    col2.write(f"Frame: {i}")
            except ValueError as e:
                # Catch image size mismatch errors
                st.error(f"Error drawing on frame {i}: {e}")
                placeholder.image(frame_image, channels="BGR", width='content',)

            # Check if user toggled 'Animate' off during loop
            if not st.session_state.animate:
                st.sidebar.info("Animation stopped.")

        playback_end = playback_started_at + (len(df) - frame_start) * frame_interval
        remaining_playback_time = playback_end - time.monotonic()
        if remaining_playback_time > 0:
            time.sleep(remaining_playback_time)
        
        st.rerun()

    else:
        # --- Single Frame Logic ---
        image = load_frame_image(selected_traj_path, frame_start)

        if image is None:
            st.error(f"Could not load image: {selected_traj_path}/{frame_start}.jpg")
        else:
            # 1. Get relative trajectory
            # this doesn't assume any specific coordinate frame
            # conversion from robot's specific frame to optical frame
            # contained in the camera config class
            rel_poses_xy = get_relative_trajectory(position, rotation, frame_start, traj_len)

            if rel_poses_xy.size > 0:

                # 2. Draw on image
                try:
                    vis_image = visualizer.draw_trajectory_on_image(image, rel_poses_xy)
                    placeholder.image(
                        vis_image, channels="BGR", width='content',
                    )
                except ValueError as e:
                    # Catch image size mismatch errors (e.g., from config)
                    st.error(f"Error drawing on frame: {e}")
                    st.exception(e)
                    placeholder.image(image, channels="BGR", width='content',)
            else:
                # No trajectory to draw, just show the image
                placeholder.image(image, channels="BGR", width='content',)
