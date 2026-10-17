# Trajectory Visualization Toolkit

A simple Python library for projecting 3D vehicle trajectories onto a 2D camera image, as seen in many autonomous driving visualizations. This has been tested with Python 3.10 and 3.12.

***CRITICAL***: Must clone repository into same directory where top-level dataset directories are.

*streamlit_app.py* : projection of 3D trajectories onto 2D camera image \
*streamlit_app_labeling.py* : projection of 3D trajectories onto 2D camera image. Additional functionality to create labels for trajectory

# Installation

To install the core library:

1. Create python3 environment: `python3 -m venv [environment-name]`

2. Activate environment: `source [environment-name]/bin/activate`

3. Import required packages: `pip3 install -r requirements.txt`

## Running the Streamlit App

1. Install the app dependencies (see above).

2. Run the app:
```bash
streamlit run traj_viz/streamlit_app_labeling.py
```

3. In the Streamlit sidebar, select the robot config you just created from the dropdown menu.

## Roadmap / Future Features

Here are some potential features to expand the toolkit, especially for research in robotics and deep learning:

- [ ] **Add Bird's-Eye View (BEV) Panel:** Add a `matplotlib` panel to the Streamlit app to show a top-down 2D map of the full trajectory and the current robot position/heading.
- [ ] **Prediction vs. Ground Truth:** Update the visualizer to accept and draw multiple trajectories, each with its own color (e.g., a green "ground truth" path and a red "predicted" path from a model).
- [ ] **Differentiable Projection (PyTorch):** Create a parallel `projection_torch.py` using `torch` and `kornia` to build a differentiable projection layer. This would allow computing image-space losses for end-to-end model training.
- [ ] **Visualize Uncertainty:** Modify the visualizer to accept covariance or standard deviation for each waypoint, drawing a "cone" of uncertainty instead of a fixed-width polygon.
- [ ] **ROS Bag Support:** Add a data loader to `streamlit_app.py` that can read and parse `/tf`, `/image_raw`, and `/odometry` topics from ROS bags to generate the data `DataFrame` on the fly.
- [ ] **Streamlit App: Data Export:** Add a button to save the current visualization or animation as an `mp4` video.