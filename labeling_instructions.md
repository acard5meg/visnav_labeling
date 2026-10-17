The saved labels can be found in the same directory as the images.

# Overview
After running `streamlit_app_labeling.py` the **Control Panel** on the left side has a number of options.<br/>

1. *Select Dataset*: Pick dataset you'll be labeling.
2. Once you've selected your dataset a list of trajectories will be populated. Bolded trajectories **do not** have a pickle file with labels.
3. *Trajectory length (steps)*: This is how many odometry measurements to project onto the image. 
4. *Animation FPS*: Speed of animation.
5. *Animate*: Slider to start the animation. <br/>

The main window **Trajectory Visualization Tool** has the following: <br/>

1. *Go to Frame*: You can either increment with the +/- or enter the frame number you'd like to jump to. ***If you change this the animation will restart at the frame entered.***
2. A spreadsheet with various columns which denote the frame-by-frame labeling. These will be defined in the next section.
3. *Environment labels*: Drop down menu which lists labels that are applied to the entire trajectory.
4. *Object labels*: Drop down menu which lists labels for objects in the environment that are applied to the entire trajectory.
5. *Additional Labels*: Field which allows you to write additional labels which are applied to entire trajectory if labels from 3 and 4 are insufficient.
6. *Save labels* button: Will save pickle files of frame level labels and trajectory level labels and bring you to the next unlabeled trajectory.
7. *Frame* slider: Another option to change which frame currently at. <br/>

***Please note***: The starting frame has a priority logic. If *Go to Frame* is non-zero this has priority for the starting frame, otherwise the *Frame* slider selects the beginning frame.


# Download The Data

This is one of multiple ways to download the dataset from HuggingFace. You can run the following command lines in a Python 3 virtual environment if preferred

```
pip install huggingface_hub

# Download the test dataset for labeling  (~10 GB)
python download_data.py --test --cache-dir <download_dir>

# Download the FULL dataset for labeling (~190 GB)
python download_data.py --cache-dir <download_dir>

# Use the current directory as the cache directory
python download_data.py --cache-dir .

# If --cache-dir is omitted, the script uses the current working directory
```

After downloading, the tarball for the data will be in `<download_dir>/snapshots/<some_hash_id>/`. Extract it (`test_labels.tar.gz` for the test dataset) to your desired location for the labeling app to work on.


# Running App
After selecting dataset, trajectory, trajectory length, animation FPS, and starting frame click *Animate* button. <br/>

The tool will be running and you will see a counter in the top right hand corner of the image which displays the current frame.

# Labeling
This section describes the current approach to labeling. ***NOTE: The spreadsheet acts like an Excel document allowing you to drag your selection by press and hold of square in bottom right corner of cell and dragging.***

## Frame-Level Labels
1. ***Backwards movement should be false for both dynamics and policy.***
2. Dynamics - If the robot is moving this should be selected. This includes instances where the robot is spinning in place or moving sideways.
3. Policy - Something in the image forces the robot to take a certain behavior. If multiple things in the image could force the robot behavior use hierarchical reasoning about what action is most dangerous. Examples below. 
- Following one of two paths of a fork in the road is good policy. 
- For areas where the robot is going straight, if the environment is less than 2 times the width of the robot (light green overlay is robot width) this is good policy because the tight space forces the robot behavior.
- Off-road. Bushes on either side of the robot forces it has to go straight is good policy. This also applies to a dirt path through a grass field.
- Off-road. Driving through puddle to follow ruts or the path is good for policy v. an open field and the robot drives through a puddle is bad for policy.
- Off-road. Once the robot has line of sight to a clear path and begins making a move to get onto path this is considered policy. 
- Outdoor. Following sidewalk is good policy.
- Social. Crowded area of people forces the robot to go straight is good for policy v. a few people where robot can take a number of trajectories is not good for policy.
- Social: With people around, if the people make a move to yield to the robot and allow it to pass this can still be considered good policy. Additionally, if the robot has no other choice but to get close to people or split a group this is also good policy. 
- We consider the robot embodiment when defining policy. For example, a differential drive robot should not use stairs or steps if it can be avoided.

4. We ***do not*** want to include "goal seeking" behavior which are actions taken by the operator which are arbitrary from the point of the robot. An example of this would be the robot is in an open area (from above definition someplace 2x larger than the robot width) and the robot makes a right turn. From the robot's point of view there's no reason for this.

5. Interaction - Interaction between the robot and the environment took place which caused robot to alter course.
- Broadly defined as a close proximity encounter which causes a sudden change in angular or linear velocity. 
- This could be a pedestrian interaction (e.g. pedestrian in front of robot stops which requires robot to stop) or interaction with environment which caused robot to make trajectory correction (e.g. robot turns onto new path and fallen tree in way requiring quick adjustment).
- If the robot can see the object for a long period of time and smoothly adjust trajectory this ***is not*** an interaction
- For social navigation (SCAND and GND): If the robot begins a maneuver within 40 steps due to an entity (animate or otherwise) it is considered an interaction. If the maneuver is initiated outside the 40 steps it is only policy.
  
6. Notes - Plain text box if want to add information about interaction. Please keep it short.

## Environment labels
Please use judgement when choosing these labels. At least 10% of the trajectory should include whatever label you choose. <br/>

The difference between outdoor and off-road is that outdoor trajectories have clearly marked paths e.g. sidewalks whereas off-road does not. <br/>

When the robot must navigate between and among a group of individuals this is considered a crowd. A crowd is a superset of people so if crowd is selected, people doesn't need to be selected. If the robot observes individuals that are not in its way or is moving around individuals without having to navigate in between individuals this is considered people. The robot only needs to encounter a crowd once for the label to be used. <br/>

***Additional Labels***: Please be thoughtful and judicious with this entry. 

