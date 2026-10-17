#!/usr/bin/env python
import numpy as np
import math


def quaternion_to_angle(q):
    """
    Convert a quaternion into an angle in radians.
    In roll, pitch, yaw order
    """
    def _quat_xyzw(quat):
        if hasattr(quat, "x") and hasattr(quat, "y") and hasattr(quat, "z") and hasattr(quat, "w"):
            return float(quat.x), float(quat.y), float(quat.z), float(quat.w)
        if isinstance(quat, (list, tuple, np.ndarray)) and len(quat) == 4:
            return float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        raise ValueError("Quaternion must be an object with x/y/z/w or an iterable [x, y, z, w]")

    if isinstance(q, list) and len(q) > 0 and not all(isinstance(v, (int, float, np.floating)) for v in q):
        return np.array([euler_from_quaternions(*_quat_xyzw(quat)) for quat in q], dtype=np.float64).tolist()

    return list(euler_from_quaternions(*_quat_xyzw(q)))


def euler_from_quaternions(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians


def to_robot_np(Robot_frame, P_relative):
    '''
    Filter out backward frames using 6 DOF 
    '''
    SE3 = True

    Robot_frame = np.asarray(Robot_frame, dtype=np.float64)
    P_relative = np.asarray(P_relative, dtype=np.float64)

    if len(Robot_frame.shape) == 1:
        Robot_frame = np.expand_dims(Robot_frame, axis=0)

    if len(P_relative.shape) == 1:
        P_relative = np.expand_dims(P_relative, axis=0)
  
    if len(Robot_frame.shape) > 2 or len(P_relative.shape) > 2:
        raise ValueError(f"Input must be 1D for  unbatched and 2D for batched got input dimensions {Robot_frame.shape} and {P_relative.shape}")
    
    if Robot_frame.shape != P_relative.shape:
        raise ValueError("Input tensors must have same shape")
    
    if Robot_frame.shape[-1] != 6 and Robot_frame.shape[-1] != 3:
        raise ValueError(f"Input tensors must have last dim equal to 6 for SE3 and 3 for SE2 got {Robot_frame.shape[-1]}")
    
    if Robot_frame.shape[-1] == 3:
        SE3 = False
        Robot_frame_ = np.zeros((Robot_frame.shape[0], 6), dtype=Robot_frame.dtype)
        Robot_frame_[:, [0,1,5]] = Robot_frame
        Robot_frame = Robot_frame_
        P_relative_ = np.zeros((P_relative.shape[0], 6), dtype=P_relative.dtype)
        P_relative_[:, [0,1,5]] = P_relative
        P_relative = P_relative_
        
    """ Assemble a batch of SE3 homogeneous matrices from a batch of 6DOF poses """
    batch_size = Robot_frame.shape[0]
    ones = np.ones((batch_size, 1), dtype=P_relative.dtype)
    T1 = np.zeros((batch_size, 4, 4), dtype=Robot_frame.dtype)
    T2 = np.zeros((batch_size, 4, 4), dtype=P_relative.dtype)

    T1[:, :3, :3] = euler_to_rotation_matrix(Robot_frame[:, 3:])
    T2[:, :3, :3] = euler_to_rotation_matrix(P_relative[:, 3:])
    T1[:, :3,  3] = Robot_frame[:, :3]
    T2[:, :3,  3] = P_relative[:, :3]
    T1[:,  3,  3] = 1
    T2[:,  3,  3] = 1 
    
    T1_inv = np.linalg.inv(T1)
    tf3_mat = T2 @ T1_inv
    
    transform = np.zeros_like(Robot_frame)
    transform[:, :3] = (T1_inv @ np.expand_dims(np.concatenate((P_relative[:, :3], ones), axis=1), axis=2)).squeeze(axis=2)[:, :3]
    transform[:, 3:] = extract_euler_angles_from_se3_batch(tf3_mat)
    
    if not SE3:
        transform = transform[:, [0,1,5]]
    
    return transform
def euler_to_rotation_matrix(euler_angles):
    """ Convert Euler angles to a rotation matrix """
    euler_angles = np.asarray(euler_angles, dtype=np.float64)
    if euler_angles.ndim == 1:
        euler_angles = np.expand_dims(euler_angles, axis=0)
    if euler_angles.ndim != 2 or euler_angles.shape[1] != 3:
        raise ValueError("Input must have shape (batch, 3) or (3,)")

    # Compute sin and cos for Euler angles
    cos = np.cos(euler_angles)
    sin = np.sin(euler_angles)
    zero = np.zeros_like(euler_angles[:, 0])
    one = np.ones_like(euler_angles[:, 0])
    # Constructing rotation matrices (assuming 'xyz' convention for Euler angles)
    R_x = np.stack([one, zero, zero, zero, cos[:, 0], -sin[:, 0], zero, sin[:, 0], cos[:, 0]], axis=1).reshape(-1, 3, 3)
    R_y = np.stack([cos[:, 1], zero, sin[:, 1], zero, one, zero, -sin[:, 1], zero, cos[:, 1]], axis=1).reshape(-1, 3, 3)
    R_z = np.stack([cos[:, 2], -sin[:, 2], zero, sin[:, 2], cos[:, 2], zero, zero, zero, one], axis=1).reshape(-1, 3, 3)

    return R_z @ R_y @ R_x
    
def extract_euler_angles_from_se3_batch(tf3_matx):
    # Validate input shape
    tf3_matx = np.asarray(tf3_matx)
    if tf3_matx.ndim != 3 or tf3_matx.shape[1:] != (4, 4):
        raise ValueError("Input tensor must have shape (batch, 4, 4)")

    # Extract rotation matrices
    rotation_matrices = tf3_matx[:, :3, :3]

    # Initialize tensor to hold Euler angles
    batch_size = tf3_matx.shape[0]
    euler_angles = np.zeros((batch_size, 3), dtype=tf3_matx.dtype)

    # Compute Euler angles
    euler_angles[:, 0] = np.arctan2(rotation_matrices[:, 2, 1], rotation_matrices[:, 2, 2])  # Roll
    euler_angles[:, 1] = np.arctan2(-rotation_matrices[:, 2, 0], np.sqrt(rotation_matrices[:, 2, 1] ** 2 + rotation_matrices[:, 2, 2] ** 2))  # Pitch
    euler_angles[:, 2] = np.arctan2(rotation_matrices[:, 1, 0], rotation_matrices[:, 0, 0])  # Yaw

    return euler_angles
