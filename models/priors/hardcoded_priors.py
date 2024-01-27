import torch
import torch.nn as nn
import math

# Keypoint mappings
keypoints_map = {
    "neck": 1, "left_shoulder": 2, "right_shoulder": 5,
    "left_elbow": 3, "right_elbow": 6, "left_wrist": 4, "right_wrist": 7,
    "hip": 8, "left_hip": 9, "right_hip": 12, "left_knee": 10,
    "right_knee": 13, "left_ankle": 11, "right_ankle": 14
}


def symmetry_loss(pred_poses: torch.Tensor):
    """
    Custom loss function that accounts for symmetry. Minimizes the distance ratios to be as close to 1 as possible.
    Args:
        pred_poses: A tensor of shape (batch_size, num_joints(19), 3) representing predicted 3D poses.

    Returns:
        Symmetry loss.
    """
    mse_loss = nn.MSELoss()
    loss = 0.0
    angle_loss = 0.0

    symmetric_pairs = [
        ("neck", "left_shoulder", "neck", "right_shoulder"),
        ("left_shoulder", "left_elbow", "right_shoulder", "right_elbow"),
        ("left_elbow", "left_wrist", "right_elbow", "right_wrist"),
        ("hip", "left_hip", "hip", "right_hip"),
        ("left_hip", "left_knee", "right_hip", "right_knee"),
        ("left_knee", "left_ankle", "right_knee", "right_ankle")
    ]

    eps = 1e-6  # To avoid dividing stuff by 0

    for left_start, left_end, right_start, right_end in symmetric_pairs:
        left_distance = torch.linalg.norm(
            pred_poses[:, keypoints_map[left_start]] - pred_poses[:, keypoints_map[left_end]], dim=1)
        right_distance = torch.linalg.norm(
            pred_poses[:, keypoints_map[right_start]] - pred_poses[:, keypoints_map[right_end]], dim=1)

        ratio1 = left_distance/(right_distance + eps)  # avoids dividing by 0
        ratio2 = right_distance/(left_distance+eps)  # avoids dividing by 0
        ones = torch.ones_like(ratio1)
        loss += (mse_loss(ratio1, ones) + mse_loss(ratio2, ones))

    # Calculating vectors for angle constraints (shoulders and hips)
    vector_neck_ls = pred_poses[:, keypoints_map['neck']
                                ] - pred_poses[:, keypoints_map['left_shoulder']]
    vector_neck_rs = pred_poses[:, keypoints_map['neck']
                                ] - pred_poses[:, keypoints_map['right_shoulder']]
    vector_hip_lh = pred_poses[:, keypoints_map['hip']
                               ] - pred_poses[:, keypoints_map['left_hip']]
    vector_hip_rh = pred_poses[:, keypoints_map['hip']
                               ] - pred_poses[:, keypoints_map['right_hip']]

    # Normalizing the vectors vectors (adding eps to avoid dividing by 0)
    vector_neck_ls = vector_neck_ls / \
        (torch.linalg.norm(vector_neck_ls, dim=1, keepdim=True) + eps)
    vector_neck_rs = vector_neck_rs / \
        (torch.linalg.norm(vector_neck_rs, dim=1, keepdim=True) + eps)
    vector_hip_lh = vector_hip_lh / \
        (torch.linalg.norm(vector_hip_lh, dim=1, keepdim=True) + eps)
    vector_hip_rh = vector_hip_rh / \
        (torch.linalg.norm(vector_hip_rh, dim=1, keepdim=True) + eps)

    # Calculating the dot products to fund the angle between.... using cos(theta) = a.b
    dot_product_neck = torch.sum(vector_neck_ls * vector_neck_rs, dim=1)
    dot_product_hip = torch.sum(vector_hip_lh * vector_hip_rh, dim=1)

    # cos(180) = -1
    straight_line_target = torch.full_like(dot_product_neck, -1)
    # calculating the angle loss
    angle_loss += mse_loss(dot_product_neck, straight_line_target)
    angle_loss += mse_loss(dot_product_hip, straight_line_target)

    # joint_angle_loss_knee = joint_angle_loss(pred_poses=pred_poses)
    hip_length_loss = hip_leg_length_loss(pred_poses=pred_poses)

    hip_leg_length_prior = hip_leg_length_loss(pred_poses=pred_poses)
    # adding the symetry loss to the angle loss
    total_loss = (1e-3 * loss)  # + angle_loss + hip_length_loss
    return total_loss


def bone_proportions_prior(pred_poses):
    """Ensures the proportions are met

    head_to_body_ratio : 1/8
    waist to height ratio = (0.46, 0.45) in men, (0.45, 0.49) in women

    Args:
        pred_poses (torch.Tensor): the tensor including the keypoints
    """
    ...


def hip_leg_length_loss(pred_poses, min_hip_ratio=0.5):
    """
    Ensures the thigh length is greather than the length of the hip
    """
    mse_loss = nn.MSELoss()

    # calculating the distance between the left hip to the right hip
    left_hip_pos = pred_poses[:, keypoints_map['left_hip']]
    right_hip_pos = pred_poses[:, keypoints_map['right_hip']]
    hip_width = torch.linalg.norm(left_hip_pos - right_hip_pos, dim=1)

    # Calculating the distance between the left hip to the left knee
    left_knee_pos = pred_poses[:, keypoints_map['left_knee']]
    leg_length = torch.linalg.norm(left_hip_pos - left_knee_pos, dim=1)

    # Ensuring the hip with is about 0.8 times the thigh length
    min_hip_width = leg_length * min_hip_ratio

    # Obtaining the loss
    loss = mse_loss(torch.relu(hip_width - leg_length), torch.zeros_like(hip_width)) + \
        mse_loss(torch.relu(min_hip_width - hip_width),
                 torch.zeros_like(hip_width))

    return loss


def joint_angle_loss(pred_poses):
    """
    Calculate joint angle loss with consideration for natural bending directions of elbows and knees. 
    I ommited this because it doesnt seem to work well.
    """
    mse_loss = nn.MSELoss()
    epsilon = 1e-6  # Prevent division by zero

    # Define keypoints for limbs
    limb_points = {
        'left_elbow': ('left_shoulder', 'left_elbow', 'left_wrist'),
        'right_elbow': ('right_shoulder', 'right_elbow', 'right_wrist'),
        'left_knee': ('left_hip', 'left_knee', 'left_ankle'),
        'right_knee': ('right_hip', 'right_knee', 'right_ankle')
    }

    joint_loss = 0.0

    for limb, (upper, joint, lower) in limb_points.items():
        # For elbows
        if 'elbow' in limb:
            upper_joint_vector = pred_poses[:, keypoints_map[joint]
                                            ] - pred_poses[:, keypoints_map[upper]]
            joint_lower_vector = pred_poses[:, keypoints_map[lower]
                                            ] - pred_poses[:, keypoints_map[joint]]
        # For knees
        elif 'knee' in limb:
            upper_joint_vector = pred_poses[:, keypoints_map[upper]
                                            ] - pred_poses[:, keypoints_map[joint]]
            joint_lower_vector = pred_poses[:, keypoints_map[joint]
                                            ] - pred_poses[:, keypoints_map[lower]]

        # Normalizing vectors
        upper_joint_vector_norm = upper_joint_vector / \
            (torch.linalg.norm(upper_joint_vector, dim=1, keepdim=True) + epsilon)
        joint_lower_vector_norm = joint_lower_vector / \
            (torch.linalg.norm(joint_lower_vector, dim=1, keepdim=True) + epsilon)

        # cosine of angle (dot product in this case)
        dot_product = torch.sum(
            upper_joint_vector_norm * joint_lower_vector_norm, dim=1)

        # radians to degrees
        angle = torch.acos(dot_product) * (180 / math.pi)

        # Penalize angles less than 180 degrees
        joint_loss += mse_loss(torch.relu(180 - angle),
                               torch.zeros_like(angle))

    return joint_loss
