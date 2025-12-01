import torch
from omni.isaac.lab.envs import ManagerBasedRLEnv

def remap_h1_to_smpl(h1_joint_pos_rel, h1_root_pos, h1_root_rot):
    """
    Remap H1 robot joint positions (and root) to SMPL 24-joint format (approximate).
    
    Args:
        h1_joint_pos_rel (torch.Tensor): H1 joint positions relative to default (Batch, 19)
        h1_root_pos (torch.Tensor): Root position (Batch, 3)
        h1_root_rot (torch.Tensor): Root rotation (quaternion) (Batch, 4)
        
    Returns:
        torch.Tensor: SMPL-like joint positions (Batch, 24, 3)
    """
    batch_size = h1_joint_pos_rel.shape[0]
    device = h1_joint_pos_rel.device
    
    # H1 Joints (19 DOF) indices based on typical H1 config:
    # 0: left_hip_yaw, 1: left_hip_roll, 2: left_hip_pitch, 3: left_knee, 4: left_ankle
    # 5: right_hip_yaw, 6: right_hip_roll, 7: right_hip_pitch, 8: right_knee, 9: right_ankle
    # 10: torso_joint
    # 11: left_shoulder_pitch, 12: left_shoulder_roll, 13: left_shoulder_yaw, 14: left_elbow
    # 15: right_shoulder_pitch, 16: right_shoulder_roll, 17: right_shoulder_yaw, 18: right_elbow
    
    # SMPL Joints (24) indices:
    # 0: Pelvis, 1: L_Hip, 2: R_Hip, 3: Spine1, 4: L_Knee, 5: R_Knee, 6: Spine2, 7: L_Ankle, 8: R_Ankle, 9: Spine3
    # 10: L_Foot, 11: R_Foot, 12: Neck, 13: L_Collar, 14: R_Collar, 15: Head, 16: L_Shoulder, 17: R_Shoulder
    # 18: L_Elbow, 19: R_Elbow, 20: L_Wrist, 21: R_Wrist, 22: L_Hand, 23: R_Hand
    
    # Placeholder for mapped positions (Batch, 24, 3)
    # Ideally, we should compute Forward Kinematics here to get XYZ positions.
    # Since we don't have full FK readily available in this lightweight function,
    # we will approximate using heuristic mapping if inputs are joint angles,
    # BUT wait, MotionCLIP expects XYZ positions for the 'xyz' pose_rep.
    # RL environment usually provides joint angles (dof_pos).
    # To get XYZ, we should use Isaac Lab's asset data (body positions).
    
    # We need access to body link positions, not just joint angles.
    # This function assumes it's called within a context where we can access env.scene['robot'].data.body_pos_w
    return None # Placeholder, logic needs env access

def get_h1_body_pos_mapped_to_smpl(env: ManagerBasedRLEnv):
    """
    Extract H1 body link positions and map them to SMPL 24 joints.
    """
    # Get all body positions (Batch, Num_Bodies, 3)
    # H1 body names (approx): pelvis, left_hip_yaw_link, ..., torso_link, ...
    # We need to manually check indices or use find_bodies
    
    robot = env.scene["robot"]
    body_pos_w = robot.data.body_pos_w # (Batch, Num_Bodies, 3)
    
    # Map H1 bodies to SMPL joints
    # Note: This mapping depends on H1 USD structure. 
    # Assuming standard H1 names.
    
    # Helper to get index
    def get_idx(pattern):
        ids = robot.find_bodies(pattern)[0]
        if len(ids) > 0:
            return ids[0]
        return 0 # Fallback to pelvis (0)
        
    # SMPL 24 Joint Mapping (Indices 0-23)
    # 0: Pelvis
    idx_pelvis = get_idx("pelvis")
    # 1: L_Hip (Approx: left_hip_roll_link)
    idx_l_hip = get_idx("left_hip_roll_link")
    # 2: R_Hip
    idx_r_hip = get_idx("right_hip_roll_link")
    # 3: Spine1 (Approx: torso_link)
    idx_spine1 = get_idx("torso_link")
    # 4: L_Knee (left_knee_link)
    idx_l_knee = get_idx("left_knee_link")
    # 5: R_Knee
    idx_r_knee = get_idx("right_knee_link")
    # 6: Spine2 (Same as Spine1 or higher?) -> Use torso_link
    idx_spine2 = idx_spine1
    # 7: L_Ankle (left_ankle_link)
    idx_l_ankle = get_idx("left_ankle_link")
    # 8: R_Ankle
    idx_r_ankle = get_idx("right_ankle_link")
    # 9: Spine3 (Same as Spine1)
    idx_spine3 = idx_spine1
    # 10: L_Foot (Approx: left_ankle_link or foot if exists) -> left_ankle_link
    idx_l_foot = idx_l_ankle
    # 11: R_Foot
    idx_r_foot = idx_r_ankle
    # 12: Neck (No neck in H1? Use torso or logo_link?) -> torso_link
    idx_neck = idx_spine1
    # 13: L_Collar (left_shoulder_roll_link)
    idx_l_collar = get_idx("left_shoulder_roll_link")
    # 14: R_Collar
    idx_r_collar = get_idx("right_shoulder_roll_link")
    # 15: Head (No head? Use torso)
    idx_head = idx_spine1
    # 16: L_Shoulder (left_shoulder_yaw_link)
    idx_l_shoulder = get_idx("left_shoulder_yaw_link")
    # 17: R_Shoulder
    idx_r_shoulder = get_idx("right_shoulder_yaw_link")
    # 18: L_Elbow (left_elbow_link)
    idx_l_elbow = get_idx("left_elbow_link")
    # 19: R_Elbow (right_elbow_link)
    idx_r_elbow = get_idx("right_elbow_link")
    # 20: L_Wrist (left_hand_link? or elbow) -> left_elbow_link (H1 usually has no hands in base)
    idx_l_wrist = idx_l_elbow 
    # 21: R_Wrist
    idx_r_wrist = idx_r_elbow
    # 22: L_Hand
    idx_l_hand = idx_l_elbow
    # 23: R_Hand
    idx_r_hand = idx_r_elbow
    
    indices = [
        idx_pelvis, idx_l_hip, idx_r_hip, idx_spine1, idx_l_knee, idx_r_knee, idx_spine2, idx_l_ankle, idx_r_ankle, idx_spine3,
        idx_l_foot, idx_r_foot, idx_neck, idx_l_collar, idx_r_collar, idx_head, idx_l_shoulder, idx_r_shoulder,
        idx_l_elbow, idx_r_elbow, idx_l_wrist, idx_r_wrist, idx_l_hand, idx_r_hand
    ]
    
    # Gather positions (Batch, 24, 3)
    smpl_pos = body_pos_w[:, indices, :]
    
    # Normalize relative to pelvis (Root centering)
    root_pos = smpl_pos[:, 0:1, :]
    smpl_pos_rel = smpl_pos - root_pos
    
    return smpl_pos_rel

