import numpy as np

joint_names = [  
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',  
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',  
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',  
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'  
]  

joint_indices = list(range(17))  


def extract_keypoints(original_keypoints):  
    """  
    Extract the first 17 keypoints from a keypoints tensor with shape (1, 179, 133, 2).  

    Parameters:  
    original_keypoints (torch.Tensor): The original keypoints tensor of shape (1, 179, 133, 2).  

    Returns:  
    torch.Tensor: A new tensor with shape (1, 179, 17, 2), containing only the keypoints of interest.  
    """  
    # Validate input tensor shape  
    assert original_keypoints.shape[2] >= 17, "Input tensor does not have enough keypoints for extraction."  

    # Extract the first 17 keypoints  
    extracted_keypoints = original_keypoints[:, :, :17, :]  
    
    return extracted_keypoints  


def extract_keypoints_with_foot(original_keypoints):  
    """  
    Extract the first 17 keypoints from a keypoints tensor with shape (1, 179, 133, 2).  

    Parameters:  
    original_keypoints (torch.Tensor): The original keypoints tensor of shape (1, 179, 133, 2).  

    Returns:  
    torch.Tensor: A new tensor with shape (1, 179, 22, 2), containing only the keypoints of interest.  
    """  
    # Validate input tensor shape  
    assert original_keypoints.shape[2] >= 17, "Input tensor does not have enough keypoints for extraction."  

    # Extract the first 17 keypoints  
    extracted_keypoints = original_keypoints[:, :, :22, :]  
    
    return extracted_keypoints  