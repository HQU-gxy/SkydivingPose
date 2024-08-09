import numpy as np  
from scipy.signal import savgol_filter  

def smooth_keypoints_savgol(keypoints, window_length=15, poly_order=2):  
    """  
    Apply Savitzky-Golay filter to smooth keypoints, preserving the batch dimension.  

    Parameters:  
    - keypoints: numpy array of keypoints with shape (batch_size, frames, num_keypoints, 2)  
    - window_length: the length of the filter window (must be odd)  
    - poly_order: the order of the polynomial used to fit the samples  

    Returns:  
    - smoothed_keypoints: numpy array of smoothed keypoints with the same shape as input  
    """  
    smoothed_keypoints = np.copy(keypoints)  

    # Iterate over each entry in the batch  
    for b in range(keypoints.shape[0]):  # Iterate over batch size  
        for i in range(keypoints.shape[2]):  # Iterate over keypoints  
            for j in range(keypoints.shape[3]):  # Iterate over x, y coordinates  
                smoothed_keypoints[b, :, i, j] = savgol_filter(  
                    keypoints[b, :, i, j],   
                    window_length=window_length,   
                    polyorder=poly_order  
                )  

    return smoothed_keypoints  