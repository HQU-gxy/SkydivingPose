import os 
import sys
from typing import Literal
import cv2
import glob
import copy
import torch
import argparse
import numpy as np
from tqdm import tqdm
# from demo.lib.preprocess import h36m_coco_format, revise_kpts
# from demo.lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose
from IPython import embed

import warnings
import matplotlib
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

from app.deploy.gen_wholebody_kps import gen_video_kpts
from app.deploy.preprocess import h36m_coco_format, revise_kpts

plt.switch_backend('agg')
warnings.filterwarnings('ignore')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sys.path.append(os.getcwd())
from common.utils import *
from common.camera import *
from model.mixste.hot_mixste import Model




def visualize_and_save_keypoints(keypoints, video_path, output_dir):  
    # Open video file  
    cap = cv2.VideoCapture(video_path)  
    frame_count = 0  

    # Ensure output directory exists  
    os.makedirs(output_dir, exist_ok=True)  
    
    while cap.isOpened():  
        ret, frame = cap.read()  
        if not ret:  
            break  

        if frame_count >= keypoints.shape[1]:  # Check if frame_count exceeds number of keypoints frames  
            break  

        # Get keypoints for the current frame  
        frame_keypoints = keypoints[0, frame_count]  # As per your data description  

        # Loop through each keypoint (x, y) pair  
        for i in range(frame_keypoints.shape[0]):  
            x, y = frame_keypoints[i]  
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)  

        # Save the frame with keypoints visualized  
        frame_filename = os.path.join(output_dir, f'frame_{frame_count:04d}.png')  
        cv2.imwrite(frame_filename, frame)  

        frame_count += 1  

    cap.release()  

def get_pose2D(video_path, output_dir, video_name):
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print('\nGenerating 2D pose...')
    with torch.no_grad():
        # the first frame of the video should be detected a person
        keypoints, scores = gen_video_kpts(video_path, output_dir, num_peroson=1, gen_output=True)

    print("=====keypoints", keypoints.shape)
    output_dir = f'output/wholebody/{video_name}'  
    visualize_and_save_keypoints(keypoints, video_path, output_dir)  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='aaa.mp4', help='input video')
    parser.add_argument('--gpu', type=str, default='0', help='input video')
    parser.add_argument('--fix_z', action='store_true', help='fix z axis')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # video_path = './sample/' + args.video
    video_path = '/home/public/leaving/54-p1-14.mp4'

    video_name = video_path.split('/')[-1].split('.')[0]
    output_dir = './output/' + video_name + '/'

    get_pose2D(video_path, output_dir, video_name)


