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

from app.deploy.gen_labelme_kps import gen_video_kpts
from app.deploy.preprocess import h36m_coco_format, revise_kpts

plt.switch_backend('agg')
warnings.filterwarnings('ignore')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sys.path.append(os.getcwd())
from common.utils import *
from common.camera import *
from model.mixste.hot_mixste import Model




def get_pose2D(video_path, output_dir, video_name):
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print('\nGenerating 2D pose...')
    output_dir = "/home/zlt/Documents/SkydivingPose/output/labelme"
    with torch.no_grad():
        # the first frame of the video should be detected a person
        keypoints, scores = gen_video_kpts(video_path, video_name, output_dir, num_peroson=1, gen_output=True)

    print("=====keypoints", keypoints.shape)
    # output_dir = f'output/wholebody/{video_name}'  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='aaa.mp4', help='input video')
    parser.add_argument('--gpu', type=str, default='0', help='input video')
    parser.add_argument('--fix_z', action='store_true', help='fix z axis')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # video_path = './sample/' + args.video
    video_path = '/home/public/leaving/57-p1-14.mp4'

    video_name = video_path.split('/')[-1].split('.')[0]
    output_dir = './output/' + video_name + '/'

    get_pose2D(video_path, output_dir, video_name)


