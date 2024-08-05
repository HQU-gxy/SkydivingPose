import os   
import sys  
import cv2  
import glob  
import torch  
import argparse  
import numpy as np  
from tqdm import tqdm  
from IPython import embed  

import warnings  
import matplotlib  
import matplotlib.pyplot as plt   
from mpl_toolkits.mplot3d import Axes3D  
import matplotlib.gridspec as gridspec  

from app.deploy.gen_kps import gen_video_kpts  
from app.deploy.preprocess import h36m_coco_format, revise_kpts  

plt.switch_backend('agg')  
warnings.filterwarnings('ignore')  
matplotlib.rcParams['pdf.fonttype'] = 42  
matplotlib.rcParams['ps.fonttype'] = 42  

sys.path.append(os.getcwd())  
from common.utils import *  
from common.camera import *  
from model.mixste.hot_mixste import Model  


def show2Dpose(kps, img):  
    colors = [(138, 201, 38),  
              (25, 130, 196),  
              (255, 202, 58)]   

    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],  
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],  
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]  

    LR = [3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]  

    thickness = 3  

    for j, c in enumerate(connections):  
        start = map(int, kps[c[0]])  
        end = map(int, kps[c[1]])  
        start = list(start)  
        end = list(end)  
        cv2.line(img, (start[0], start[1]), (end[0], end[1]), colors[LR[j] - 1], thickness)  
        cv2.circle(img, (start[0], start[1]), thickness=-1, color=colors[LR[j] - 1], radius=3)  
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=colors[LR[j] - 1], radius=3)  

    return img  


def get_pose2D(video_path, output_dir, video_name):  
    cap = cv2.VideoCapture(video_path)  
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  

    print('\nGenerating 2D pose...')  
    with torch.no_grad():  
        keypoints, scores = gen_video_kpts(video_path, num_peroson=1, gen_output=True)  

    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)  
    re_kpts = revise_kpts(keypoints, scores, valid_frames)  
    print('Generating 2D pose successfully!')  

    # output_dir += 'input_2D/'  
    os.makedirs(output_dir, exist_ok=True)  

    output_npz = output_dir + f'{video_name}_keypoints_2d.npz'  
    np.savez_compressed(output_npz, reconstruction=keypoints)  


def img2video(video_path, output_dir):  
    cap = cv2.VideoCapture(video_path)  
    fps = int(cap.get(cv2.CAP_PROP_FPS)) + 5  

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  

    names = sorted(glob.glob(os.path.join(output_dir + 'pose/', '*.png')))  
    img = cv2.imread(names[0])  
    size = (img.shape[1], img.shape[0])  

    videoWrite = cv2.VideoWriter(output_dir + video_name + '.mp4', fourcc, fps, size)   

    for name in names:  
        img = cv2.imread(name)  
        videoWrite.write(img)  

    videoWrite.release()  


def showimage(ax, img):  
    ax.set_xticks([])  
    ax.set_yticks([])   
    plt.axis('off')  
    ax.imshow(img)  


if __name__ == "__main__":  
    parser = argparse.ArgumentParser()  
    parser.add_argument('--folder', type=str, default='sample/test-001', help='input folder containing videos')  
    parser.add_argument('--gpu', type=str, default='0', help='GPU device number')  
    parser.add_argument('--fix_z', action='store_true', help='fix z axis')  

    args = parser.parse_args()  

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  

    video_folder = args.folder  

    video_files = glob.glob(os.path.join(video_folder, '*.mp4'))  

    for video_file in video_files:  
        video_name = os.path.splitext(os.path.basename(video_file))[0]  
        output_dir = './output/test-001/'
        # output_dir = os.path.join('./output/test-001', video_name)  
        get_pose2D(video_file, output_dir, video_name)