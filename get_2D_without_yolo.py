import base64
import os 
from pathlib import Path
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

sys.path.append("../")
from app.deploy.gen_wholebody_kps import gen_video_kpts, gen_frame_kpts_direct_bbox
# from app.deploy.preprocess import h36m_coco_format, revise_kpts
# from app.utils.wholebody_skeleton import extract_keypoints, extract_keypoints_with_foot
# from app.deploy.smooth_keypoints import smooth_keypoints_savgol

plt.switch_backend('agg')
warnings.filterwarnings('ignore')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sys.path.append(os.getcwd())
from common.utils import *
from common.camera import *
from model.mixste.hot_mixste import Model


# Keypoint pairs for connections  
connections = [  
    # Head connections  
    (0, 1),   # nose - left_eye  
    (0, 2),   # nose - right_eye  
    (1, 3),   # left_eye - left_ear  
    (2, 4),   # right_eye - right_ear  

    # Arm connections  
    (5, 7),   # left_shoulder - left_elbow  
    (7, 9),   # left_elbow - left_wrist  
    (6, 8),   # right_shoulder - right_elbow  
    (8, 10),  # right_elbow - right_wrist  

    # Torso connections  
    (3, 5),   # left_ear - left_shoulder  
    (4, 6),   # right_ear - right_shoulder  
    (5, 6),   # left_shoulder - right_shoulder  
    (5, 11),  # left_shoulder - left_hip  
    (6, 12),  # right_shoulder - right_hip  
    (11, 12), # left_hip - right_hip  

    # Leg connections  
    (11, 13), # left_hip - left_knee  
    (13, 15), # left_knee - left_ankle  
    (15, 17), # left_ankle - left_big_toe  
    (17, 18), # left_big_toe - left_small_toe  
    (15, 19), # left_ankle - left_heel  
    (12, 14), # right_hip - right_knee  
    (14, 16), # right_knee - right_ankle  
    (16, 20), # right_ankle - right_big_toe  
    (20, 21)  # right_big_toe - right_small_toe  
]  


def show2Dpose(kps, img):
    colors = [(138, 201, 38),
              (25, 130, 196),
              (255, 202, 58)] 

    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    LR = [3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]

    thickness = 3

    for j,c in enumerate(connections):
        start = map(int, kps[c[0]])
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)
        cv2.line(img, (start[0], start[1]), (end[0], end[1]), colors[LR[j]-1], thickness)
        cv2.circle(img, (start[0], start[1]), thickness=-1, color=colors[LR[j]-1], radius=3)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=colors[LR[j]-1], radius=3)

    return img




def show3Dpose(vals, ax, fix_z):
    ax.view_init(elev=15., azim=70)

    colors = [(138/255, 201/255, 38/255),
            (255/255, 202/255, 58/255),
            (25/255, 130/255, 196/255)]

    I = np.array( [0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array( [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    LR = [3, 3, 3, 3, 3, 3, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1]

    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        ax.plot(x, y, z, lw=3, color = colors[LR[i]-1])

    RADIUS = 0.72

    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])

    if fix_z:
        left_z = max(0.0, -RADIUS+zroot)
        right_z = RADIUS+zroot
        # ax.set_zlim3d([left_z, right_z])
        ax.set_zlim3d([0, 1.5])
    else:
        ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])

    ax.set_aspect('equal') # works fine in matplotlib==2.2.2 or 3.7.1

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white) 
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom = False)
    ax.tick_params('y', labelleft = False)
    ax.tick_params('z', labelleft = False)



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

def visualize_and_save_keypoints_with_connections(frame, keypoints, name, output_dir):  
 
     
    # Get keypoints for the current frame  
    frame_keypoints = keypoints  # As per your data description  

    # Draw keypoints  
    print(frame_keypoints.shape[0])
    for i in range(frame_keypoints.shape[0]):  
        x, y = frame_keypoints[i]  
        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)  # Green circles for keypoints  

    # Draw connections  
    for kpt_a, kpt_b in connections:  
        x1, y1 = frame_keypoints[kpt_a]  
        x2, y2 = frame_keypoints[kpt_b]  

        # Choose color based on connection type  
        if {kpt_a, kpt_b} in [{5, 7}, {7, 9}, {6, 8}, {8, 10}]:  # Arm connections  
            color = (0, 255, 0)  # Green  
        elif {kpt_a, kpt_b} in [{15, 17}, {17, 18}, {15, 19},   
                                {16, 20}, {20, 21}]:  # Leg connections  
            color = (0, 255, 255) # Yellow  
        else:  # Other connections  
            color = (255, 0, 0)  # Blue  

        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)  

    # Save the frame with keypoints and connections visualized  
    frame_filename = os.path.join(output_dir, f'{name}.png')  
    cv2.imwrite(frame_filename, frame)  


def get_pose2D(video_path, detect_path, output_dir):
    import json
    print(video_path)
    for filename in os.listdir(video_path):  
        if filename.endswith('.png'):  
            image_path = os.path.join(video_path, filename)  
            print("===> image_path: ", image_path)
            name = filename[:-4]
            detect_name = os.path.join(detect_path, f"{name}.json")
            with open(detect_name, 'r', encoding='utf-8') as f:  
                data = json.load(f)
            point = data["shapes"][0]["points"]
            lpoint, rpoint = point[0], point[1]
            bbox = [lpoint[0],  lpoint[1], rpoint[0], rpoint[1]]
            frame = cv2.imread(image_path)
            try:
                kpts, scores = gen_frame_kpts_direct_bbox(frame, bbox)
                print("======>", kpts.shape)
                # print(kpts)      
                # frame_filename = f"images/{video_name}_frame_{ii:04d}.png"  
                # frame_path = os.path.join(output_dir, frame_filename)  
                # cv2.imwrite(frame_path, frame)  
                # 将frame编码为PNG格式  
                _, buffer = cv2.imencode('.png', frame)  
                
                # 将PNG格式的图像数据转换为base64编码的字符串  
                img_base64 = base64.b64encode(buffer).decode('utf-8')  

    
                # 准备JSON数据  
                json_data = {  
                    "version": "5.5.0",  
                    "flags": {},  
                    "shapes": [],  
                    "imagePath": filename,  
                    "imageHeight": frame.shape[0],
                    "imageData":   img_base64,
                    "imageWidth": frame.shape[1]  
                }  

                keypoint_names = [  
                    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',  
                    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',  
                    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',  
                    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'  
                ]  

                for kpt_idx, (x, y) in enumerate(kpts):  
                    print(kpt_idx)
                    shape = {  
                        "label": str(kpt_idx),  
                        "points": [[float(x), float(y)]],  
                        "group_id": None,  
                        "description": keypoint_names[kpt_idx],  
                        "shape_type": "point",  
                        "flags": {}  
                    }  
                    json_data["shapes"].append(shape)  

                # 保存JSON文件  
                json_filename = f"keypoint/{name}.json"  
                json_path = os.path.join(output_dir, json_filename)  
                with open(json_path, 'w') as f:  
                    json.dump(json_data, f, indent=2) 
            except:
                continue


            # print(kpts, scores)
            # visualize_and_save_keypoints_with_connections(frame, kpts, name, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default='aaa.mp4', help='input video')
    parser.add_argument('--gpu', type=str, default='0', help='input video')
    parser.add_argument('--fix_z', action='store_true', help='fix z axis')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    file_path = Path("/home/zlt/Documents/SkydivingPose/sample/labelme")
    video_path = file_path / "images"
    detect_path = file_path / "detect"
    output_dir = file_path / "output"
    get_pose2D(video_path, detect_path, output_dir)

