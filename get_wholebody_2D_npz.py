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
from app.utils.wholebody_skeleton import extract_keypoints, extract_keypoints_with_foot
from app.deploy.smooth_keypoints import smooth_keypoints_savgol

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

def visualize_and_save_keypoints_with_connections(keypoints, video_path, output_dir):  
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

        # Draw keypoints  
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
    coco_17 = extract_keypoints(keypoints)  
    coco_22 = extract_keypoints_with_foot(keypoints)  

        # Example usage  
    smoothed_keypoints_22 = smooth_keypoints_savgol(coco_22)  
    smoothed_keypoints_17 = smooth_keypoints_savgol(coco_17)  

    print("=====coco_22", smoothed_keypoints_22.shape)

    visualize_and_save_keypoints_with_connections(coco_22, video_path, f"{output_dir}/2D")
    keypoints, scores, valid_frames = h36m_coco_format(smoothed_keypoints_17, scores)

    re_kpts = revise_kpts(keypoints, scores, valid_frames)
    print('Generating 2D pose successfully!')

    output_dir += 'input_2D/'
    os.makedirs(output_dir, exist_ok=True)

    output_npz = output_dir + 'input_keypoints_2d.npz'
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


def get_pose3D(video_path, output_dir, fix_z):
    args, _ = argparse.ArgumentParser().parse_known_args()
    args.layers, args.channel, args.d_hid, args.frames = 8, 512, 1024, 243
    args.token_num, args.layer_index = 81, 3
    args.pad = (args.frames - 1) // 2
    args.previous_dir = 'checkpoint/pretrained/hot_mixste'
    args.n_joints, args.out_joints = 17, 17

    ## Reload 
    model = Model(args).cuda()

    model_dict = model.state_dict()
    # Put the pretrained model in 'checkpoint/pretrained/hot_mixste'
    model_path = sorted(glob.glob(os.path.join(args.previous_dir, '*.pth')))[0]

    pre_dict = torch.load(model_path)
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in pre_dict.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    model.eval()

    # 记录 get_pose3D 执行时间
    start_time_3d = time.time()
    ## input
    keypoints = np.load(output_dir + 'input_2D/input_keypoints_2d.npz', allow_pickle=True)['reconstruction']
    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("============video_length: ", video_length)
    n_chunks = video_length // args.frames + 1
    offset = (n_chunks * args.frames - video_length) // 2
    ret, img = cap.read()
    img_size = img.shape

    ## 3D
    print('\nGenerating 3D pose...')
    frame_sum = 0
    for i in tqdm(range(n_chunks)):

        ## input frames
        start_index = i*args.frames - offset
        end_index = (i+1)*args.frames - offset

        low_index = max(start_index, 0)
        high_index = min(end_index, video_length)
        pad_left = low_index - start_index
        pad_right = end_index - high_index

        print("==================", keypoints[0].shape)
        if pad_left != 0 or pad_right != 0:
            input_2D_no = np.pad(keypoints[0][low_index:high_index], ((pad_left, pad_right), (0, 0), (0, 0)), 'edge')
        else:
            input_2D_no = keypoints[0][low_index:high_index]
        
        joints_left =  [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]
        # print(keypoints.shape)
        # print("========> input_2D_no", input_2D_no.shape)
        input_2D = normalize_screen_coordinates(input_2D_no, w=img_size[1], h=img_size[0])  

        input_2D_aug = copy.deepcopy(input_2D)
        input_2D_aug[ :, :, 0] *= -1
        input_2D_aug[ :, joints_left + joints_right] = input_2D_aug[ :, joints_right + joints_left]
        input_2D = np.concatenate((np.expand_dims(input_2D, axis=0), np.expand_dims(input_2D_aug, axis=0)), 0)
        
        input_2D = input_2D[np.newaxis, :, :, :, :]

        input_2D = torch.from_numpy(input_2D.astype('float32')).cuda()

        N = input_2D.size(0)

        ## estimation
        with torch.no_grad():
            # print("input_2D[:, 0]", input_2D[:, 0].shape)
            output_3D_non_flip = model(input_2D[:, 0])
            output_3D_flip     = model(input_2D[:, 1])

        output_3D_flip[:, :, :, 0] *= -1
        output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :] 

        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        if pad_left != 0 and pad_right != 0:
            output_3D = output_3D[:, pad_left:-pad_right]
            input_2D_no = input_2D_no[pad_left:-pad_right]
        elif pad_left != 0:
            output_3D = output_3D[:, pad_left:]
            input_2D_no = input_2D_no[pad_left:]
        elif pad_right != 0:
            output_3D = output_3D[:, :-pad_right]
            input_2D_no = input_2D_no[:-pad_right]

        output_3D[:, :, 0, :] = 0
        post_out = output_3D[0].cpu().detach().numpy()

        if i == 0:
            output_3d_all = post_out
        else:
            output_3d_all = np.concatenate([output_3d_all, post_out], axis = 0)

        # print(output_3d_all.shape)
        # print(output_3d_all)
        ## h36m_cameras_extrinsic_params in common/camera.py
        # https://github.com/facebookresearch/VideoPose3D/blob/main/common/custom_dataset.py#L23
        rot =  [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
        rot = np.array(rot, dtype='float32')
        post_out = camera_to_world(post_out, R=rot, t=0)

        # 2D
        for j in range(low_index, high_index):
            jj = j - frame_sum
            if i == 0 and j == 0:
                pass
            else:
                ret, img = cap.read()
                img_size = img.shape

            # print(input_2D_no[jj])
            # print(input_2D_no[jj].shape)
            image = show2Dpose(input_2D_no[jj], copy.deepcopy(img))

            output_dir_2D = output_dir +'pose2D/'
            os.makedirs(output_dir_2D, exist_ok=True)
            cv2.imwrite(output_dir_2D + str(('%04d'% j)) + '_2D.png', image)

            ## 3D
            fig = plt.figure(figsize=(9.6, 5.4))
            gs = gridspec.GridSpec(1, 1)
            gs.update(wspace=-0.00, hspace=0.05) 
            ax = plt.subplot(gs[0], projection='3d')

            post_out[jj, :, 2] -= np.min(post_out[jj, :, 2])
            show3Dpose(post_out[jj], ax, fix_z)

            output_dir_3D = output_dir +'pose3D/'
            os.makedirs(output_dir_3D, exist_ok=True)
            plt.savefig(output_dir_3D + str(('%04d'% j)) + '_3D.png', dpi=200, format='png', bbox_inches = 'tight')

        frame_sum = high_index
    

    ## save 3D keypoints
    os.makedirs(output_dir + 'output_3D/', exist_ok=True)
    output_npz = output_dir + 'output_3D/' + 'output_keypoints_3d.npz'
    np.savez_compressed(output_npz, reconstruction=output_3d_all)

    print('Generating 3D pose successfully!')
    end_time_3d = time.time()
    print(f"get_pose3D took {end_time_3d - start_time_3d:.2f} seconds")
    ## all
    image_dir = 'results/' 
    image_2d_dir = sorted(glob.glob(os.path.join(output_dir_2D, '*.png')))
    image_3d_dir = sorted(glob.glob(os.path.join(output_dir_3D, '*.png')))

    print('\nGenerating demo...')
    for i in tqdm(range(len(image_2d_dir))):
        image_2d = plt.imread(image_2d_dir[i])
        image_3d = plt.imread(image_3d_dir[i])

        ## crop
        edge = (image_2d.shape[1] - image_2d.shape[0]) // 2 - 1
        # image_2d = image_2d[:, edge:image_2d.shape[1] - edge]
        edge_1 = 10
        image_2d = image_2d[edge_1:image_2d.shape[0] - edge_1, edge + edge_1:image_2d.shape[1] - edge - edge_1]

        edge = 130
        image_3d = image_3d[edge:image_3d.shape[0] - edge, edge:image_3d.shape[1] - edge]

        ## show
        font_size = 12
        fig = plt.figure(figsize=(9.6, 5.4))
        ax = plt.subplot(121)
        showimage(ax, image_2d)
        ax.set_title("Input", fontsize = font_size)

        ax = plt.subplot(122)
        showimage(ax, image_3d)
        ax.set_title("Reconstruction", fontsize = font_size)

        ## save
        output_dir_pose = output_dir +'pose/'
        os.makedirs(output_dir_pose, exist_ok=True)
        plt.savefig(output_dir_pose + str(('%04d'% i)) + '_pose.png', dpi=200, bbox_inches = 'tight')
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default='aaa.mp4', help='input video')
    parser.add_argument('--gpu', type=str, default='0', help='input video')
    parser.add_argument('--fix_z', action='store_true', help='fix z axis')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # video_path = './sample/' + args.video
    video_path = args.video_path

    video_name = video_path.split('/')[-1].split('.')[0]
    # output_dir = './output/' + video_name + '/'
    output_dir = f'output/camera/{video_name}/'  

    get_pose2D(video_path, output_dir, video_name)

    get_pose3D(video_path, output_dir, args.fix_z)
    img2video(video_path, output_dir)
