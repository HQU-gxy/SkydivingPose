import os 
import random
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

from app.deploy.gen_kps import gen_video_kpts, gen_frame_kpts, get_3D_kpts
from app.deploy.preprocess import h36m_coco_format, revise_kpts
from app.deploy.sort import Sort
from app.utils.visualize import show_skeleton, show3Dpose

plt.switch_backend('agg')
warnings.filterwarnings('ignore')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sys.path.append(os.getcwd())
from common.utils import *
from common.camera import *
from model.mixste.hot_mixste import Model


import threading


DETECT_WINDOW = 100




def get_pose2D(video_path, output_dir, fix_z):
    # 设置其他参数，初始化模型等
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


    cap = cv2.VideoCapture(0)  # 使用第一个可用摄像头

    CAP_WIDTH, CAP_HEIGHT = 960, 720
    FPS = 60

    cap.set(cv2.CAP_PROP_FPS, FPS)
    # 设置摄像头捕获的宽度和高度
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)


    people_sort = Sort(min_hits=0)
    frame_idx = 0
    kpts_result = []
    scores_result = []
    bboxs_pre, person_scores_pre = None, None
    skeleton_color = [(154,194,182),(123,151,138),(0,208,244),(8,131,229),(18,87,220)] # 选择自己喜欢的颜色
    color = random.choice(skeleton_color)

                        
    cv2.namedWindow('3D Pose Visualization', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('3D Pose Visualization', 800, 600)  # 设置窗口大小为 800x600 像素

    print('\nGenerating 2D pose...')
    while True:
        ret, frame = cap.read()
        # frame = cv2.flip(frame, 1)
        if not ret:
            break

        with torch.no_grad():
            # the first frame of the video should be detected a person
            kpts, scores, bboxs_pre, person_scores_pre = gen_frame_kpts(frame, people_sort, bboxs_pre, person_scores_pre,\
                                                                        num_peroson=1, gen_output=True)
            kpts_result.append(kpts)
            scores_result.append(scores)

        frame_with_kps = show_skeleton(frame, kpts, scores, color=color)

        keypoints = np.array(kpts_result)
        scores = np.array(scores_result)

        keypoints = keypoints.transpose(1, 0, 2, 3)  # (T, M, N, 2) --> (M, T, N, 2)
        scores = scores.transpose(1, 0, 2)  # (T, M, N) --> (M, T, N)
        keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
        re_kpts = revise_kpts(keypoints, scores, valid_frames)

        print(keypoints.shape)
        # 每隔DETECT_WINDOW帧检测一次
        if (frame_idx+1) % DETECT_WINDOW == 0:
            # print('\nGenerating 3D pose...')
            n_chunks = DETECT_WINDOW // args.frames + 1
            offset = (n_chunks * args.frames - DETECT_WINDOW) // 2
            output_3d_all = None
            keypoints_window = keypoints[:, keypoints.shape[1]-DETECT_WINDOW:, :, :]
            # print("keypoinkeypoints_windowts", keypoints_window.shape)
            frame_sum = 0
            for i in range(n_chunks):
                post_out, output_3d_all, low_index, high_index = get_3D_kpts(i, args, model, offset, DETECT_WINDOW,\
                                                    keypoints_window, output_3d_all, CAP_WIDTH, CAP_HEIGHT)
                for j in range(low_index, high_index):
                    jj = j - frame_sum
                    if i == 0 and j == 0:
                        pass
                    fig = plt.figure(figsize=(9.6, 5.4))
                    gs = gridspec.GridSpec(1, 1)
                    gs.update(wspace=-0.00, hspace=0.05) 
                    ax = plt.subplot(gs[0], projection='3d')
                    post_out[jj, :, 2] -= np.min(post_out[jj, :, 2])
                    show3Dpose(post_out[jj], ax, fix_z)
                    
                    # 将 Matplotlib 生成的图像转换为 OpenCV 格式
                    fig.canvas.draw()
                    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    
                    # 显示图像在 OpenCV 窗口中
                    cv2.imshow('3D Pose Visualization', frame)
                    # 等待一段时间自动更新到下一帧
                    cv2.waitKey(1)  # 设置延迟时间为 100 毫秒 (自动更新)

                frame_sum = high_index

        # 显示实时视频流
        cv2.imshow('Real-time Video Stream', frame_with_kps)

        frame_idx += 1

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # 按下ESC键退出
            break

    cap.release()
    cv2.destroyAllWindows()
    print('Generating 2D pose successfully!')

    output_dir_2D = output_dir + 'input_2D/'
    os.makedirs(output_dir, exist_ok=True)

    output_npz = output_dir_2D + 'input_keypoints_2d.npz'
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
    parser.add_argument('--video', type=str, default='camera.mp4', help='input video')
    parser.add_argument('--gpu', type=str, default='0', help='input video')
    parser.add_argument('--fix_z', action='store_true', help='fix z axis')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    video_path = './sample/' + args.video
    video_name = video_path.split('/')[-1].split('.')[0]
    output_dir = './output/' + video_name + '/'

    get_pose2D(video_path, output_dir, args.fix_z)
    # get_pose3D(video_path, output_dir, args.fix_z)
    # img2video(video_path, output_dir)


