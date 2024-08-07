import os
import sys

import os.path as osp
import argparse
from pathlib import Path
import time
from typing import Literal
import numpy as np
from tqdm import tqdm
import json
import torch
import torch.backends.cudnn as cudnn
import cv2
import copy

from ultralytics import YOLO
from app.deploy.sort import Sort
from app.utils.utilitys import plot_keypoint, PreProcess, write, load_json
from app.utils.camera import normalize_screen_coordinates, camera_to_world
import time
from mmcv.image import imread

from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples


DeviceType = Literal["cuda", "cpu", "xpu"]
DEVICE: DeviceType = "cuda"

DETECT_TYPE  = Literal["yolov8n", "yolov8s", "yolov8m", "yolov8l"]
DETECT_CONFIG: DETECT_TYPE = "yolov8l"
DETECT_PATH = Path("checkpoint/yolo")
DETECT_MODEL = YOLO("{}.yaml".format(DETECT_CONFIG))
DETECT_MODEL = YOLO("{}/{}.pt".format(DETECT_PATH, DETECT_CONFIG))

THRED_SCORE = 0.5

POSE_TYPE  = Literal["simple3Dbaseline_h36m-f0ad73a4_20210419"]
POSE_CONFIG: DETECT_TYPE = "simple3Dbaseline_h36m-f0ad73a4_20210419"
POSE_CFG_DICT = {
    "simple3Dbaseline_h36m-f0ad73a4_20210419": "configs/pose_3D/motionbert_dstformer-243frm_8xb32-240e_h36m-original.py",
}
# configs/crowded/rtmpose-m_8xb64-210e_crowdpose-256x192.py
# POSE_CFG = Path("configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py")
POSE_CFG = Path(POSE_CFG_DICT[POSE_CONFIG])
POSE_CKPT = Path("checkpoint/pose_3D/{}.pth".format(POSE_CONFIG))

# 使用初始化接口构建模型
POSE_MODEL = init_model(str(POSE_CFG), str(POSE_CKPT), device=DEVICE)
# 初始化可视化器
VISUALIZER = VISUALIZERS.build(POSE_MODEL.cfg.visualizer)
# 设置数据集元信息
VISUALIZER.set_dataset_meta(POSE_MODEL.dataset_meta)

IMAGE_SIZE_DICT = {
    "yolov8n": [640, 640],
    "yolov8s": [640, 640],
    "yolov8l": [640, 640]

}
IMAGE_SIZE = IMAGE_SIZE_DICT[DETECT_CONFIG]


def gen_frame_3D_kpts(frame, people_sort, bboxs_pre, person_scores_pre, num_peroson=1, gen_output=False):
    kpts = np.zeros((num_peroson, 17, 2), dtype=np.float32)
    scores = np.zeros((num_peroson, 17), dtype=np.float32)
    frame_bbox = None
    frame_person_score = None

    # 只需要检测人
    results = DETECT_MODEL(frame, conf=THRED_SCORE, device=DEVICE, classes=[0], verbose=False)
    boxes = results[0].boxes
    bboxs, person_scores = boxes.xyxy, boxes.conf

    if bboxs is None or not bboxs.any():
        # print('No person detected!')
        if bboxs_pre is None or person_scores_pre is None:
            return kpts, scores, frame_bbox, frame_person_score
        bboxs = bboxs_pre
        person_scores = person_scores

    frame_bbox, frame_person_score = bboxs, person_scores

    # Using Sort to track people
    bboxs = np.array(bboxs.cpu())

    people_track = people_sort.update(bboxs)


    # Track the first two people in the video and remove the ID
    if people_track.shape[0] == 1:
        people_track_ = people_track[-1, :-1].reshape(1, 4)
    elif people_track.shape[0] >= 2:
        people_track_ = people_track[-num_peroson:, :-1].reshape(num_peroson, 4)
        people_track_ = people_track_[::-1]
    else:
        return kpts, scores, frame_bbox, frame_person_score

    track_bboxs = []
    person_positions = []

    for bbox in people_track_:
        bbox = [round(i, 2) for i in list(bbox)]
        track_bboxs.append(bbox)
        x1, y1, x2, y2 = map(int, bbox)
        person_positions.append((x1, y1, x2, y2))
        person_img = frame[y1:y2, x1:x2]


    for idx, bbox in enumerate(track_bboxs[:num_peroson]):
        x1, y1, x2, y2 = map(int, bbox)
        person_img = frame[y1:y2, x1:x2]
        x1_orig, y1_orig, x2_orig, y2_orig = person_positions[idx]
        person_results = inference_topdown(POSE_MODEL, person_img)
        pred_instances = person_results[0].pred_instances
        pred_kps, pred_socre = pred_instances.keypoints[0], pred_instances.keypoint_scores[0]
        kpts[idx] = pred_kps
        scores[idx] = pred_socre.squeeze()

        kpts[idx, :, 0] += x1_orig
        kpts[idx, :, 1] += y1_orig

    return kpts, scores, bboxs_pre, person_scores_pre


def get_3D_kpts(i, args, model, offset, DETECT_WINDOW,\
                keypoints, output_3d_all, CAP_WIDTH, CAP_HEIGHT):

    ## input frames
    start_index = i*args.frames - offset
    end_index = (i+1)*args.frames - offset
    low_index = max(start_index, 0)
    high_index = min(end_index, DETECT_WINDOW)
    pad_left = low_index - start_index
    pad_right = end_index - high_index

    if pad_left != 0 or pad_right != 0:
        input_2D_no = np.pad(keypoints[0][low_index:high_index], ((pad_left, pad_right), (0, 0), (0, 0)), 'edge')
    else:
        input_2D_no = keypoints[0][low_index:high_index]
    joints_left =  [4, 5, 6, 11, 12, 13]
    joints_right = [1, 2, 3, 14, 15, 16]
    input_2D = normalize_screen_coordinates(input_2D_no, w=CAP_WIDTH, h=CAP_HEIGHT)  

    input_2D_aug = copy.deepcopy(input_2D)
    input_2D_aug[ :, :, 0] *= -1
    input_2D_aug[ :, joints_left + joints_right] = input_2D_aug[ :, joints_right + joints_left]
    input_2D = np.concatenate((np.expand_dims(input_2D, axis=0), np.expand_dims(input_2D_aug, axis=0)), 0)
    
    input_2D = input_2D[np.newaxis, :, :, :, :]

    input_2D = torch.from_numpy(input_2D.astype('float32')).cuda()

    N = input_2D.size(0)

    ## estimation
    with torch.no_grad():
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

    rot =  [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
    rot = np.array(rot, dtype='float32')
    post_out = camera_to_world(post_out, R=rot, t=0)

    return post_out, output_3d_all, low_index, high_index


def gen_video_3D_kpts(video, num_peroson=1, gen_output=False):
    cap = cv2.VideoCapture(video)
    people_sort = Sort(min_hits=0)

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    kpts_result = []
    scores_result = []

    total_detection_time = 0
    total_tracking_time = 0
    total_pose_estimation_time = 0


    pre_kps, pre_score = None, None

    for ii in tqdm(range(video_length)-1):
    # for ii in tqdm(range(70)):
        ret, frame = cap.read()

        if not ret:
            continue

        # 只需要检测人
        start_detection_time = time.time()
        results = DETECT_MODEL(frame, conf=THRED_SCORE, device=DEVICE, classes=[0], verbose=False)
        detection_time = time.time() - start_detection_time
        total_detection_time += detection_time

        boxes = results[0].boxes
        bboxs, scores = boxes.xyxy, boxes.conf

        if bboxs is None or not bboxs.any():
            # print('No person detected!')
            try:
                bboxs = bboxs_pre
                scores = scores_pre
            except:
                continue

        else:
            bboxs_pre = copy.deepcopy(bboxs)
            scores_pre = copy.deepcopy(scores)

        # Using Sort to track people
        start_tracking_time = time.time()
        bboxs = np.array(bboxs.cpu())
        people_track = people_sort.update(bboxs)
        tracking_time = time.time() - start_tracking_time
        total_tracking_time += tracking_time

        # Track the first two people in the video and remove the ID
        if people_track.shape[0] == 1:
            people_track_ = people_track[-1, :-1].reshape(1, 4)
        elif people_track.shape[0] >= 2:
            people_track_ = people_track[-num_peroson:, :-1].reshape(num_peroson, 4)
            people_track_ = people_track_[::-1]
        else:
            continue

        track_bboxs = []
        # 在获取人物图像部分时，记录下检测框在整个视频帧中的位置
        person_positions = []

        for bbox in people_track_:
            bbox = [round(i, 2) for i in list(bbox)]
            track_bboxs.append(bbox)
            x1, y1, x2, y2 = map(int, bbox)
            # 记录检测框在整个视频帧中的位置
            person_positions.append((x1, y1, x2, y2))
            person_img = frame[y1:y2, x1:x2]  # 获取边界框内的图像部分

        kpts = np.zeros((num_peroson, 17, 2), dtype=np.float32)
        scores = np.zeros((num_peroson, 17), dtype=np.float32)

        # 对每个跟踪到的人进行识别
        for idx, bbox in enumerate(track_bboxs[:num_peroson]):
            x1, y1, x2, y2 = map(int, bbox)
            person_img = frame[y1:y2, x1:x2]  # 获取边界框内的图像部分
            # 获取人物图像部分在整个视频帧中的位置信息
            x1_orig, y1_orig, x2_orig, y2_orig = person_positions[idx]
            # # 计算关键点位置在整个视频帧中的绝对位置
            # kpts[idx, :, 0] += x1_orig
            # kpts[idx, :, 1] += y1_orig
            # 使用识别模型识别
            start_pose_estimation_time = time.time()
            # print("==============================")
            try:
                person_results = inference_topdown(POSE_MODEL, person_img)
                print("======> person_results", person_results)

                pose_estimation_time = time.time() - start_pose_estimation_time
                total_pose_estimation_time += pose_estimation_time

                pred_instances = person_results[0].pred_instances
                pred_kps, pred_socre = pred_instances.keypoints[0], pred_instances.keypoint_scores[0]
                print("======>", pred_socre)
                kpts[idx] = pred_kps
                
                
                scores[idx] = pred_socre.squeeze()
            except Exception as e:
                print("No Pose Detect!\n=======> ", e)
                kpts, scores = pre_kps, pre_score 
                
        pre_kps, pre_score = kpts, scores


        kpts_result.append(kpts)
        print("============kpts", len(kpts_result))

        scores_result.append(scores)

    keypoints = np.array(kpts_result)
    scores = np.array(scores_result)

    keypoints = keypoints.transpose(1, 0, 2, 3)  # (T, M, N, 2) --> (M, T, N, 2)
    scores = scores.transpose(1, 0, 2)  # (T, M, N) --> (M, T, N)

    print(f"Total detection time: {total_detection_time:.2f} seconds")
    print(f"Total tracking time: {total_tracking_time:.2f} seconds")
    print(f"Total pose estimation time: {total_pose_estimation_time:.2f} seconds")

    return keypoints, scores

