import os
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from model.SGraFormer import sgraformer  # 确保这个模型类已经正确导入
from common.utils import *
import torch.utils.data as data
from common.opt import opts
from torch.utils.data import DataLoader
import torchvision.transforms as transforms  
# 假设 normalize_screen_coordinates 函数已经定义
def normalize_screen_coordinates(X, w, h):
    return X / w * 2 - [1, h / w]

class Fusion(data.Dataset):
    def __init__(self, opt, npz_paths, start_frame=200, num_frames=27, num_cameras=4, test_aug=True):
        self.start_frame = start_frame
        self.num_frames = num_frames
        self.test_aug = test_aug  
        self.hop1 = torch.tensor([[0, 1,	0,	0,	1,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [1,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	1,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [1,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	0,	1,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [1,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	0,	1,	0,	0,	1,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	1,	0,	0,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	0,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	1,	0],
                            [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	1],
                            [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0]])

        self.hop2 = torch.tensor([[0,	0,	1,	0,	0,	1,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	1,	1,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	1,	0,	0,	0,	0,	1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	1,	0,	0,	1,	0,	0,	0,	0,	1,	0,	1,	0,	0,	1,	0,	0],
                            [1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	0,	0,	1,	0],
                            [0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	1,	0,	0,	1,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	0,	0,	0,	1,	1,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	0,	0,	0,	0,	0,	0,	1],
                            [0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0]])

        self.hop3 = torch.tensor([[0,	0,	0,	1,	0,	0,	1,	0,	0,	1,	0,	1,	0,	0,	1,	0,	0],
                            [0,	0,	0,	0,	0,	1,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	0,	1,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	1,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	1,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	1,	0,	0,	1,	0,	0,	0,	0,	1,	0,	1,	0,	0,	1,	0],
                            [0,	1,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	1],
                            [1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	1,	0],
                            [0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	1,	0,	0,	1,	0,	0],
                            [1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	1,	0],
                            [0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	0,	0,	0,	0,	1,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0],
                            [1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	0,	0,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	0,	1,	0,	0,	0,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0]])

        self.hop4 = torch.tensor([[0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	0,	0,	1,	0],
                            [0,	0,	0,	0,	0,	0,	1,	0,	0,	1,	0,	1,	0,	0,	1,	0,	0],
                            [0,	0,	0,	0,	0,	1,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	0,	1,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	1,	0,	0,	0,	0,	0,	1,	0,	1,	0,	0,	1,	0,	0],
                            [0,	0,	1,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	1,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	1,	0,	0,	1,	0,	0,	0,	0,	0,	0,	1,	0,	0,	1],
                            [0,	0,	1,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	1,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	1],
                            [1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	1,	0],
                            [0,	1,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1],
                            [1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	1,	0],
                            [0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	0,	0,	0,	0,	1,	0,	0],
                            [0,	1,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0],
                            [1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	0,	0,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	0,	1,	0,	0,	0,	0,	0]])
        self.data = self.load_data(npz_paths, num_frames, num_cameras)
        self.opt = opt

    def augment_data(self, data):  
        # return data
        # 创建数据增强组合  
        augmentations = transforms.Compose([  
            transforms.RandomHorizontalFlip(p=0.5),  
            transforms.RandomRotation(degrees=15),  
        ])  
        augmented_data = [augmentations(frame).numpy() for frame in data]  
        return torch.tensor(augmented_data, dtype=torch.float32) 
    
    def load_data(self, npz_paths, num_frames, num_cameras):
        camera_data = []
        print("\n===============>", num_frames)
        for j in range(num_cameras):
            loaded = np.load(npz_paths[j])['reconstruction'][0]
            frames = loaded.shape[0]
            if frames > num_frames:
                loaded = loaded[self.start_frame:self.start_frame+num_frames, :, :]  # Trim to fixed number of frames
            elif frames < num_frames:
                repeat_times = (num_frames + frames - 1) // frames
                loaded = np.tile(loaded, (repeat_times, 1, 1))[self.start_frame:self.start_frame+num_frames, :, :]  # Extend frames to match
            
            # 假设图像的宽度和高度
            w, h = 1080, 1920   # 你可以根据实际情况修改

            # 归一化关键点到 [-1, 1]
            loaded[..., :2] = normalize_screen_coordinates(loaded[..., :2], w, h)
            
            camera_data.append(torch.tensor(loaded, dtype=torch.float32))
        
        # Stack all camera data with new dimension for cameras and time at the correct positions
        combined_data = torch.stack(camera_data, dim=1)  # Camera dimension is second
        # Add a fake dimension for data augmentation (e.g., original and augmented)
        combined_data = combined_data.unsqueeze(0).repeat(2, 1, 1, 1, 1)  # Repeat data for augmentation
        return combined_data  # No need to add a batch dimension here
    
    def hop_normalize(self, x1, x2, x3, x4):
        # Normalize hops and add a batch dimension
        x1 = x1 / torch.sum(x1, dim=1, keepdim=True)
        x2 = x2 / torch.sum(x2, dim=1, keepdim=True)
        x3 = x3 / torch.sum(x3, dim=1, keepdim=True)
        x4 = x4 / torch.sum(x4, dim=1, keepdim=True)
        hops = torch.stack((x1, x2, x3, x4), dim=0)
        return hops
    
    def __getitem__(self, index):  
        input_data = self.data[index]  
        if self.test_aug:  
            augmented_data = self.augment_data(input_data)  
            input_data = torch.stack((input_data, augmented_data), dim=0)  
        
        hops = self.hop_normalize(self.hop1, self.hop2, self.hop3, self.hop4)  
        return input_data, hops  

    def __len__(self):
        return len(self.data)
    

def load_and_prepare_data(npz_paths, num_frames=27):
    """加载并准备数据以符合模型的输入要求."""
    all_cameras_data = []
    
    for path in npz_paths:
        data = np.load(path)['reconstruction'][0]  # 加载数据
        if data.shape[0] > num_frames:
            # 如果帧数超过24，选取前24帧
            data = data[:num_frames]
        elif data.shape[0] < num_frames:
            # 如果帧数不足24，进行填充
            repeat_times = num_frames // data.shape[0] + 1
            data = np.tile(data, (repeat_times, 1, 1))[:num_frames]
        
        all_cameras_data.append(data)
    
    batch_data = np.stack(all_cameras_data, axis=0)  # shape: (4, 24, 17, 2)
    return torch.tensor(batch_data, dtype=torch.float32)  # 转换为torch tensor

def visualize_skeletons(input_2D, output_3D, gt_3D, idx=0, output_dir='./output'):  
    # Ensure the tensors are on the CPU and convert them to numpy arrays  
    input_2D = input_2D.cpu().numpy()  
    output_3D = output_3D.cpu().numpy()  
    gt_3D = gt_3D.cpu().numpy()  

    # print("====> input_2D: ", input_2D[-1])
    # Get the first action and first sample from the batch  
    input_sample = input_2D[idx, 0]  
    output_sample = output_3D[idx, 0]  
    gt_3D_sample = gt_3D[idx, 0]  

    print(f'\ninput_sample shape: {input_sample.shape}')  
    print(f'output_sample shape: {output_sample.shape}')  

    fig = plt.figure(figsize=(25, 5))  

    # Define the connections (bones) between joints  
    bones = [  
        (0, 1), (1, 2), (2, 3),  # Left leg  
        (0, 4), (4, 5), (5, 6),  # Right leg  
        (0, 7), (7, 8), (8, 9), (9, 10),  # Spine  
        (7, 11), (11, 12), (12, 13),  # Right arm  
        (7, 14), (14, 15), (15, 16)   # Left arm  
    ]  

    # Colors for different parts  
    bone_colors = {  
        "leg": 'green',  
        "spine": 'blue',  
        "arm": 'red'  
    }  

    # Function to get bone color based on index  
    def get_bone_color(start, end):  
        if (start in [1, 2, 3] or end in [1, 2, 3] or   
            start in [4, 5, 6] or end in [4, 5, 6]):  
            return bone_colors["leg"]  
        elif start in [7, 8, 9, 10] or end in [7, 8, 9, 10]:  
            return bone_colors["spine"]  
        else:  
            return bone_colors["arm"]  

    # Plotting 2D skeletons from different angles  
    for i in range(4):  
        ax = fig.add_subplot(1, 7, i + 1)  
        ax.set_title(f'2D angle {i+1}')  
        ax.scatter(input_sample[i, :, 0], input_sample[i, :, 1], color='blue')  

        # Draw the bones  
        for start, end in bones:  
            bone_color = get_bone_color(start, end)  
            ax.plot([input_sample[i, start, 0], input_sample[i, end, 0]],  
                    [input_sample[i, start, 1], input_sample[i, end, 1]], color=bone_color)  

        ax.set_xlabel('X')  
        ax.set_ylabel('Y')  
        ax.set_xlim(np.min(input_sample[:, :, 0]) - 1, np.max(input_sample[:, :, 0]) + 1)  
        ax.set_ylim(np.min(input_sample[:, :, 1]) - 1, np.max(input_sample[:, :, 1]) + 1)  
        ax.grid()  

    # Plotting predicted 3D skeleton  
    ax = fig.add_subplot(1, 7, 5, projection='3d')  
    ax.set_title('3D Predicted Skeleton')  
    ax.scatter(output_sample[:, 0], output_sample[:, 1], output_sample[:, 2], color='red', label='Predicted')  

    # Draw the bones in 3D for output_sample  
    for start, end in bones:  
        bone_color = get_bone_color(start, end)  
        ax.plot([output_sample[start, 0], output_sample[end, 0]],  
                [output_sample[start, 1], output_sample[end, 1]],  
                [output_sample[start, 2], output_sample[end, 2]], color=bone_color)  

    ax.set_xlabel('X')  
    ax.set_ylabel('Y')  
    ax.set_zlabel('Z')  
    ax.set_xlim(np.min(output_sample[:, 0]) - 1, np.max(output_sample[:, 0]) + 1)  
    ax.set_ylim(np.min(output_sample[:, 1]) - 1, np.max(output_sample[:, 1]) + 1)  
    ax.set_zlim(np.min(output_sample[:, 2]) - 1, np.max(output_sample[:, 2]) + 1)  
    ax.legend()  

    # Plotting ground truth 3D skeleton  
    ax = fig.add_subplot(1, 7, 6, projection='3d')  
    ax.set_title('3D Ground Truth Skeleton')  
    ax.scatter(gt_3D_sample[:, 0], gt_3D_sample[:, 1], gt_3D_sample[:, 2], color='blue', label='Ground Truth')  

    # Draw the bones in 3D for gt_3D_sample  
    for start, end in bones:  
        bone_color = get_bone_color(start, end)  
        ax.plot([gt_3D_sample[start, 0], gt_3D_sample[end, 0]],  
                [gt_3D_sample[start, 1], gt_3D_sample[end, 1]],  
                [gt_3D_sample[start, 2], gt_3D_sample[end, 2]], color=bone_color, linestyle='--')  

    ax.set_xlabel('X')  
    ax.set_ylabel('Y')  
    ax.set_zlabel('Z')  
    ax.set_xlim(np.min(gt_3D_sample[:, 0]) - 1, np.max(gt_3D_sample[:, 0]) + 1)  
    ax.set_ylim(np.min(gt_3D_sample[:, 1]) - 1, np.max(gt_3D_sample[:, 1]) + 1)  
    ax.set_zlim(np.min(gt_3D_sample[:, 2]) - 1, np.max(gt_3D_sample[:, 2]) + 1)  
    ax.legend()  

    plt.grid()  

    # Save the figure  
    plt.tight_layout()  
    plt.savefig(f'{output_dir}/skeletons_visualization.png')  
    plt.show()  
    
def input_augmentation(input_2D, hops, model):
    input_2D_non_flip = input_2D[:, 0]
    output_3D_non_flip = model(input_2D_non_flip, hops)

    return input_2D_non_flip, output_3D_non_flip

def predict_3d_skeletons(opt, model, input_data, device):
    """使用模型进行3D骨骼预测."""
    model.eval()  # 设置模型为评估模式

    TQDM = tqdm(enumerate(input_data), total=len(input_data), ncols=100)
    for i, data in TQDM:
        input_2D, hops = data

        input_2D = input_2D.to(device)
        hops = hops.to(device)

        print("=====> input_2D: ", input_2D.shape)
        print("=====> hops: ", hops.shape)
        print("=====> input_2D: ", input_2D.shape)
        print("=====> hops: ", hops.shape)

        with torch.no_grad():  # 关闭梯度计算
            input_2D, output_3D = input_augmentation(input_2D, hops, model)  # 进行预测
        if output_3D.shape[1] != 1:
            output_3D = output_3D[:, opt.pad].unsqueeze(1)
        output_3D[:, :, 1:, :] -= output_3D[:, :, :1, :]
        output_3D[:, :, 0, :] = 0

        visualize_skeletons(input_2D, output_3D, output_3D)  

    return output_3D

def main():
    npz_paths = [
        '/home/zlt/Documents/SkydivingPose/output/test-001/001_keypoints_2d.npz',
        '/home/zlt/Documents/SkydivingPose/output/test-001/002_keypoints_2d.npz',
        '/home/zlt/Documents/SkydivingPose/output/test-001/003_keypoints_2d.npz',
        '/home/zlt/Documents/SkydivingPose/output/test-001/004_keypoints_2d.npz',
    ]
    
    # 加载并准备数据
    # input_data = load_and_prepare_data(npz_paths)
    # print("=====> input_data", input_data.shape)

    # # 增加一个维度，作为batch维度
    # input_data = input_data.unsqueeze(0)  # 在第0位增加一个维度
    # print("Updated Input Data Shape:", input_data.shape)  # 显示更新后的数据形状
    opt = opts().parse()
    root_path = opt.root_path
    opt.manualSeed = 1

    dataset = Fusion(opt, npz_paths)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)



    # 模型初始化和配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = sgraformer(num_frame=27, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                       num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0.1)
    model = model.to(device)
        # 定义一个函数来去除 'module.' 前缀
    def remove_module_prefix(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k  # 去除 `module.`
            new_state_dict[name] = v
        return new_state_dict

    model_dict = model.state_dict()
   
    model_path = '/home/zlt/Documents/SkydivingPose/checkpoint/SGraFormer/model_28_2733.pth'
    pre_dict = torch.load(model_path)
    # print("=====> pre_dict:", pre_dict.keys())
    # 去除 'module.' 前缀
    state_dict = remove_module_prefix(pre_dict)
    # print("=====> state_dict:", state_dict.keys())
    # 只保留在模型字典中的键值对
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict.keys()}
    # 更新模型字典
    model_dict.update(state_dict)
    # 加载更新后的模型字典
    model.load_state_dict(model_dict)

    # 假设你已经有预训练模型
    # model.load_state_dict(torch.load('path_to_pretrained_model.pth'))

    # 进行3D预测
    output_3D = predict_3d_skeletons(opt, model, data_loader, device)
    print("Predicted 3D Skeletons Shape:", output_3D.shape)

if __name__ == '__main__':
    main()
