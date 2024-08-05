import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio


def show3Dpose_corrected(vals, ax, fix_z=True):
    ax.clear()
    ax.view_init(elev=15., azim=70)
    colors = [(138/255, 201/255, 38/255),
              (255/255, 202/255, 58/255),
              (25/255, 130/255, 196/255)]
    I = np.array([0, 0, 1, 4, 2, 5, 0, 7, 8, 8, 14, 15, 11, 12, 8, 9])
    J = np.array([1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])
    LR = [3, 3, 3, 3, 3, 3, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1]

    # Adjusting to match matplotlib's default coordinate system: x, y, z
    for i in np.arange(len(I)):
        # Extracting values and rearranging coordinates to x, y, z from the original data order
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in (0, 1, 2)]
        ax.plot(x, z, -y, lw=2, color=colors[LR[i]-1])  # Adjusting z and y for display in matplotlib

    RADIUS = 0.72
    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS - yroot, RADIUS - yroot])

    ax.set_aspect('equal')
    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)
    ax.tick_params(axis='x', labelbottom=False)
    ax.tick_params(axis='y', labelleft=False)
    ax.tick_params(axis='z', labelleft=False)



def calculate_view_angles(vals):
    # 选择关键点
    shoulder_left = vals[14, :]  # 假设1号为左肩
    shoulder_right = vals[11, :] # 假设4号为右肩
    hip_left = vals[1, :]      # 假设11号为左胯
    hip_right = vals[4, :]     # 假设14号为右胯

    # 计算两个向量
    vector1 = shoulder_right - shoulder_left
    vector2 = hip_right - hip_left

    # 计算法线向量（正视图方向）
    normal_vector = np.cross(vector1, vector2)
    normal_angle = np.arctan2(normal_vector[1], normal_vector[0]) * 180. / np.pi

    # 计算侧视图方向（法线向量与向上向量的外积）
    up_vector = np.array([0, 0, 1])
    side_vector = np.cross(normal_vector, up_vector)
    side_angle = np.arctan2(side_vector[1], side_vector[0]) * 180. / np.pi

    return normal_angle, side_angle

def show3Dpose_corrected(vals, ax, view="original"):
    ax.clear()
    colors = [(138/255, 201/255, 38/255),
              (255/255, 202/255, 58/255),
              (25/255, 130/255, 196/255)]
    I = np.array([0, 0, 1, 4, 2, 5, 0, 7, 8, 8, 14, 15, 11, 12, 8, 9])
    J = np.array([1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])
    LR = [3, 3, 3, 3, 3, 3, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1]

    shoulder_angle, side_angle = calculate_view_angles(vals)

    if view == "front":
        ax.view_init(elev=10., azim=shoulder_angle)
    elif view == "side":
        ax.view_init(elev=10., azim=side_angle)
    else:
        ax.view_init(elev=15., azim=70)  # 默认视角

    # 绘制
    for i in range(len(I)):
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in (0, 1, 2)]
        ax.plot(x, z, -y, lw=2, color=colors[LR[i]-1])

    RADIUS = 0.72
    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_aspect('equal')
    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)
    ax.tick_params(axis='x', labelbottom=False)
    ax.tick_params(axis='y', labelleft=False)
    ax.tick_params(axis='z', labelleft=False)

name = "test-1"
# Load the npz file with skeletal data
data = np.load(f'/home/zlt/Documents/SkydivingPose/output/{name}/output_3D/output_keypoints_3d.npz')
reconstruction = data['reconstruction']
# Path to the directory containing the PNG images
image_dir = f'/home/zlt/Documents/SkydivingPose/output/{name}/pose2D'

# Retrieve a list of PNG files sorted by name (assuming they are named sequentially)
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('_2D.png')])


# 定义子图布局，其中第一个子图为2D，其余为3D
fig = plt.figure(figsize=(24, 6))
ax_img = fig.add_subplot(1, 4, 1)  # 2D用于显示图像
axs = [ax_img] + [fig.add_subplot(1, 4, i + 2, projection='3d') for i in range(3)]  # 后续三个子图为3D

# 创建GIF用的文件名列表
filenames = []

for idx, frame in enumerate(reconstruction):
    # 在第一个子图中显示图像
    # axs[0].clear()  # 清除上一个图像
    axs[0].imshow(plt.imread(os.path.join(image_dir, image_files[idx])))
    axs[0].axis('off')  # 隐藏坐标轴

    # 在其余的3D子图中显示3D姿态
    show3Dpose_corrected(frame, axs[1], view="original")
    show3Dpose_corrected(frame, axs[2], view="front")
    show3Dpose_corrected(frame, axs[3], view="side")

    # 保存每一帧到文件
    filename = f'./gif/frame_{idx}.png'
    plt.savefig(filename)
    filenames.append(filename)
    plt.pause(0.05)  # 更新图表的暂停时间

# Create GIF using imageio
with imageio.get_writer(f'./animation-{name}.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)