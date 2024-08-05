import numpy as np

# 定义metadata信息
metadata = {
    'layout_name': 'skydriving',
    'num_joints': 17,
    'keypoints_symmetry': [[4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]]
}

# 加载四个文件
file_paths = [
    'output/test-001/001_keypoints_2d.npz',
    'output/test-001/002_keypoints_2d.npz',
    'output/test-001/003_keypoints_2d.npz',
    'output/test-001/004_keypoints_2d.npz'
]

# 初始化数据字典
data = {'S1': {'platform': []}}

# 读取文件内容并将其合并到'S1'的'platform'列表中
for file_path in file_paths:
    npz_data = np.load(file_path, allow_pickle=True)
    data['S1']['platform'].append(npz_data['reconstruction'])

# 将metadata信息加入data字典中
data['metadata'] = metadata

# 保存到新的 npz 文件
output_file_path = '/home/zlt/Documents/SkydivingPose/dataset/data_2d_skydriving_cpn_ft_platform_dbb.npz'
np.savez(output_file_path, **data)

print(f'Data saved to {output_file_path}')
