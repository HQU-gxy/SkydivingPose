import numpy as np

# 加载npz文件
npz_file = np.load('dataset/data_2d_h36m_cpn_ft_h36m_dbb.npz', allow_pickle=True)

# 查看npz文件中的所有数组名称
print("Arrays in the npz file:")
print(npz_file.files)

# 辅助函数，用于打印字典的内容
def print_dict_info(d, indent=0):
    indent_str = ' ' * indent
    for key, value in d.items():
        value_type = type(value)
        value_shape = getattr(value, 'shape', 'N/A')
        if isinstance(value, list):
            value_length = len(value)
            print(f"{indent_str}Key: {key}, Type: {value_type}, Length: {value_length}")
            # 查看列表内元素的类型和形状
            if value_length > 0:
                first_elem_type = type(value[0])
                first_elem_shape = getattr(value[0], 'shape', 'N/A')
                print(f"{indent_str}  First element type: {first_elem_type}, shape: {first_elem_shape}")
        else:
            value_length = 'N/A'
            print(f"{indent_str}Key: {key}, Type: {value_type}, Shape: {value_shape}, Length: {value_length}")
        if isinstance(value, dict):
            print_dict_info(value, indent + 2)

# 遍历并打印每个数组的维度（shape）及字典的键和对应的值的类型和形状
for array_name in npz_file.files:
    print(f"\nArray name: {array_name}")
    array_content = npz_file[array_name]
    
    if array_content.shape == ():  # 0-dimensional array
        item_content = array_content.item()
        if isinstance(item_content, dict):
            print(f"Contained item type: {type(item_content)}")
            print(f"Contained dict keys: {list(item_content.keys())}")
            print_dict_info(item_content)
    else:
        # 打印数组的形状
        print(f"Array shape: {array_content.shape}")
        
# 遍历并打印每个数组的内容和类型
for array_name in npz_file.files:
    print(f"Array name: {array_name}")
    array_content = npz_file[array_name]
    print(f"Type: {type(array_content)}")
    print(f"Content: {array_content}")