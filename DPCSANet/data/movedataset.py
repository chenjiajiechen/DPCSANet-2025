import os
import shutil

# 定义文件夹路径
image_folder = '/home/chenjiajie/桌面/Remo_detection/datasets/LEVIR-YOLO/fogtest/mist'  # 替换为实际路径
label_folder = '/home/chenjiajie/桌面/Remo_detection/datasets/LEVIR-YOLO/test/labels'  # 替换为实际路径
output_folder = '/home/chenjiajie/桌面/Remo_detection/datasets/LEVIR-YOLO/fogtest/mistlabel'  # 替换为实际路径

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 遍历图片文件夹
for filename in os.listdir(image_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # 根据需要添加其他图片格式
        # 提取文件名（去掉后缀）
        file_base_name = os.path.splitext(filename)[0]
        # 构造对应的标签文件路径
        label_file_path = os.path.join(label_folder, f'{file_base_name}.txt')

        # 检查标签文件是否存在
        if os.path.exists(label_file_path):
            # 复制标签文件到输出文件夹
            shutil.copy(label_file_path, output_folder)
            print(f'Copied: {label_file_path} to {output_folder}')
        else:
            print(f'Label file not found for: {filename}')
