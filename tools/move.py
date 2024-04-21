import shutil
import os

def move_label_images(input_folder, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith('image.png'):
            # 构建源文件路径和目标文件路径
            source_file = os.path.join(input_folder, filename)
            destination_file = os.path.join(output_folder, filename)
            # 移动图片文件到目标文件夹
            shutil.move(source_file, destination_file)

# 输入文件夹路径
input_folder = "D:/Search/Remote sensing/Dataset/chicago_org"
# 输出文件夹路径
output_folder = "D:\Search\Remote sensing\Dataset\chicago\JPEGImages"

# 移动以"..."结尾的PNG图片到目标文件夹
move_label_images(input_folder, output_folder)