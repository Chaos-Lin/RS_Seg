from PIL import Image
import numpy as np
import os

# 图像文件夹路径
folder_path = "D:/Search/Remote sensing/Dataset/Zurich/label"
save_folder = "D:/Search/Remote sensing/Dataset/Zurich/labels"

# 获取文件夹中所有以"image"开头的jpg图像文件
image_paths = [file for file in os.listdir(folder_path) if file.endswith("_labels.png")]

for path in image_paths:
    # 构建完整路径
    image_path = os.path.join(folder_path, path)

    # 打开图像
    image = Image.open(image_path)

    # 转换为灰度图像
    grayscale_image = image.convert("L")

    # 将像素值限制在0和1之间
    normalized_image = np.array(grayscale_image) / 255.0

    # 保存处理后的图像
    save_path = os.path.join(save_folder, path)
    Image.fromarray((normalized_image * 255).astype(np.uint8)).save(save_path)