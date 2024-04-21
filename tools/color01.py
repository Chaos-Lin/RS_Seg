from PIL import Image
import os


# 图像文件夹路径
folder_path = "D:/Search/Remote sensing/Dataset/zurich_org"
save_folder = "D:/Search/Remote sensing/Dataset/zurich/123"

# 获取文件夹中所有以"image"开头的jpg图像文件
image_paths = [file for file in os.listdir(folder_path) if file.endswith("labels.png")]

for path in image_paths:
    # 构建完整路径
    image_path = os.path.join(folder_path, path)
    
    # 保存处理后的图像
    save_path = os.path.join(save_folder, path)
    # 打开图像
    image = Image.open(image_path)
    # 转换为灰度图像
    image = image.convert("L")

    # 应用阈值操作将灰度图像转换为只有黑白的 0 和 1
    threshold = 128
    image_bw = image.point(lambda x: 0 if x < threshold else 1, mode="1")
    image_bw.save(save_path)


    
