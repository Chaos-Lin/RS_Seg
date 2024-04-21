import os
import glob

def delete_images_with_suffix(folder, suffix):
    # 构建匹配的文件路径模式
    pattern = os.path.join(folder, f"*")

    # 查找匹配的文件
    files = glob.glob(pattern)
    
    # 删除每个匹配的文件
    for file in files:
        os.remove(file)

def delete_images_without_suffix(folder, suffix):
    # 构建匹配的文件路径模式
    pattern = os.path.join(folder, "*")
    
    # 查找所有文件
    files = glob.glob(pattern)
    
    # 删除不以指定结尾的文件
    for file in files:
        if not file.endswith(suffix):
            os.remove(file)
            
# 指定文件夹路径
folder = "../Dataset/Chicago/SegmentationClass"
# 指定要删除/保留的文件结尾
suffix = ".png"

# 删除指定结尾的图片
delete_images_without_suffix(folder, suffix)