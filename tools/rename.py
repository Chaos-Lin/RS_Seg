import os

# 指定文件所在的目录
directory = 'D:/Search/Remote sensing/Dataset/forest/SegmentationClass_BW'

# 遍历目录中的文件
for filename in os.listdir(directory):
    # 构建旧文件的完整路径
    old_name = os.path.join(directory, filename)

    # 构建新文件名，将"labels"替换为"image"
    new_name = os.path.join(directory, filename.replace("mask", "sat"))

    # 重命名文件
    os.rename(old_name, new_name)