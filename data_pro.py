import numpy as np
import cv2  # 如果需要读取图像文件

# 假设输入的灰度图为一个NumPy数组，读取灰度图
# gray_image = cv2.imread('path_to_label_image.png', cv2.IMREAD_GRAYSCALE)

def convert_to_one_hot(gray_image, num_classes=8):
    # 获取图像的高和宽
    height, width = gray_image.shape
    
    # 初始化一个空的8通道图像
    one_hot_image = np.zeros((height, width, num_classes), dtype=np.uint8)
    
    # 独热编码转换
    for c in range(num_classes):
        one_hot_image[:, :, c] = (gray_image == c).astype(np.uint8)
    
    return one_hot_image

# 示例：假设gray_image是一个包含类别标签的灰度图像
gray_image = np.array([
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 0]
])

one_hot_image = convert_to_one_hot(gray_image)

print("Gray image:")
print(gray_image)
print("One-hot encoded image:")
print(one_hot_image)


def convert_one_hot_to_gray(one_hot_image):
    # 使用argmax找到最后一个维度（通道）上的最大值索引
    gray_image = np.argmax(one_hot_image, axis=-1)
    return gray_image

# 示例：假设one_hot_image是一个8通道的独热编码图像
one_hot_image = np.array([
    [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0]],
    [[0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0]],
    [[0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0]]
])

gray_image = convert_one_hot_to_gray(one_hot_image)

print("One-hot encoded image:")
print(one_hot_image)
print("Converted gray image:")
print(gray_image)

