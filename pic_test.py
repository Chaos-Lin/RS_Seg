
from torchvision import transforms
from PIL import Image
import numpy as np

# 定义图像转换
transform = transforms.Compose([
    transforms.ToTensor()  # 将图像转换为Tensor
])

# 打开图像
image_path1 = "D:\Search\RS\Dataset\LoveDA\Train\\Rural\masks_png\\0.png"
image_path2 = "D:\Search\RS\Dataset\LoveDA\Train\\Rural\masks_png\\1.png"
image_path3 = "D:\Search\RS\Dataset\LoveDA\Train\\Rural\masks_png\\2.png"
image_path4 = "D:\Search\RS\Dataset\LoveDA\Train\\Rural\masks_png\\3.png"
image_path5 = "D:\Search\RS\Dataset\LoveDA\Train\\Rural\masks_png\\4.png"
image_path6 = "D:\Search\RS\Dataset\LoveDA\Train\\Rural\masks_png\\5.png"
image_path7 = "D:\Search\RS\Dataset\LoveDA\Train\\Rural\masks_png\\47.png"
image_path8 = "D:\Search\RS\Dataset\LoveDA\Train\\Rural\masks_png\\479.png"
def cheek(img):
    image = Image.open(img)
    # 应用转换
    # print(np.array(image))
    image_tensor = transform(image).squeeze()
    # 查看Tensor形状
    print(image_tensor.shape)
    print(image_tensor)# 输出: (C, H, W)
    unique_numbers = np.unique(image_tensor)
    print(unique_numbers)

# cheek(image_path1)
# cheek(image_path2)
# cheek(image_path3)
# cheek(image_path4)
# cheek(image_path5)
# cheek(image_path6)
# cheek(image_path7)
# cheek(image_path8)
cheek("D:\Search\RS\Dataset\LoveDA\Train\Rural\images_png\\0.png")
value_to_int = {
    0.0: 0,
    0.00392157: 1,
    0.00784314: 2,
    0.01176471: 3,
    0.01568628: 4,
    0.01960784: 5,
    0.02352941: 6,
    0.02745098: 7
}


import numpy as np
import cv2

# # 假设输入矩阵
# matrix = np.random.choice([0.0, 0.00392157, 0.00784314, 0.01176471, 0.01568628, 0.01960784, 0.02352941, 0.02745098], (1024, 1024))

# 定义像素值到整数值的映射
value_to_int = {
    0.0: 0,
    0.00392157: 1,
    0.00784314: 2,
    0.01176471: 3,
    0.01568628: 4,
    0.01960784: 5,
    0.02352941: 6,
    0.02745098: 7
}
image = Image.open(image_path1)
# 应用转换
matrix = transform(image).squeeze()
# 创建一个空的整数矩阵
int_matrix = np.zeros(matrix.shape, dtype=np.uint8)

# 遍历矩阵，应用映射
for value, int_value in value_to_int.items():
    int_matrix[matrix == value] = int_value

# 打印转换后的矩阵
print("转换后的整数矩阵:")
print(int_matrix)

# 保存为灰度图
cv2.imwrite('converted_image.png', int_matrix)


