from PIL import Image
import numpy as np
import os
import cv2
import math

def crop_image_into_nine_parts(image, filename ,save_path):
    height, width, _ = image.shape

    # 计算每个小图片的宽度和高度
    crop_width = width // 3 + 1
    crop_height = height // 3 + 1

    cropped_images = []

    index = 0

    # 遍历每个裁剪区域的起始位置
    for y in range(0, height, crop_height):
        for x in range(0, width, crop_width):
            # 根据起始位置和小图片的宽度和高度进行裁剪
            cropped = image[y:y+crop_height, x:x+crop_width]
            cropped_images.append(cropped)
            cv2.imwrite(os.path.join(save_path,f"{index}{filename}"), cropped)
            index +=1

    return cropped_images

def crop_images_in_folder(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        crop_image_into_nine_parts(image, filename, output_folder)

original_path = '../original_pic'
save_path = '../test_pic'

crop_images_in_folder(original_path, save_path)

