import os.path
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from utils import *
from torchvision import transforms
import random
import numpy as np
from PIL import Image
import torch

transform = transforms.Compose([
    transforms.ToTensor()
    # 将输入的 PIL 图像或 NumPy 数组转换为 PyTorch 的张量（Tensor）格式。
])
# transforms.Compose 类接受一个操作序列作为参数，
# 并将其组合为一个可调用对象，可以连续地将多个操作应用到图像上。

class SEGDataset(Dataset):
    def __init__(self,args, mode=None):
        self.args = args
        self.path = args['dataset_path']
        # 数据集的根目录
        self.name = os.listdir(os.path.join(self.path,'JPEGImages'))
        # os.listdir可以获取该目录下的所有名字

    def __len__(self):
        return len(self.name)
    # 获取name这个列表的长度，也就是那么数据集的样本数量

    def __getitem__(self, item):
        segment_name = self.name[item]
        if self.args['is_color']:
            segment_path = os.path.join(self.path, 'SegmentationClass_CL', segment_name)
        else:
            segment_path = os.path.join(self.path, 'SegmentationClass_BW', segment_name)
        image_path = os.path.join(self.path, 'JPEGImages', segment_name)     
        
        segment_image = keep_image_size_open(segment_path)
        image = keep_image_size_open(image_path)
        return transform(image),transform(segment_image)
    # 数据的每一项的构成
    
    def shuffle(self):
        random.shuffle(self.name)
    
    def split_dataset(self, split_ratio=0.8):
        split_index = int(len(self.name) * split_ratio)
        train_names = self.name[:split_index]
        val_names = self.name[split_index:]

        train_dataset = SEGDataset(self.args)
        train_dataset.name = train_names

        val_dataset = SEGDataset(self.args)
        val_dataset.name = val_names

        return train_dataset, val_dataset

class CLSDataset(Dataset):
    def __init__(self,args, mode = "train"):
        super(CLSDataset, self).__init__()
        self.args = args
        self.path = args['dataset_path']
        self.data_txt = os.path.join(self.path,f'{mode}.txt')
        imgs = []
        fh = open(self.data_txt, 'r')
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))  # imgs中包含有图像路径和标签
        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        # self.txt[:-4]，下文加载txt时，路径中不需要有后缀，所以去掉.txt四个字符
        img = keep_image_size_open(os.path.join(self.data_txt[:-4], fn), size=self.args.input_size)
        img = transform(img)
        return img, label

class INSDataset(Dataset):
    def __init__(self,args, mode="Train", place=None):
        '''
        :param args:
        :param mode: Train, Val, Test
        :param place: Rural || Urban || All
        '''
        super(INSDataset, self).__init__()
        self.args = args
        self.path = args['dataset_path']
        self.patch_size = args['patch_size']
        self.overlap = args['overlap']
        # self.overlap = 64
        self.place = place
        self.mode = mode

        if self.place == "ALL":
            self.name = os.listdir(os.path.join(self.path, self.mode, "Rural", "images_png"))
            self.name.extend(os.listdir(os.path.join(self.path, self.mode, "Urban", "images_png")))
        else:
            self.name = os.listdir(os.path.join(self.path, self.mode, self.place, "images_png"))

        # os.listdir可以获取该目录下的所有名字
    def __len__(self):
        return len(self.name)
    def __getitem__(self, item):


        if self.place == "ALL":
            if self.mode == "Train":
                segment_name = self.name[item]
                if item >= 1366:
                    place = "Urban"
                else:
                    place = "Rural"
                segment_path = os.path.join(self.path, self.mode, place, "masks_png", segment_name)
                image_path = os.path.join(self.path, self.mode, place, "images_png", segment_name)
            elif self.mode == "Val":
                segment_name = self.name[item]
                if item >= 992:
                    place = "Urban"
                else:
                    place = "Rural"
                segment_path = os.path.join(self.path, self.mode, place, "masks_png", segment_name)
                image_path = os.path.join(self.path, self.mode, place, "images_png", segment_name)
        else:
            segment_name = self.name[item]
            segment_path = os.path.join(self.path, self.mode, self.place, "masks_png", segment_name)
            image_path = os.path.join(self.path, self.mode, self.place, "images_png", segment_name)

        segment_image = Image.open(segment_path)
        image = Image.open(image_path)
        images = transform(image)
        masks = torch.tensor(np.array(segment_image)).long().unsqueeze(0)

        # image_patches = split_image_to_patches(images, self.patch_size, self.overlap)
        # segment_image_patches = split_image_to_patches(masks, self.patch_size, self.overlap)
        #
        # return torch.stack(image_patches), torch.stack(segment_image_patches).squeeze()
        return images, masks
    # 数据的每一项的构成


# def split_image_to_patches(image, patch_size, overlap):
#     """
#     将图像分割成小块，并加入重叠部分。
#
#     Args:
#         image (Tensor): 输入图像，形状为 (C, H, W)。
#         patch_size (int): 每个小块的大小。
#         overlap (int): 小块之间的重叠大小。
#
#     Returns:
#         patches (List[Tensor]): 分割后的图像小块列表。
#     """
#     C, H, W = image.shape
#     step = patch_size - overlap
#     patches = []
#     for i in range(0, H, step):
#         for j in range(0, W, step):
#             patch = image[:, i:i + patch_size, j:j + patch_size]
#             if patch.shape[1] != patch_size or patch.shape[2] != patch_size:
#                 patch = TF.pad(patch, (0, 0, patch_size - patch.shape[2], patch_size - patch.shape[1]))
#             patches.append(patch)
#     return patches

def split_image_to_patches(image, patch_size, overlap):
    """
    将图像分割成小块，并加入重叠部分。

    Args:
        image (Tensor): 输入图像，形状为 (C, H, W)。
        patch_size (int): 每个小块的大小。
        overlap (int): 小块之间的重叠大小。

    Returns:
        patches (List[Tensor]): 分割后的图像小块列表。
    """
    # 图像的通道数、高度和宽度
    C, H, W = image.shape
    # 滑窗步长，小块大小减去重叠部分大小
    step = patch_size - overlap
    patches = []
    for i in range(0, H, step):
        for j in range(0, W, step):
            # 提取图像小块
            patch = image[:, i:i + patch_size, j:j + patch_size]
            # 如果小块尺寸不等于指定尺寸，进行填充
            if patch.shape[1] != patch_size or patch.shape[2] != patch_size:
                patch = TF.pad(patch, (0, 0, patch_size - patch.shape[2], patch_size - patch.shape[1]))
            # 添加小块到列表中
            patches.append(patch)
    return patches

class MyDataset():
    def __init__(self):
        self.TRAIN_MAP = {
            "classification": CLSDataset,
            "segmentation": SEGDataset,
            "InstanceSeg": INSDataset
        }

    def getDataset(self, args, mode=None):
        return self.TRAIN_MAP[args['mode']](args, mode)

if __name__=='__main__':
    data = SEGDataset("D:\A file of SHU\Remote sensing\My_Unet\data")
    print(data[0][0].shape)
    print(data[0][1].shape)
