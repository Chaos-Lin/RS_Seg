import os.path

from torch.utils.data import Dataset

from utils import *
from torchvision import transforms
import random

transform = transforms.Compose([
    transforms.ToTensor()
    # 将输入的 PIL 图像或 NumPy 数组转换为 PyTorch 的张量（Tensor）格式。
])
# transforms.Compose 类接受一个操作序列作为参数，
# 并将其组合为一个可调用对象，可以连续地将多个操作应用到图像上。

class MyDataset(Dataset):
    def __init__(self,args):
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

        train_dataset = MyDataset(self.args)
        train_dataset.name = train_names

        val_dataset = MyDataset(self.args)
        val_dataset.name = val_names

        return train_dataset, val_dataset
    
# class List_to_Dataset:
#     def __init__(self, dataset):
#         self.dataset = dataset

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, index):
#         return self.dataset[index]

if __name__=='__main__':
    data = MyDataset("D:\A file of SHU\Remote sensing\My_Unet\data")
    print(data[0][0].shape)
    print(data[0][1].shape)
