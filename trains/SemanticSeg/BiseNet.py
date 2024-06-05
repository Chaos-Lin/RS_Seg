import logging
from PIL import Image
import torch
from torch import nn
from torch import optim
import os
from torchvision.utils import save_image
import pandas as pd
from tqdm import tqdm

from utils import Metrics

from pathlib import Path

from utils import OhemCELoss
from torchvision import transforms
logger = logging.getLogger('RS')


class BiseNet():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.BCELoss()
        self.patch_size = args.patch_size
        self.num_classes = args.num_classes
        self.overlap = args.overlap
        self.pic_size = args.pic_size
        # epochs, best_epoch = 0, 0
        self.criteria_pre = OhemCELoss(0.7, lb_ignore=255)
        self.criteria_aux1 = OhemCELoss(0.7, lb_ignore=255)
        self.criteria_aux2 = OhemCELoss(0.7, lb_ignore=255)

    def do_train(self, model, train_dataloader, epoch, return_epoch_results=False):
        opt = optim.Adam(params=model.parameters(), lr=self.args.learning_rate)
        model.train()

        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }

        while True:
            # epochs += 1
            t = 0
            with tqdm(train_dataloader) as data_loader:
                for i, (image, segment_image) in enumerate(data_loader):
                    image = image.view(-1, 3, self.patch_size, self.patch_size ).to(self.args.device)
                    segment_image = segment_image.view(-1, self.patch_size, self.patch_size).to(self.args.device)
                    # 数据移动至设备
                    out, out16, out32 = model(image)
                    out = out.to(self.args.device)
                    out16 = out16.to(self.args.device)
                    out32 = out32.to(self.args.device)
                    loss_pre = self.criteria_pre(out, segment_image)
                    loss_aux1 = self.criteria_aux1(out16, segment_image)
                    loss_aux2 = self.criteria_aux2(out32, segment_image)
                    loss = loss_pre + loss_aux1 + loss_aux2
                    opt.zero_grad()
                    # 清空梯度
                    loss.backward()
                    # 反向计算
                    opt.step()
                    # 更新梯度

                    if i % 30 == 0:
                        _image = image[0].to('cpu')
                        _segment_image = segment_image[0].to('cpu')
                        _out_image = out[0].argmax(dim=0).to('cpu')
                        _segment_image_colored = self.apply_colormap(_segment_image)
                        _out_image_colored = self.apply_colormap(_out_image)
                        img = torch.stack([_image, _segment_image_colored, _out_image_colored], dim=0)
                        # 将每个batch里的第一张图进行拼接展示
                        save_dir = f"{self.args['image_save_path']}/{self.args['model_name']}/{self.args['dataset_name']}"
                        Path(save_dir).mkdir(parents=True, exist_ok=True)
                        image_save_path = save_dir + "/" + f"train_{self.args['super_epoch']}_{epoch}_{t}.png"
                        save_image(img, image_save_path)
                        t += 1
                logger.info(
                    f"super-epoch:{self.args['super_epoch']},train-epoch:{epoch}-train_loss===>>{round(loss.item(), 4)}")
                # 其实保存条件也应该设置一下
                # torch.save(model.state_dict(), self.args['model_save_path'])
                # logger.info('save weight:' + str(self.args['model_save_path']))



                # val_results = self.do_test(model, test_dataloder)
                # return val_results
                return None


                # if epochs - best_epoch >= self.args.early_stop:
                #     return val_results if return_epoch_results else None

    def do_test(self, model, test_dataloader):
        model.eval()
        eval_results = {
            'loss': [],
            'pix_acc': [],
            'mean_iou': [],
            'freq_iou': [],
            '0-background':[],
            '1-uninterested':[],
            '2-building':[],
            '3-road':[],
            '4-water':[],
            '5-barren':[],
            '6-forest':[],
            '7-agricultural':[]
        }
        t = 0

        with torch.no_grad():
            with tqdm(test_dataloader) as data_loader:
                for i, (image, segment_image) in enumerate(data_loader):
                    image = image.view(-1, 3, self.patch_size, self.patch_size).to(self.args.device)
                    segment_image = segment_image.view(-1, self.patch_size, self.patch_size).to(self.args.device)
                    # image = image.to(self.args['device'])


                    out, _, _ = model(image)
                    out = out.to(self.args.device)
                    # todo how to match is to one-hot or tensor
                    test_loss = self.criteria_pre(out, segment_image)
                    eval_results['loss'].append(test_loss.item())

                    if i % 10 == 0:
                        _image = image[0].to('cpu')
                        _segment_image = segment_image[0].to('cpu')
                        _out_image = out[0].argmax(dim=0).to('cpu')
                        _segment_image_colored = self.apply_colormap(_segment_image)
                        _out_image_colored = self.apply_colormap(_out_image)
                        img = torch.stack([_image, _segment_image_colored, _out_image_colored], dim=0)
                        # 将每个batch里的第一张图进行拼接展示
                        save_dir = f"{self.args['image_save_path']}/{self.args['model_name']}/{self.args['dataset_name']}"
                        Path(save_dir).mkdir(parents=True, exist_ok=True)
                        image_save_path = save_dir + "/" + f"test_{self.args['super_epoch']}_{t}.png"
                        save_image(img, image_save_path)
                        t += 1

                    # out = self.resize_image(out)
                    # segmented_image = self.resize_image(segmented_image)
                    pix_acc, mean_iou, freq_iou, iou = Metrics(out, segment_image)
                    eval_results['pix_acc'].append(pix_acc)
                    eval_results['mean_iou'].append(mean_iou)
                    eval_results['freq_iou'].append(freq_iou)
                    classes = ['0-background', '1-uninterested', '2-building', '3-road',
                    '4-water', '5-barren', '6-forest', '7-agricultural']
                    for i, cls in enumerate(classes):
                        eval_results[f'{cls}'].append(iou[i])
            loss = sum(eval_results['loss']) / len(eval_results['loss'])
            logger.info(f"super-epoch:{self.args['super_epoch']}-test_loss===>>{round(loss, 4)}")
            return eval_results

    def pred(self, model, dataloader):
        model.eval()
        # for i, (image, segment) in enumerate(dataloader):
        #     image = image.to(self.args.device)
        #     segment_image = segment.to(self.args.device)
        #     # 数据移动至设备
        #     out, _, _ = model(image)
        #     _image = image[0].to('cpu')
        #     _segment_image = segment_image[0].to('cpu')
        #     _out_image = out[0].argmax(dim=0).to('cpu')
        #     _segment_image_colored = self.apply_colormap(_segment_image.squeeze(0))
        #     _out_image_colored = self.apply_colormap(_out_image)
        #     img = torch.stack([_image, _segment_image_colored, _out_image_colored], dim=0)
        #     # 将每个batch里的第一张图进行拼接展示
        #     save_dir = f"{self.args['image_save_path']}/{self.args['model_name']}/{self.args['dataset_name']}"
        #     Path(save_dir).mkdir(parents=True, exist_ok=True)
        #     image_save_path = save_dir + "/" + f"pred_{self.args['super_epoch']}_{i}.png"
        #     save_image(img, image_save_path)

        for i, (image_patches, mask_patches) in enumerate(dataloader):
            a = image_patches[0][0]
            b = mask_patches[0][1]
            image_patches, mask_patches = image_patches[0], mask_patches[0]

            mask = self.stitch_patches(mask_patches.unsqueeze(1),(1,1024,1024),overlap=self.overlap)
            mask = self.apply_colormap(mask.squeeze())

            image = self.stitch_patches(image_patches,(3,1024,1024), overlap=self.overlap)
            c = self.apply_colormap(b)


            save_image(mask, f"D:\\{i}mask.png")
            save_image(image, f"D:\\{i}image.png")
            # save_image(a, "D:\\ccc.png")
            # save_image(c, "D:\\ddd.png")
            image_patch = image_patches.view(-1, 3, self.patch_size, self.patch_size)
            image_patch = image_patch.to("cuda")
            mask_patch = mask_patches.to("cpu")
            outputs, _, _ = model(image_patch)
            output = outputs.argmax(dim=1).to("cpu")
            output = self.stitch_patches(output.unsqueeze(1),(1,1024,1024), overlap=self.overlap)
            output = self.apply_colormap(output.squeeze())
            transform_to_pil = transforms.ToPILImage()
            image = transform_to_pil(output)
            image.save(f"D:\\{i}output.png")



    # def stitch_patches(self, patches, image_size=(3, 1024, 1024), patch_size=256, overlap=0):
    #     """
    #     将图像小块重新拼接成完整的图像，并去掉重叠部分。
    #
    #     Args:
    #         patches (List[Tensor]): 分割后的图像小块列表。
    #         image_size (Tuple[int, int, int]): 完整图像的大小 (C, H, W)。
    #         patch_size (int): 每个小块的大小。
    #         overlap (int): 小块之间的重叠大小。
    #
    #     Returns:
    #         image (Tensor): 拼接后的完整图像。
    #     """
    #     C, H, W = image_size
    #     step = patch_size - overlap
    #     stitched_image = torch.zeros((C, H, W))
    #     count_map = torch.zeros((C, H, W))
    #     index = 0
    #
    #     for i in range(0, H, step):
    #         for j in range(0, W, step):
    #             # Check dimensions of the current patch
    #             patch_height = min(patch_size, H - i)
    #             patch_width = min(patch_size, W - j)
    #             a = patches[index].shape
    #             if patches[index].shape[1] != patch_height or patches[index].shape[2] != patch_width:
    #                 print(f"Error at index {index}: patch size mismatch.")
    #                 print(f"Expected patch size: ({C}, {patch_height}, {patch_width})")
    #                 print(f"Actual patch size: {patches[index].shape}")
    #                 return None
    #
    #             stitched_image[:, i:i + patch_size, j:j + patch_size] += patches[index][:, :patch_height, :patch_width]
    #             count_map[:, i:i + patch_size, j:j + patch_size] += 1
    #             index += 1
    #
    #     # Handle potential divide-by-zero issues in count_map
    #     count_map = torch.where(count_map == 0, torch.ones_like(count_map), count_map)
    #
    #     return stitched_image / count_map

    # def stitch_patches(self, patches, image_size=(3, 1024, 1024), patch_size=256, overlap=64):
    #     # 在重贴区域赋予类别时，取重叠两次或多次预测结果中得分最高的那个类别
    #
    #     """
    #     将图像小块重新拼接成完整的图像，并去掉重叠部分。
    #
    #     Args:
    #         patches (List[Tensor]): 分割后的图像小块列表。
    #         image_size (Tuple[int, int, int]): 完整图像的大小 (C, H, W)。
    #         patch_size (int): 每个小块的大小。
    #         overlap (int): 小块之间的重叠大小。
    #
    #     Returns:
    #         image (Tensor): 拼接后的完整图像。
    #     """
    #     C, H, W = image_size
    #     step = patch_size - overlap
    #     stitched_image = torch.zeros((C, H, W))
    #     count_map = torch.zeros((C, H, W))
    #     index = 0
    #
    #     for i in range(0, H, step):
    #         for j in range(0, W, step):
    #             # Check dimensions of the current patch
    #             patch_height = min(patch_size, H - i)
    #             patch_width = min(patch_size, W - j)
    #             current_patch = patches[index]
    #
    #             # Adjust the current patch to fit within the image boundaries
    #             if current_patch.shape[1] != patch_height or current_patch.shape[2] != patch_width:
    #                 current_patch = current_patch[:, :patch_height, :patch_width]
    #
    #             stitched_image[:, i:i + patch_height, j:j + patch_width] += current_patch
    #             count_map[:, i:i + patch_height, j:j + patch_width] += 1
    #             index += 1
    #
    #     # Handle potential divide-by-zero issues in count_map
    #     count_map = torch.where(count_map == 0, torch.ones_like(count_map), count_map)
    #
    #     return stitched_image / count_map

    def stitch_patches(patches, image_size=(3, 1024, 1024), patch_size=256, overlap=64):
        """
        将图像小块重新拼接成完整的图像，并去掉重叠部分。

        Args:
            patches (List[Tensor]): 分割后的图像小块列表。
            image_size (Tuple[int, int, int]): 完整图像的大小 (C, H, W)。
            patch_size (int): 每个小块的大小。
            overlap (int): 小块之间的重叠大小。

        Returns:
            image (Tensor): 拼接后的完整图像。
        """
        # 图像通道数，高度和宽度
        C, H, W = image_size
        # 滑窗步长，小块大小减去重叠部分大小
        step = patch_size - overlap
        # 初始化拼接后的图像和计数图，用于归一化重叠区域的像素值
        stitched_image = torch.zeros((C, H, W))
        count_map = torch.zeros((C, H, W))
        index = 0

        # 中心区域大小，去掉两边重叠部分
        center_patch_size = patch_size - 2 * overlap // 2
        center_start = overlap // 2
        center_end = center_start + center_patch_size

        for i in range(0, H, step):
            for j in range(0, W, step):
                if index >= len(patches):
                    break

                # 提取当前patch的中心区域
                current_patch = patches[index][:, center_start:center_end, center_start:center_end]

                # 计算中心区域在拼接图像中的终止位置
                i_end = i + center_patch_size
                j_end = j + center_patch_size

                # 将中心区域的像素值加到拼接图像中对应位置
                stitched_image[:, i:i_end, j:j_end] += current_patch
                # 增加计数图中对应位置的计数
                count_map[:, i:i_end, j:j_end] += 1
                index += 1

        # 处理计数图中的零值，避免除零错误
        count_map = torch.where(count_map == 0, torch.ones_like(count_map), count_map)

        # 返回归一化后的拼接图像
        return stitched_image / count_map

    def apply_colormap(self, label):
        transform = transforms.Compose([
            transforms.ToTensor()  # 将图像转换为Tensor
        ])
        """
        将单通道标签图像转换为3通道彩色图像。

        label: torch.Tensor, 形状为 (H, W)，每个值表示类别索引
        colormap: torch.Tensor, 形状为 (num_classes, 3)，每行表示一个类别的颜色
        return: torch.Tensor, 形状为 (3, H, W)，彩色图像
        """
        colormap = torch.tensor([
            [236, 235, 240],  # 类别 0 - 白色 背景
            [143,151,154],  # 类别 1 - 灰色 不感兴趣
            [112, 110, 180],  # 类别 2 - 紫色 建筑
            [247, 183, 84],  # 类别 3 - 黄色 道路
            [104, 149, 191],  # 类别 4 - 蓝色 水
            [207,183,157],  # 类别 5 - 棕色 贫瘠
            [49, 122, 95],  # 类别 6 - 深绿 森林
            [59, 154, 112]  # 类别 7 - 浅绿 农业
        ], dtype=torch.float32)

        colormap = colormap / 255.0


        # 获取图像尺寸
        H, W = label.shape
        # 创建一个空的3通道图像
        color_image = torch.zeros(3, H, W, dtype=torch.float32)

        for cls in range(colormap.size(0)):
            mask = label == cls
            color_image[0][mask] = colormap[cls][0]
            color_image[1][mask] = colormap[cls][1]
            color_image[2][mask] = colormap[cls][2]

        return color_image
    def set_args(self, args):
        self.args = args
        return
