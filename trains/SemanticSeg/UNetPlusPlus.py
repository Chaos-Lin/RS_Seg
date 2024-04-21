import logging

import torch
from torch import nn
from torch import optim
import os
from torchvision.utils import save_image
import pandas as pd
from tqdm import tqdm

from metrics import Metrics

from pathlib import Path


logger = logging.getLogger('MMSA')
class UNetPlusPlus():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.BCELoss()

    def do_train(self, model, train_dataloader, test_dataloder, epoch, return_epoch_results=False):
        opt = optim.Adam(params=model.parameters(),lr=self.args.learning_rate)
        # epochs, best_epoch = 0, 0
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
                    image, segment_image = image.to(self.args.device), segment_image.to(self.args.device)
                    # 数据移动至设备
                    out_image = model(image).to(self.args.device)
                    # 前向计算
                    loss = self.criterion(out_image, segment_image)
                    # 计算损失
                    opt.zero_grad()
                    # 清空梯度
                    loss.backward()
                    # 反向计算
                    opt.step()
                    # 更新梯度

                    if i % 20 == 0:
                        # 设置阈值


                        _image = image[0]
                        _segment_image = segment_image[0]
                        _out_image = out_image[0]
                        if not self.args['is_color']:
                            threshold = 0.5
                            _out_image = torch.where(_out_image > threshold, torch.tensor(1.0), torch.tensor(0.0))

                        img = torch.stack([_image, _segment_image, _out_image], dim=0)
                        # 将每个batch里的第一张图进行拼接展示
                        save_dir = f"{self.args['image_save_path']}/{self.args['dataset_name']}"
                        Path(save_dir).mkdir(parents=True, exist_ok=True)
                        image_save_path = save_dir + "/" + f"train_{self.args['super_epoch']}_{epoch}_{t}.png"
                        save_image(img, image_save_path)
                        t += 1
                logger.info(f"super-epoch:{self.args['super_epoch']},train-epoch:{epoch}-train_loss===>>{round(loss.item(), 4)}")
                # 其实保存条件也应该设置一下
                # torch.save(model.state_dict(), self.args['model_save_path'])
                # logger.info('save weight:' + str(self.args['model_save_path']))
                if self.args['is_seeds']:
                    val_results = self.do_test(model, test_dataloder)
                    return val_results
                else:
                    return None

                # if epochs - best_epoch >= self.args.early_stop:
                #     return val_results if return_epoch_results else None

    def do_test(self, model, test_dataloader):
        model.eval()
        eval_results = {
            'loss':[],
            'pix_acc':[],
            'mean_iou':[],
            'freq_iou':[]
        }
        t = 0
        with torch.no_grad():
            with tqdm(test_dataloader) as data_loader:
                for i, (image, segment_image) in enumerate(data_loader):
                    image, segment_image = image.to(self.args['device']), segment_image.to(self.args['device'])
                    # 数据移动至设备
                    out_image = model(image).to(self.args['device'])
                    # 前向计算
                    train_loss = self.criterion(out_image, segment_image)
                    # 计算损失
                    eval_results['loss'].append(train_loss)

                    if not self.args['is_color']:
                        # 设置阈值
                        threshold = 0.5
                        # 阈值化操作
                        out_image = torch.where(out_image > threshold, torch.tensor(1.0), torch.tensor(0.0))


                    #展示照片
                    _image = image[0]
                    _segment_image = segment_image[0]
                    _out_image = out_image[0]
                    # 将每个batch里的第一张图进行拼接展示
                    img = torch.stack([_image, _segment_image, _out_image], dim=0)
                    save_dir = f"{self.args['image_save_path']}/{self.args['dataset_name']}"
                    Path(save_dir).mkdir(parents=True, exist_ok=True)
                    image_save_path = save_dir + "/" + f"test_{self.args['super_epoch']}_{t}.png"
                    save_image(img, image_save_path)

                    segment_image = torch.squeeze(torch.mean(segment_image, dim=1))

                    pix_acc, mean_iou, freq_iou = Metrics(out_image, segment_image)
                    eval_results['pix_acc'].append(pix_acc)
                    eval_results['mean_iou'].append(mean_iou)
                    eval_results['freq_iou'].append(freq_iou)
                    t += 1
            loss = sum(eval_results['loss']) / len(eval_results['loss'])
            logger.info(f"super-epoch:{self.args['super_epoch']}-test_loss===>>{round(loss.item(),4)}")
            return eval_results

    def set_args(self,args):
        self.args = args
        return
