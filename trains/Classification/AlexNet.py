import logging
import torch
from torch import nn
from torch import optim
import sys
import os
from torchvision.utils import save_image
import pandas as pd
from tqdm import tqdm
from utils import Metrics
from pathlib import Path

logger = logging.getLogger('RS')


class AlexNet():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.CrossEntropyLoss()

    def do_train(self, model, train_dataloader, test_dataloder, return_epoch_results=False):
        epoch, best_epoch= 0, 0
        model.train()
        opt = optim.Adam(params=model.parameters(), lr=self.args.learning_rate)
        if return_epoch_results:
            train_losses = []
            train_acces = []
            eval_losses = []
            eval_acces = []
        best_valid = 0
        while True:
            model.train()
            train_loss = 0
            train_acc = 0
            with tqdm(train_dataloader, file=sys.stdout) as train_dataloader:
                for i, (image, labels) in enumerate(train_dataloader):
                    image = image.to(self.args.device)
                    labels = labels.to(self.args.device)
                    # 数据移动至设备
                    outputs = model(image).to(self.args.device)
                    # 前向计算
                    loss = self.criterion(outputs, labels)
                    # 计算损失
                    opt.zero_grad()
                    # 清空梯度
                    loss.backward()
                    # 反向计算
                    opt.step()
                    # 更新梯度
                    train_loss += loss.item()
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc = torch.eq(predict_y, labels).sum().item()
                    train_acc += acc

            # print(len(train_dataloader),len(train_dataloader))
            # print(train_loss, train_acc)
            train_loss = train_loss / (len(train_dataloader)* self.args.batch_size)
            train_acc = train_acc / (len(train_dataloader)*self.args.batch_size)
            epoch +=1
            logger.info(
                f"train-epoch:{epoch}-train_loss===>>{round(loss.item(), 4)}")

            if return_epoch_results:
                train_losses.append(train_loss)
                train_acces.append(train_acc)
                val_results = self.do_test(model, test_dataloder, epoch)
                eval_acces.append(val_results["val_acc"])
                eval_losses.append(val_results["val_loss"])

            # save best model
            cur_valid = val_results["val_acc"]
            isBetter = cur_valid > best_valid
            # save best model
            if isBetter:
                best_valid, best_epoch = cur_valid, epoch
                # save model
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)

            if epoch - best_epoch >= self.args.early_stop:
                if return_epoch_results:
                    result = {
                        "train_loss": train_losses,
                        "train_acc":train_acces,
                        "eval_acc": eval_acces,
                        "eval_loss": eval_losses
                    }
                    return result
                else:
                    return None

    def do_test(self, model, test_dataloader, epoch=0):
        model.eval()
        with torch.no_grad():
            eval_loss = 0
            eval_acc = 0
            with tqdm(test_dataloader, file=sys.stdout) as test_dataloader:
                for i, (image, labels) in enumerate(test_dataloader):
                    image = image.to(self.args.device)
                    labels = labels.to(self.args.device)
                    outputs = model(image).to(self.args.device)
                    loss = self.criterion(outputs, labels)
                    eval_loss += loss.item()
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc = torch.eq(predict_y, labels.to(self.args.device)).sum().item()
                    eval_acc += acc

            val_loss = eval_loss / (len(test_dataloader)*self.args.batch_size)
            val_acc = eval_acc /(len(test_dataloader)*self.args.batch_size)
            result = {
                "val_loss": val_loss,
                "val_acc": val_acc
            }
            logger.info(
                f"test-epoch:{epoch}-test_acc===>>{round(val_acc, 4)}")
            return result


    def set_args(self, args):
        self.args = args
        return
