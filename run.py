import errno
import logging
import gc
import time
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader

from data_loader import *
from net import *

import os
from pathlib import Path

from config import get_config_regression
from utils import setup_seed, count_parameters, assign_gpu
from data_loader import MyDataset
from models import AMIO
from trains import ATIO

logger = logging.getLogger('RS')

def _set_logger(log_dir, model_name, dataset_name, verbose_level):

    # base logger
    log_file_path = Path(log_dir) / f"{model_name}-{dataset_name}.log"
    logger = logging.getLogger('RS')
    logger.setLevel(logging.DEBUG)

    # file handler
    # 设定文件输出方法和等级
    fh = logging.FileHandler(log_file_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # stream handler
    # 用于将日志消息输出到控制台。
    stream_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
    ch = logging.StreamHandler()
    ch.setLevel(stream_level[verbose_level])
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger


# 先不写调参的部分
def RS_run(model_name: str, dataset_name: str, config_file: str = "",
        config: dict = None, seeds: list = [],
        num_workers: int = 0, verbose_level: int = 1,
        gpu_ids:list=[0],
        model_save_dir: str = Path(__file__).parent / "saved_models",
        res_save_dir: str = Path(__file__).parent / "res_save_dir",
        log_dir: str = Path(__file__).parent / "logs",
        super_epoch: int = 0,
        is_seeds: bool = False,
        is_color:bool = False):

    # Initialization
    model_name = model_name.lower()
    dataset_name = dataset_name.lower()

    # about dir
    if config_file != "":
        config_file = Path(config_file)
    else:
        config_file = Path(__file__).parent / "configs" / "config_regression.json"
    if not config_file.is_file():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_file)
    if model_save_dir == "":
        model_save_dir = Path(__file__).parent / "model_save_dir" / model_name / dataset_name
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    if res_save_dir == "":
        res_save_dir = Path(__file__).parent / "res_save_dir"
    Path(res_save_dir).mkdir(parents=True, exist_ok=True)
    if log_dir == "":
        log_dir = Path(__file__).parent / "logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    seeds = seeds if seeds !=[] else [1111
                                      # ,2222, 3333, 4444
                                      ]

    logger = _set_logger(log_dir, model_name, dataset_name, verbose_level)
    logger.info("======================================== Program Start ========================================")

    args = get_config_regression(model_name, dataset_name, config_file)
    args['is_color'] = is_color
    args['is_seeds'] = is_seeds
    args['super_epoch']=super_epoch
    if is_color:
        args['model_save_path'] = Path(model_save_dir) / f"{args['model_name']}-{args['dataset_name']}-CL-{super_epoch}.pth"
    else:
        args['model_save_path'] = Path(
            model_save_dir) / f"{args['model_name']}-{args['dataset_name']}-{super_epoch}.pth"
    # args['device'] = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    args['device'] = assign_gpu(gpu_ids)
    args['image_save_path'] = Path(__file__).parent / 'train_image'
    # if config:  # override some arguments 这个或许有用到时候再改
    #     if config.get(model_name):
    #         assert (config['model_name'] == args['model_name'])
    #     args.update(config)

    torch.cuda.set_device(args['device'])
    # 无gpu请注释上一行代码

    logger.info("Running with args:")
    logger.info(args)
    logger.info(f"Seeds: {seeds}")

    model_results = []
    if is_seeds:
        for i, seed in enumerate(seeds):

            args['cur_seed'] = i + 1
            logger.info(f"{'-' * 30} Running with seed {seed} [{i + 1}/{len(seeds)}] {'-' * 30}")
            result = _run(args,num_workers)
            logger.info(f"Result for seed {seed}: {result}")
            model_results.append(result)
        criterions = list(model_results[0].keys())
        # save result to csv
        csv_file = res_save_dir / f"{args['model_name']}-{args['dataset_name']}.csv"
        if csv_file.is_file():
            df = pd.read_csv(csv_file)
        else:
            df = pd.DataFrame(columns=["Model","data"] + criterions)
        # save results
        res = [model_name]
        res.append(f"{args['dataset_name']}_{args['super_epoch']}")
        for c in criterions:
            values = [r[c].data.cpu().numpy() for r in model_results]
            mean = round(np.mean(values) * 100, 2)
            std = round(np.std(values) * 100, 2)
            res.append((mean, std))
        df.loc[len(df)] = res
        df.to_csv(csv_file, index=None)
        logger.info(f"Results saved to {csv_file}.")
    else:
        seed = seeds[0]
        setup_seed(seed)
        data_loader = MyDataset(args)
        # 划分数据集为训练集和测试集
        train_dataset, test_dataset = data_loader.split_dataset(split_ratio=args['train_ratio'])
        # 加载数据集
        train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=True)

        model = AMIO(args).to(args['device'])
        logger.info(f'The model has {count_parameters(model)} trainable parameters')
        trainer = ATIO().getTrain(args)

        epoch = 0
        if os.path.exists(args['model_save_path']):
            model.load_state_dict(torch.load(args['model_save_path']))
            logger.info('successful load weight')
        else:
            logger.info('not successful load weight')

        super_epoch += 1
        args['super_epoch'] = super_epoch
        if is_color:
            args['model_save_path'] = Path(
                model_save_dir) / f"{args['model_name']}-{args['dataset_name']}-CL-{super_epoch}.pth"
        else:
            args['model_save_path'] = Path(
                model_save_dir) / f"{args['model_name']}-{args['dataset_name']}-{super_epoch}.pth"
        trainer.set_args(args)
        while True:
            model.train()
            while epoch < args['max_epoch']:
                logger.info(f'{epoch}/{args["max_epoch"]}')
                trainer.do_train(model, train_dataloader, test_dataloader, epoch)
                epoch += 1
            torch.save(model.state_dict(), args['model_save_path'])
            logger.info('save weight:' + str(args['model_save_path']))
            model.eval()
            model_results = trainer.do_test(model, test_dataloader)
            epoch = 0

            csv_file = res_save_dir / f"{args['model_name']}.csv"
            if os.path.isfile(csv_file):
                df = pd.read_csv(csv_file)
            else:
                df = pd.DataFrame(columns=['modal', 'data', 'loss', 'pix_acc', 'mean_iou', 'freq_iou'])
            res = [f"{args['model_name']}"]
            if is_color:
                res.append(f"{args['dataset_name']}_CL_{args['super_epoch']}")
            else:
                res.append(f"{args['dataset_name']}_BW_{args['super_epoch']}")
            criterions = list(model_results.keys())
            for c in criterions:
                # values = model_results[c]
                values = []
                for tensor in model_results[c]:
                    tensor_cpu = tensor.cpu().numpy()
                    values.append(tensor_cpu)
                mean = round(np.mean(values) * 100, 2)
                std = round(np.std(values) * 100, 2)
                res.append((mean, std))
            df.loc[len(df)] = res
            df.to_csv(csv_file, index=None)

            super_epoch += 1
            args['super_epoch'] = super_epoch
            args['model_save_path'] = Path(model_save_dir) / f"{args['model_name']}-{args['dataset_name']}-{super_epoch}.pth"
            trainer.set_args(args)

        # 其实都运行不到这里，可以设置一个条件
        del model
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(1)


def _run(args, num_workers=4, is_tune=False):
    data_loader = MyDataset(args)
    # 划分数据集为训练集和测试集
    train_dataset, test_dataset = data_loader.split_dataset(split_ratio=args['train_ratio'])
    # 加载数据集
    train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=True)

    model = AMIO(args).to(args['device'])
    logger.info(f'The model has {count_parameters(model)} trainable parameters')
    trainer = ATIO().getTrain(args)
    # train
    trainer.do_train(model, train_dataloader, test_dataloader)
    # test
    results = trainer.do_test(model, test_dataloader)
    # 如果只有最后测试的时候需要
    del model
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(1)
    return results

