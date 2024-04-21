import argparse

from run import RS_run


def parse_args():
    # 解析命令行参数
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', type=str, default='unetplusplus', help='Name of model',
                        choices=['unet', 'unetplusplus', 'u2net', 'segnet', 'hrnet'])
    parser.add_argument('-d', '--dataset', type=str, default='forest',
                        choices=['berlin', 'zurich', 'chicago', 'water', 'forest'], help='Name of dataset')
    parser.add_argument('-c', '--config', type=str, default='',
                        help='Path to config file. If not specified, default config file will be used.')
    parser.add_argument('-t', '--tune', action='store_true', default=False,
                        help='Whether to tune hyper parameters. Default: False')
    parser.add_argument('-tt', '--tune-times', type=int, default=50,
                        help='Number of times to tune hyper parameters. Default: 50')
    parser.add_argument('-s', '--seeds', action='append', type=int, default=[],
                        help='Random seeds. Specify multiple times for multiple seeds. Default: [1111, 1112, 1113, 1114, 1115]')
    parser.add_argument('-n', '--num-workers', type=int, default=0,
                        help='Number of workers used to load data. Default: 4')
    parser.add_argument('-v', '--verbose', type=int, default=1,
                        help='Verbose level of stdout. 用于控制训练信息的输出级别 0 for error, 1 for info, 2 for debug. Default: 1')
    parser.add_argument('--model-save-dir', type=str, default='',
                        help='Path to save trained models. Default: "~/MMSA/saved_models"')
    parser.add_argument('--res-save-dir', type=str, default='',
                        help='Path to save csv results. Default: "~/MMSA/results"')
    parser.add_argument('--log-dir', type=str, default='',
                        help='Path to save log files. Default: "~/MMSA/logs"')
    parser.add_argument('-g', '--gpu-ids', action='append', default=[],
                        help='Specify which gpus to use. If an empty list is supplied, will automatically assign to the most memory-free gpu. \
                              Currently only support single gpu. Default: []')
    parser.add_argument('-e', '--super-epoch', type=int, default=0, help='')
    parser.add_argument('-cl', '--color', type=bool, default=False, help='')

    return parser.parse_args()


if __name__ == '__main__':
    # 使用parse_args()函数解析命令行参数，并将解析结果传递给MMSA_run()函数进行处理
    cmd_args = parse_args()
    RS_run(
        model_name=cmd_args.model,
        dataset_name=cmd_args.dataset,
        config_file=cmd_args.config,
        seeds=cmd_args.seeds,
        model_save_dir=cmd_args.model_save_dir,
        res_save_dir=cmd_args.res_save_dir,
        log_dir=cmd_args.log_dir,
        gpu_ids=cmd_args.gpu_ids,
        num_workers=cmd_args.num_workers,
        verbose_level=cmd_args.verbose,
        super_epoch=cmd_args.super_epoch,
        is_color=cmd_args.color
    )
