#    YZDAD (li laijian)
#       ┏┓     ┏┓
#     ┏━┛┻━━━━━┛┻━┓
#     ┃           ┃
#     ┃     ━     ┃
#     ┃  ┳┛   ┗┳  ┃
#     ┃           ┃
#     ┃     ┻     ┃
#     ┃           ┃
#     ┗━┓       ┏━┛  Codes are far away from bugs with the animal protecting
#       ┃       ┃    神兽保佑,代码无bug
#       ┃       ┃
#       ┃       ┗━━━┓
#       ┃           ┣┓
#       ┃　　　　    ┏┛
#       ┗┓┓┏━━━━━┳┓┏┛
#        ┃┫┫     ┃┫┫
#        ┗┻┛     ┗┻┛
import os
import shutil
import yaml
import argparse
from pathlib import Path

from models.DGLSNet import DGLSNet

import pprint
import torch
# lightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

import utils.get_path as gp


def parse_args():
    #
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest='command')
    command_list = [i.stem for i in list(Path('pl_model/').iterdir())]
    for i in command_list:
        if not i.find('__'):
            continue
        p_train = subparsers.add_parser(i)
        p_train.add_argument('config', type=str, default='config/default.yaml')
        p_train.add_argument('exper_name', type=str)
        p_train.add_argument('--eval', action='store_true')
        p_train.add_argument('--test', action='store_true')
        # p_train.add_argument('--debug', action='store_true', default=False, help='turn on debuging mode')
        mod = __import__('pl_model.{}'.format(i), fromlist=[''])
        p_train.set_defaults(model=getattr(mod, i))

    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()


if __name__ == '__main__':
    # global var
    torch.set_default_tensor_type(torch.FloatTensor)

    # add parser
    args = parse_args()
    print("\033[31m---------Parser {0}----------\033[0m".format(args.command))
    print("config file: {0}".format(args.config))
    print("\033[31m---------Parser {0} end------\033[0m".format(args.command))
    print()
    # rank_zero_only(pprint.pprint)(vars(args))  # 当线程执行数据美化输出

    # 加载和保存配置文件
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if args.test or args.eval:
        config['model']['crossMatch']['train'] = False
    else:
         config['model']['crossMatch']['train'] = True
    # TODO
    # args.gpus = _n_gpus = setup_gpus(args.gpus)  # GPU数
    # config.TRAINER.WORLD_SIZE = _n_gpus * args.num_nodes  # 全部设备 num_gpu * num_nodes???
    # config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size  # 总的batch数量
    # _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS  # 设置的batch与默认的比值
    # config.TRAINER.SCALING = _scaling
    # config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * _scaling  # 据据比值调整学习率
    # config.TRAINER.WARMUP_STEP = math.floor(config.TRAINER.WARMUP_STEP / _scaling)  # 调整warmup的阈值步伐

    #
    output_dir = os.path.join(config['run_path'], args.exper_name)
    os.makedirs(output_dir, exist_ok=True)
    if args.test:
        with open(os.path.join(output_dir, 'test.yml'), 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        with open(os.path.join(output_dir, 'config.yml'), 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    ####################################################################################################################
    torch.set_default_tensor_type(torch.FloatTensor)
    if config['pretrained']:  # 加载这个地方学习一些 tandem
        print("\033[32mpretrained: {0}\033[0m".format(config['pretrained']))
        state = torch.load(config['pretrained'])
        model = args.model(config)
        model.load_state_dict(state_dict=state['state_dict'], strict=False)
        # 下面这个写法训练会变慢
        # model = args.model.load_from_checkpoint(checkpoint_path=config['pretrained'], cfg=config,
        #                                         test_save_path=predictions_path)
    else:
        model = args.model(config)

    # 通过监控数量定期保存模型
    checkpoint_path = gp.get_checkpoint_path(output_dir)
    # shutil.rmtree(checkpoint_path)  # 删除文件
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", mode='min',
        dirpath=checkpoint_path,
        filename="sample-mnist-{epoch:02d}-{val_loss:.2f}",
        auto_insert_metric_name=True,
        save_top_k=10,
        save_last=True,
        every_n_epochs=1,
    )
    # 进度条
    bar = TQDMProgressBar(refresh_rate=10)
    # 记录器
    logger = TensorBoardLogger(gp.get_logger_path(output_dir), name="")
    # trainer
    trainer = Trainer(
        accelerator='gpu',
        devices=[0],  # gpus = max(1, torch.cuda.device_count()),
        max_epochs=config['max_epochs'],
        callbacks=[bar, checkpoint_callback],
        logger=logger,
        log_every_n_steps=10,  # Lightning 每 50 行或 50 个训练步骤记录一次
        # strategy="ddp",
        strategy="ddp_spawn",
    )
    # train val test
    if args.eval:
        if not config['pretrained']:
            print("no model get")
            exit()
        trainer.validate(model)
    elif args.test:
        if not config['pretrained']:
            print("no model get")
            exit()
        trainer.test(model)
    else:
        trainer.fit(model)
        # trainer.fit(model, ckpt_path="some/path/to/my_checkpoint.ckpt")  # 恢复完整的训练,和加载整个模型一样训练会变慢
