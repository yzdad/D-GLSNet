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
    # rank_zero_only(pprint.pprint)(vars(args))  # 

    # 
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if args.test or args.eval:
        config['model']['crossMatch']['train'] = False
    else:
         config['model']['crossMatch']['train'] = True
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
    if config['pretrained']:  #
        print("\033[32mpretrained: {0}\033[0m".format(config['pretrained']))
        state = torch.load(config['pretrained'])
        model = args.model(config)
        model.load_state_dict(state_dict=state['state_dict'], strict=False)

        # model = args.model.load_from_checkpoint(checkpoint_path=config['pretrained'], cfg=config,
        #                                         test_save_path=predictions_path)
    else:
        model = args.model(config)


    checkpoint_path = gp.get_checkpoint_path(output_dir)
    # shutil.rmtree(checkpoint_path)  # 
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", mode='min',
        dirpath=checkpoint_path,
        filename="sample-mnist-{epoch:02d}-{val_loss:.2f}",
        auto_insert_metric_name=True,
        save_top_k=10,
        save_last=True,
        every_n_epochs=1,
    )
    # 
    bar = TQDMProgressBar(refresh_rate=10)
    # 
    logger = TensorBoardLogger(gp.get_logger_path(output_dir), name="")
    # trainer
    trainer = Trainer(
        accelerator='gpu',
        devices=[0],  # gpus = max(1, torch.cuda.device_count()),
        max_epochs=config['max_epochs'],
        callbacks=[bar, checkpoint_callback],
        logger=logger,
        log_every_n_steps=10,  # Lightning
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
        # trainer.fit(model, ckpt_path="some/path/to/my_checkpoint.ckpt")  # 
