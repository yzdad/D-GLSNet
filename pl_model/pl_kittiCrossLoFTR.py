from abc import ABC
from functools import partial
import cv2
import open3d as o3d
import torch

from torchvision import transforms
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader

import utils.factory as factory

from dataset.dataset_utils.collate_fn import *
from dataset.dataset_utils.worker_init_fn import worker_init_fn

from optimizers import build_optimizer, build_scheduler

from utils.supervision import compute_supervision_coarse_kitti, compute_supervision_fine_kitti
from utils.visualization import show_match_kitti_2d

def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('model size: {:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)

class pl_kittiCrossLoFTR(LightningModule, ABC):
    def __init__(self, cfg):
        print("\033[31m---------pl_kittiCrossLoFTR init----------\033[0m")
        super().__init__()
        self.cfg = cfg

        # data 工厂
        self.Dataset = factory.get_module('dataset', cfg['data']['dataset'])

        train_set = self.Dataset(task='train', **self.cfg['data'])
        neighbor_limits = calibrate_neighbors_stack_mode(
            train_set,
            registration_collate_fn_stack_mode,
            self.cfg['data']['num_stages'],
            self.cfg['data']['voxel_size'],
            self.cfg['data']['search_radius'],
        )
        print('Calibrate neighbors: {}.'.format(neighbor_limits))
        self.collate_fn = partial(
            registration_collate_fn_stack_mode,
            num_stages=self.cfg['data']['num_stages'],
            voxel_size=self.cfg['data']['voxel_size'],
            search_radius=self.cfg['data']['search_radius'],
            neighbor_limits=neighbor_limits,
            precompute_data=self.cfg['data']['precompute_data'],
        )

        # net 工厂
        self.net = factory.model_loader(
            model=cfg['model']["name"], **cfg['model'])
        getModelSize(self.net)
        # loss
        loss_list = self.cfg["loss"].keys()
        for i in loss_list:
            self.loss = factory.loss_loader(i, **self.cfg["loss"][i])

        print("\033[31m---------pl_kittiCrossLoFTR init finish----------\033[0m")
        print()

    def forward(self, x):
        outs = self.net(x)
        return outs

    def training_step(self, batch, batch_idx):
        batch['istraining'] = True
        compute_supervision_coarse_kitti(batch, self.cfg)

        self(batch)

        compute_supervision_fine_kitti(batch, self.cfg)

        self.loss(batch)

        self.logger.experiment.add_scalar(
            "loss", batch['loss'], self.global_step)
        # self.logger.experiment.add_scalar(
        #     "loss_c_pre", batch['loss_scalars']['loss_c_pre'], self.global_step)
        self.logger.experiment.add_scalar(
            "loss_c", batch['loss_scalars']['loss_c'], self.global_step)
        self.logger.experiment.add_scalar(
            "loss_f", batch['loss_scalars']['loss_f'], self.global_step)

        return {'loss': batch['loss']}

    def training_epoch_end(self, training_step_outputs):
        pass

    def validation_step(self, batch, batch_idx):
        batch['istraining'] = False
        compute_supervision_coarse_kitti(batch, self.cfg)

        self(batch)

        compute_supervision_fine_kitti(batch, self.cfg)

        self.loss(batch)

        return {'val_loss_step': batch['loss']}

    def validation_epoch_end(self, outputs):
        losses = []
        for i in outputs:
            losses.append(i["val_loss_step"])
        losses = torch.stack(losses)
        loss = torch.mean(losses)
        # save model
        self.log("val_loss", loss, sync_dist=True)  # 同时会写进记录器

    def test_step(self, batch, batch_idx):
        batch['istraining'] = False
        compute_supervision_coarse_kitti(batch, self.cfg)

        self(batch)

        compute_supervision_fine_kitti(batch, self.cfg)

        show_img = show_match_kitti_2d(batch)  # TODO
        cv2.imshow('image', show_img)
        cv2.waitKey()

    ##############################################################################################################################
    def configure_optimizers(self):
        # optimizer = torch.optim.Adam([{"params": self.parameters(), "initial_lr": self.cfg["learning_rate"]}], lr=
        # self.cfg["learning_rate"], betas=(0.9, 0.999), amsgrad=True)
        # return optimizer
        optimizer = build_optimizer(self, self.cfg['optimizer'])
        scheduler = build_scheduler(self.cfg['optimizer'], optimizer)
        return [optimizer], [scheduler]
        # return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp,
                       using_lbfgs):
        # learning rate warm up
        if self.trainer.global_step < self.cfg['optimizer']['warmup_step']:
            if self.cfg['optimizer']['warmup_type'] == 'linear':
                base_lr = self.cfg['optimizer']['warmup_ratio'] * \
                    self.cfg['optimizer']['ture_lr']
                lr = base_lr + \
                    (self.trainer.global_step / self.cfg['optimizer']['warmup_step']) * \
                    abs(self.cfg['optimizer']['ture_lr'] - base_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
            elif self.cfg['optimizer']['warmup_type'] == 'constant':
                pass
            else:
                raise ValueError(
                    f'Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}')

        # 打印学习率
        count = 0
        for pg in optimizer.param_groups:
            self.logger.experiment.add_scalar(
                "lr_" + str(count), pg['lr'], self.global_step)
            count += 1
        # update params
        optimizer.step(closure=optimizer_closure)

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad()
        # Set gradients to `None` instead of zero to improve performance. ???
        # optimizer.zero_grad(set_to_none=True)

    ####################
    # DATA RELATED HOOKS
    ####################
    def prepare_data(self):
        """
        Use this to download and prepare data.
        :return:
        """
        pass

    def setup(self, stage=None):
        """
        Called at the beginning of fit (train + validate), validate, test, or predict.
        This is a good hook when you need to build models dynamically or adjust something about them.
        This hook is called on every process when using DDP.
        :param stage:
        :return:
        """
        pass

    def train_dataloader(self):
        data_transform = transforms.Compose([
            transforms.ToTensor(),  # to Tensor
        ])
        train_set = self.Dataset(
            transform=data_transform, task='train', **self.cfg['data'])

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self.cfg['batch_size'],
            collate_fn=self.collate_fn,
            shuffle=True,
            pin_memory=True,
            num_workers=self.cfg.get('workers_train', 1),
            worker_init_fn=worker_init_fn,
            persistent_workers=True
        )  # collate_fn sampler
        print()
        print('\033[32m--->workers_train {0} split size {1} in {2} batches\033[0m'.format(
            self.cfg.get('workers_train', 1),
            len(train_loader) *
            self.cfg['batch_size'],
            len(train_loader)))

        return train_loader

    def val_dataloader(self):
        data_transform = transforms.Compose([
            transforms.ToTensor(),  # to Tensor
        ])
        train_set = self.Dataset(
            transform=data_transform, task='val', **self.cfg['data'])

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self.cfg['eval_batch_size'],
            collate_fn=self.collate_fn,
            shuffle=False,
            pin_memory=True,
            num_workers=self.cfg.get('workers_val', 1),
            worker_init_fn=worker_init_fn,
            persistent_workers=True  # 每个epoch的第一个iteration是否重新初始化worker
        )
        print()
        print('\033[32m--->workers_val {0} split size {1} in {2} batches\033[0m'.format(
            self.cfg.get('workers_val', 1),
            len(train_loader) * self.cfg['eval_batch_size'],
            len(train_loader)))

        return train_loader

    def test_dataloader(self):
        data_transform = transforms.Compose([
            transforms.ToTensor(),  # to Tensor
        ])
        train_set = self.Dataset(
            transform=data_transform, task='test', **self.cfg['data'])

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self.cfg['test_batch_size'],
            collate_fn=self.collate_fn,
            shuffle=False,
            pin_memory=True,
            num_workers=self.cfg.get('workers_test', 1),
            worker_init_fn=worker_init_fn,
            persistent_workers=True
        )
        print()
        print('\033[32m--->workers_train {0} split size {1} in {2} batches\033[0m'.format(
            self.cfg.get('workers_test', 1),
            len(train_loader) * self.cfg['test_batch_size'],
            len(train_loader)))

        return train_loader
