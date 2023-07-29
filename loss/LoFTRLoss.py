from loguru import logger

import torch
import torch.nn as nn


class LoFTRLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.loss_config = config

        # self.match_type = config['match_type']
        self.sparse_spvs = config['sparse_spvs']

        # coarse-level
        self.c_pos_w = self.loss_config['pos_weight']
        self.c_neg_w = self.loss_config['neg_weight']

        # fne-level
        self.fine_type = self.loss_config['fine_type']
        self.correct_thr = self.loss_config['correct_thr']

    def compute_coarse_loss(self, conf, conf_gt):
        """ Point-wise CE / Focal Loss with 0 / 1 confidence as gt.
        Args:
            conf (torch.Tensor): (N, hw, num_p) / (N, hw+1, num_p+1)
            conf_gt (torch.Tensor): (N, hw, num_p)
            weight (torch.Tensor): (N, hw, num_p)
        """
        pos_mask, neg_mask = conf_gt > 0, conf_gt == 0  # mask
        c_pos_w, c_neg_w = self.c_pos_w, self.c_neg_w
        # corner case: no gt coarse-level match at all
        if not pos_mask.any():  # assign a wrong gt
            pos_mask[0, 0, 0] = True
            c_pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            c_neg_w = 0.

        if self.loss_config['coarse_type'] == 'cross_entropy':

            conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
            loss_pos = - conf_gt[pos_mask] * torch.log(conf[pos_mask])
            return c_pos_w * loss_pos.mean()
        elif self.loss_config['coarse_type'] == 'focal':
            conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
            alpha = self.loss_config['focal_alpha']
            gamma = self.loss_config['focal_gamma']

            if self.sparse_spvs:
                pos_conf = conf[pos_mask]
                loss_pos = - alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()
                loss = c_pos_w * loss_pos.mean()
                return loss
            else:  # dense supervision
                loss_pos = - alpha * torch.pow(1 - conf[pos_mask], gamma) * (conf[pos_mask]).log()
                loss_neg = - alpha * torch.pow(conf[neg_mask], gamma) * (1 - conf[neg_mask]).log()
                return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
        else:
            raise ValueError('Unknown coarse loss: {type}'.format(type=self.loss_config['coarse_type']))

    def compute_fine_loss(self, expec_f, expec_f_gt):
        if self.fine_type == 'l2_with_std':
            return self._compute_fine_loss_l2_std(expec_f, expec_f_gt)
        elif self.fine_type == 'l2':
            return self._compute_fine_loss_l2(expec_f, expec_f_gt)
        else:
            raise NotImplementedError()

    def _compute_fine_loss_l2(self, expec_f, expec_f_gt):
        """
        Args:
            expec_f (torch.Tensor): [M, 2] <x, y>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
        """
        correct_mask = torch.linalg.norm(expec_f_gt, ord=float('inf'), dim=1) < self.correct_thr
        if correct_mask.sum() == 0:
            if self.training:  # this seldomly happen when training, since we pad prediction with gt
                logger.warning("assign a false supervision to avoid ddp deadlock")
                correct_mask[0] = True
            else:
                return None
        offset_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask]) ** 2).sum(-1)
        return offset_l2.mean()

    def _compute_fine_loss_l2_std(self, expec_f, expec_f_gt):
        """
        Args:
            expec_f (torch.Tensor): [M, 3] <x, y, std>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
        """
        # correct_mask tells you which pair to compute fine-loss
        correct_mask = torch.linalg.norm(expec_f_gt, ord=float('inf'), dim=1) < self.correct_thr

        # use std as weight that measures uncertainty
        std = expec_f[:, 2]
        inverse_std = 1. / torch.clamp(std, min=1e-10)
        weight = (inverse_std / torch.mean(inverse_std)).detach()  # avoid minizing loss through increase std

        # corner case: no correct coarse match found
        if not correct_mask.any():
            if self.training:  # this seldomly happen during training, since we pad prediction with gt
                # sometimes there is not coarse-level gt at all.
                logger.warning("assign a false supervision to avoid ddp deadlock")
                correct_mask[0] = True
                weight[0] = 0.
            else:
                return None

        # l2 loss with std
        offset_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask, :2]) ** 2).sum(-1)
        loss = (offset_l2 * weight[correct_mask]).mean()

        return loss

    def forward(self, data):
        """
        Update:
            data (dict): update{
                'loss': [1] the reduced loss across a batch,
                'loss_scalars' (dict): loss scalars for tensorboard_record
            }
        """
        loss_scalars = {}
        # 0. prev-level loss
        # loss_c_prev = self.compute_coarse_loss(data['conf_matrix_prev'], data['conf_matrix_gt'])

        # 1. coarse-level loss
        loss_c = self.compute_coarse_loss(data['conf_matrix'], data['conf_matrix_gt'])
        loss = loss_c * self.loss_config['coarse_weight']
        loss_scalars.update({"loss_c": loss_c.clone().detach().cpu()})

        # loss_c_pre = self.compute_coarse_loss(data['conf_matrix_prev'], data['conf_matrix_gt'])
        # loss += loss_c_pre * self.loss_config['pre_weight']
        # loss_scalars.update({"loss_c_pre": loss_c_pre.clone().detach().cpu()})

        # 2. fine-level loss
        loss_f = self.compute_fine_loss(data['expec_f'], data['expec_f_gt'])
        if loss_f is not None:
            loss += loss_f * self.loss_config['fine_weight']
            loss_scalars.update({"loss_f":  loss_f.clone().detach().cpu()})
        else:
            assert self.training is False
            loss_scalars.update({'loss_f': torch.tensor(1.)})  # 1 is the upper bound

        # 3. loss out
        # loss = loss_c * self.loss_config['coarse_weight'] + 0.1 * loss_c_prev * self.loss_config['coarse_weight']
        # loss_scalars.update(
        #     {"loss_c": loss_c.clone().detach().cpu(), "loss_c_prev": loss_c_prev.clone().detach().cpu()})

        data.update({"loss": loss, "loss_scalars": loss_scalars})
