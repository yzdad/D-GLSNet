import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
import numpy as np

INF = 1e9


def mask_border(m, b: int, v):
    """ Mask borders with value
    Args:
        m (torch.Tensor): [N, Hc, Wc, nPc]
        b (int)
        v (m.dtype)
    """
    if b <= 0:
        return

    m[:, :b] = v
    m[:, :, :b] = v
    m[:, -b:] = v
    m[:, :, -b:] = v


def mask_border_with_padding(m, bd, v, p_m0, p_m1):
    if bd <= 0:
        return

    m[:, :bd] = v
    m[:, :, :bd] = v
    m[:, :, :, :bd] = v
    m[:, :, :, :, :bd] = v

    h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
    h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()
    for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
        m[b_idx, h0 - bd:] = v
        m[b_idx, :, w0 - bd:] = v
        m[b_idx, :, :, h1 - bd:] = v
        m[b_idx, :, :, :, w1 - bd:] = v


def compute_max_candidates(p_m0, p_m1):
    """Compute the max candidates of all pairs within a batch

    Args:
        p_m0, p_m1 (torch.Tensor): padded masks
    """
    h0s, w0s = p_m0.sum(1).max(-1)[0], p_m0.sum(-1).max(-1)[0]
    h1s, w1s = p_m1.sum(1).max(-1)[0], p_m1.sum(-1).max(-1)[0]
    max_cand = torch.sum(
        torch.min(torch.stack([h0s * w0s, h1s * w1s], -1), -1)[0])
    return max_cand


class CoarseMatching(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.temperature = config['temperature']
        self.thr = config['thr']
        self.border_rm = config['border_rm']
        # 训练fine match
        self.istraining = config['train']
        self.train_coarse_percent = config['train_coarse_percent']  # 匹配对百分比
        self.train_pad_num_gt_min = config['train_pad_num_gt_min']  # 最小的真值
        if self.config['match_type'] == 'sinkhorn':
            self.score = nn.Parameter(torch.tensor(0.13, requires_grad=True))

    def forward(self, feat_c0, feat_c1, data, mask_c0=None, mask_c1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C] 图像特征
            feat1 (torch.Tensor): [N, S, C] 点云特征
            data (dict)
            mask_c0 (torch.Tensor): [N, L] (optional)
            mask_c1 (torch.Tensor): [N, S] (optional)
        Update:

        """
        N, L, S, C = feat_c0.size(0), feat_c0.size(1), feat_c1.size(1), feat_c0.size(2)

        if self.config['match_type'] == 'sinkhorn':
            sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1)
            b, m, n = sim_matrix.shape
            bins0 = self.score.expand(b, 1, n)
            sim_matrix = torch.cat([sim_matrix, bins0], 1)
            conf_matrix = F.softmax(sim_matrix / self.temperature, 1)
        else:
            sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1) / self.temperature
            # F.softmax(sim_matrix, 1) *   点对patch是会一对多的
            conf_matrix = F.softmax(sim_matrix, 1)

        data.update({'conf_matrix': conf_matrix})

        data.update(**self.get_coarse_match(conf_matrix, data))

    @torch.no_grad()
    def get_coarse_match(self, conf_matrix, data):
        """
        Args:
            conf_matrix (torch.Tensor): [N, L, S]
            data (dict): with keys ['hw_i', 'hw_f', 'hw_c']
        Returns:
            coarse_matches (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'm_bids' (torch.Tensor): [M],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
        """
        axes_lengths = {
            'hc': data['hw_c'][0],
            'wc': data['hw_c'][1],
            'npc': conf_matrix.shape[-1]
        }
        _device = conf_matrix.device

        # 1. confidence thresholding
        mask = conf_matrix > self.thr
        mask_ = rearrange(mask[:, 0:-1, :],
                          'b (hc wc) npc -> b hc wc npc',  **axes_lengths)
        mask_border(mask_, self.border_rm, False)
        mask[:, 0:-1, :] = rearrange(mask_,
                                     'b hc wc npc -> b (hc wc) npc', **axes_lengths).clone()

        # 2. mutual nearest
        mask = mask * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])

        # 3. find all valid coarse matches
        # 找每一列的最值，就是找true那个
        b_ids, i_ids, p_ids = torch.where(mask[:, 0:-1, :])
        mconf = conf_matrix[b_ids, i_ids, p_ids]

        # 4. Random sampling of training samples for fine-level match
        # (optional) pad samples with gt coarse-level matches
        if data['istraining']:
            pass
            # NOTE:
            # The sampling is performed across all pairs in a batch without manually balancing
            # #samples for fine-level increases w.r.t. batch_size

            num_candidates_max = mask.size(0) * mask.size(2)  # 最大匹配数
            num_matches_train = int(
                num_candidates_max * self.train_coarse_percent)  # 用于fine训练的匹配数
            num_matches_pred = len(b_ids)
            if self.train_pad_num_gt_min > num_matches_train:
                num_matches_train += (self.train_pad_num_gt_min -
                                      num_matches_train) * 2

            # pred_indices is to select from prediction
            if num_matches_pred <= num_matches_train - self.train_pad_num_gt_min:
                pred_indices = torch.arange(num_matches_pred, device=_device)
            else:
                pred_indices = torch.randint(
                    num_matches_pred,  (num_matches_train - self.train_pad_num_gt_min,), device=_device)

            # gt_pad_indices is to select from gt padding. e.g. max(3787-4800, 200)
            gt_pad_indices = torch.randint(len(data['spv_b_ids']),  (max(
                num_matches_train - num_matches_pred,  self.train_pad_num_gt_min), ),    device=_device)
            # set conf of gt paddings to all zero
            mconf_gt = torch.zeros(len(data['spv_b_ids']), device=_device)

            b_ids, i_ids, p_ids, mconf = map(lambda x, y: torch.cat([x[pred_indices], y[gt_pad_indices]],  dim=0),
                                             *zip([b_ids, data['spv_b_ids']], [i_ids, data['spv_i_ids']],   [p_ids, data['spv_p_ids']], [mconf, mconf_gt]))

        # These matches select patches that feed into fine-level network
        coarse_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'p_ids': p_ids}

        # 4. Update with matches in original image resolution
        scale_i_c = data['scale_i_c']
        uv_c_in_i = torch.stack([i_ids % data['hw_c'][1], i_ids //
                                 data['hw_c'][1]],  dim=1) * scale_i_c + scale_i_c / 2
        point_c_match = data['points'][-1][p_ids]

        # These matches is the current prediction (for visualization)
        coarse_matches.update({
            # 'gt_mask': mconf == 0,  # 标记gt点
            'uv_c_in_i': uv_c_in_i,  # 图像匹配点的坐标位置
            'point_c_match': point_c_match,  # 点云匹配点
            'point_c_match_gt': data['spv_uv_i'][p_ids]
            # 'mconf': mconf
        })

        return coarse_matches


class CoarseMatchingPrev(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.temperature = config['temperature']
        if self.config['match_type'] == 'sinkhorn':
            self.score = nn.Parameter(torch.tensor(
                0.13, requires_grad=True))  # TODO 0.1124  0.1801

    def forward(self, feat_c0, feat_c1, data, mask_c0=None, mask_c1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            data (dict)
            mask_c0 (torch.Tensor): [N, L] (optional)
            mask_c1 (torch.Tensor): [N, S] (optional)
        Update:

        """
        if self.config['match_type'] == 'sinkhorn':
            sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1)
            b, m, n = sim_matrix.shape
            bins0 = self.score.expand(b, 1, n)
            sim_matrix = torch.cat([sim_matrix, bins0], 1)
            conf_matrix = F.softmax(sim_matrix / self.temperature, 1)
        else:
            sim_matrix = torch.einsum(
                "nlc,nsc->nls", feat_c0, feat_c1) / self.temperature
            # F.softmax(sim_matrix, 1) *   点对patch是会一对多的
            conf_matrix = F.softmax(sim_matrix, 1)

        data.update({'conf_matrix_prev': conf_matrix})


def get_coarse_match_numpy(data):
    conf_matrix = data['conf_matrix'][:, :-1, :].squeeze()
    conf_max, index_i = torch.max(conf_matrix, 0)

    conf_max, index_i = conf_max.cpu().numpy(), index_i.cpu().numpy()
    index_p = np.linspace(
        0, conf_matrix.shape[1] - 1, conf_matrix.shape[1]).astype(np.int32)
    index_th = np.where(conf_max > 0.2)
    index_i, index_p = index_i[index_th], index_p[index_th]
    return index_i, index_p
