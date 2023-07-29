import math
import torch
import torch.nn as nn

from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid


class FineMatching(nn.Module):
    """FineMatching with s2d paradigm"""

    def __init__(self):
        super().__init__()

    def forward(self, feat_image, feat_point, data):
        """
        Args:
            feat0 (torch.Tensor): [M, WW, C]
            feat1 (torch.Tensor): [M, PN, C]
            data (dict)
        Update:
            data (dict):{
                'expec_f' (torch.Tensor): [M, 3],
                'mkpts0_f' (torch.Tensor): [M, 2],
                'mkpts1_f' (torch.Tensor): [M, 2]}
        """
        M, WW, C = feat_image.shape
        W = int(math.sqrt(WW))  # 图片窗口
        scale = data['scale_i_f']  # 图片到粗特征的尺度

        self.M, self.W, self.WW, self.C, self.scale = M, W, WW, C, scale

        # corner case: if no coarse matches found
        if M == 0:
            assert self.training == False, "M is always >0, when training, see coarse_matching.py"
            # logger.warning('No matches found in coarse-level.')
            data.update({
                'expec_f': torch.empty(0, 3, device=feat_image.device),
                'uv_f_in_i': data['uv_c_in_i'], 
            })
            return

        feat_point_picked = feat_point[:, 0, :]  # 取最近点的特征
        sim_matrix = torch.einsum(
            'mc,mrc->mr', feat_point_picked, feat_image)  # [m, ww]
        softmax_temp = 1. / C**.5
        heatmap = torch.softmax(
            softmax_temp * sim_matrix, dim=1).view(-1, W, W)

        # compute coordinates from heatmap
        coords_normalized = dsnt.spatial_expectation2d(
            heatmap[None], True)[0]  # [M, 2] (x,y) 得到期望的坐标
        grid_normalized = create_meshgrid(
            W, W, True, heatmap.device).reshape(1, -1, 2)  # [1, WW, 2]

        # compute std over <x, y>  平方的期望减掉期望的平方
        var = torch.sum(grid_normalized**2 * heatmap.view(-1,
                        WW, 1), dim=1) - coords_normalized**2  # [M, 2]
        # [M]  clamp needed for numerical stability
        std = torch.sum(torch.sqrt(torch.clamp(var, min=1e-10)), -1)

        # for fine-level supervision
        data.update({'expec_f': torch.cat(
            [coords_normalized, std.unsqueeze(1)], -1)})

        # compute absolute kpt coords
        self.get_fine_match(coords_normalized, data)

    @torch.no_grad()
    def get_fine_match(self, coords_normed, data):

        # mkpts0_f and mkpts1_f
        mkpts0_f = data['uv_c_in_i'] + \
            (coords_normed * (self.W // 2) * self.scale)

        data.update({
            "uv_f_in_i": mkpts0_f,
        })