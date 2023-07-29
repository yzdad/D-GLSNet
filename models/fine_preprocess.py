import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange, repeat


class FinePreprocess(nn.Module):
    def __init__(self, config, d_model=256):
        super().__init__()

        self.config = config
        self.cat_c_feat = config['fine_concat_coarse_feat']
        self.W = self.config['fine_window_size']

        d_model_c = self.config['d_model']
        d_model_f = self.config['d_model_f']
        self.d_model_f = d_model_f
        if self.cat_c_feat:
            self.down_proj = nn.Linear(d_model_c, d_model_f, bias=True)
            self.merge_feat = nn.Linear(2*d_model_f, d_model_f, bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def forward(self, feat_image_f, feat_point_f, feat_image_c, feat_point_c, data):
        """
        args:
            feat_image_f: 图像细特征 shape[B, C, Hf, Wf]
            feat_image_c: [B, Hc*Wc, C]
        """
        W = self.W
        data.update({'W': W})
        stride = data['scale_f_c']

        if data['b_ids'].shape[0] == 0:  # 
            feat_image = torch.empty(
                0, self.W**2, self.d_model_f, device=feat_image_f.device)  # [0, W^2, d_modle_f]
            feat_point = torch.empty(
                0, self.W**2, self.d_model_f, device=feat_image_f.device)
            return feat_image, feat_point

        # 1. unfold(crop) all local windows
        # print(feat_image_f.shape)
        feat_image_unfold = F.unfold(feat_image_f, kernel_size=(
            W, W), stride=stride, padding=W//2 - stride//2)  #  [B, c ww, l]
        feat_image_unfold = rearrange(
            feat_image_unfold, 'n (c ww) l -> n l ww c', ww=W**2)

        # print(feat_point_f.shape)
        feat_point_f = torch.cat(
            (feat_point_f, torch.zeros_like(feat_point_f[:1, :1, :])), 1)
        # print(feat_point_f.shape)
        feat_point_unfold = feat_point_f.index_select(
            1, data['subsampling'][3].view(-1))
        feat_point_unfold = rearrange(
            feat_point_unfold, 'n (p b) c -> n p b c', b=data['subsampling'][3].shape[1])

        # 2. select only the predicted matches
        # [n, ww, cf]
        # print(feat_image_unfold.shape, feat_point_unfold.shape)
        feat_image_unfold = feat_image_unfold[data['b_ids'].long(
        ), data['i_ids'].long()]
        feat_point_unfold = feat_point_unfold[data['b_ids'].long(
        ), data['p_ids'].long()]
        # print(feat_image_unfold.shape, feat_point_unfold.shape)

        # option: use coarse-level  feature as context: concat and linear
        if self.cat_c_feat:
            # [n_i+n_p, c]
            feat_image_c_ = self.down_proj(
                feat_image_c[data['b_ids'].long(), data['i_ids'].long()]).unsqueeze(1).repeat(1, W**2, 1)
            feat_point_c_ = self.down_proj(
                feat_point_c[data['b_ids'].long(), data['p_ids'].long()]).unsqueeze(1).repeat(1, data['subsampling'][3].shape[1], 1)
            # print(feat_image_c_, feat_point_c_)
            feat_image_unfold = self.merge_feat(
                torch.cat([feat_image_unfold,  feat_image_c_], -1))
            feat_point_unfold = self.merge_feat(
                torch.cat([feat_point_unfold,  feat_point_c_], -1))
        # print(feat_image_unfold.shape, feat_point_unfold.shape)

        return feat_image_unfold, feat_point_unfold