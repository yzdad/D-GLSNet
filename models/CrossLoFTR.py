import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import build_backbone_2d, build_backbone_3d
from .position_encoding import PositionEncodingSine, PositionEncodingSine_xy
from einops.einops import rearrange
from .transformer.transformer import LocalFeatureTransformer
from .coarse_matching import CoarseMatching
from .fine_preprocess import FinePreprocess
from .fine_matching import FineMatching
import time


class CrossLoFTR(nn.Module):
    def __init__(self, **config):
        super(CrossLoFTR, self).__init__()
        # fig
        self.cfg = config

        # Module
        self.image_backbone = build_backbone_2d(config['backbone'])
        self.pos_encoding_2d = PositionEncodingSine(config['coarseTransformer']['d_model'],
                                                    **config['coarseTransformer']['pos_encoding_2d'])

        self.point_backbone = build_backbone_3d(config['backbone'])
        self.pos_encoding_3d = PositionEncodingSine_xy(config['coarseTransformer']['d_model'])
        
        # ======================================================================================
        self.coarse_match = LocalFeatureTransformer(config['coarseTransformer'])
        self.coarse_matching = CoarseMatching(config['crossMatch'])

        self.fine_preprocess = FinePreprocess(config['fine_preprocess'])
        self.loftr_fine = LocalFeatureTransformer(config["fineTransformer"])
        self.fine_matching = FineMatching()

    def forward(self, data):
        torch.cuda.synchronize()
        start = time.time()
        #  ============================ backbone ================================
        # with torch.no_grad():  # 冻结
        feats_image_c, feats_image_f = self.image_backbone(data['image'])  # [B, C_c, h_c, w_c] and [B, C_f, h_f, w_f]
        feats_point_c, feats_point_f = self.point_backbone(feats=data['features'].detach(), data_dict=data)  # [B, N_c, C_c] and [B, N_f, C_f]

        data['hw_c'] = feats_image_c.shape[2:]
        data['hw_f'] = feats_image_f.shape[2:]
        data['scale_f_c'] = feats_image_f.shape[2] // feats_image_c.shape[2]
        data['scale_i_c'] = data['image'].shape[2] // feats_image_c.shape[2]
        data['scale_i_f'] = data['image'].shape[2] // feats_image_f.shape[2]
        # ============================ embedding ===============================
        feats_image_c = rearrange(feats_image_c, 'n c h w -> n (h w) c')  # [B, h_c*w_c, C_c]
        pose_image = self.pos_encoding_2d()  # [1, h_c * w_c, C_c]

        point_c = data['points'][-1].unsqueeze(0)
        pose_point = self.pos_encoding_3d(point_c, data)  # [1, N_c, C_c]

        # =========================== coarse-level ================================
        # feats_image_c = F.normalize(feats_image_c, dim=2)
        # feats_point_c = F.normalize(feats_point_c, dim=2)
        # self.coarse_matching_prev(feats_image_c, feats_point_c, data)
  
        feats_image_c, feats_point_c = self.coarse_match(feats_image_c, feats_point_c, pose_image, pose_point)

        feats_image_c = F.normalize(feats_image_c, dim=2)
        feats_point_c = F.normalize(feats_point_c, dim=2)
        self.coarse_matching(feats_image_c, feats_point_c, data)
        torch.cuda.synchronize()
        end = time.time()
        data['time_c'] = end - start

        # ======================== fine-level ====================================
        feat_image_unfold, feat_point_unfold = self.fine_preprocess(feats_image_f, feats_point_f, feats_image_c, feats_point_c, data)

        # at least one coarse level predicted
        if feat_image_unfold.shape[0] != 0:
            feat_image_unfold, feat_point_unfold = self.loftr_fine(feat_image_unfold, feat_point_unfold)

        self.fine_matching(feat_image_unfold, feat_point_unfold, data)
        torch.cuda.synchronize()
        end = time.time()
        data['time_f'] = end - start