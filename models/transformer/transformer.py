import copy
import torch
import torch.nn as nn
from .linear_attention import LinearAttention, FullAttention
from .factory import build_act_layer, build_dropout_layer


class LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message


class SelfEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(SelfEncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_pose_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_pose, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, key_pose, value = x, source, x_pose, source

        # multi-head attention
        query = self.q_proj(query).view(
            bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead,
                                    self.dim)  # [N, S, (H, D)]
        key_pose = self.k_pose_proj(key_pose).view(
            bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)

        message = self.attention(query, key + key_pose, value,
                                 q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(
            bs, -1, self.nhead * self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(LocalFeatureTransformer, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        encoder_layer = LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'])
        self.layers = []
        for i in self.layer_names:
            if i == 'self':
                self.layers.append(nn.ModuleList([copy.deepcopy(encoder_layer), copy.deepcopy(encoder_layer)]))
            elif i == 'cross':
                self.layers.append(copy.deepcopy(encoder_layer))
        self.layers = nn.ModuleList(self.layers)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat_image, feat_point, pose_image=None, pose_point=None, mask0=None, mask1=None):
        """
        Args:
            feat_image (torch.Tensor): [N, L, C]
            feat_point (torch.Tensor): [N, S, C]
            point_embeddings (torch.Tensor):

            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        assert self.d_model == feat_image.size(
            2), "the feature number of src and transformer must be equal"
        assert self.d_model == feat_point.size(
            2), "the feature number of src and transformer must be equal"

        if (pose_image is not None) and len(self.layer_names) > 0:
            feat_image = feat_image + pose_image
            feat_point = feat_point + pose_point

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                feat_image = layer[0](feat_image, feat_image, mask0, mask0)
                feat_point = layer[1](feat_point, feat_point, mask1, mask1)
            elif name == 'cross':
                feat_image = layer(feat_image, feat_point, mask0, mask1)
                feat_point = layer(feat_point, feat_image, mask1, mask0)
            else:
                raise KeyError

        return feat_image, feat_point


class SelfTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(SelfTransformer, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        encoder_layer = SelfEncoderLayer(config['d_model'], config['nhead'], config['attention'])
        self.layers = []
        for i in self.layer_names:
            if i == 'self':
                self.layers.append(nn.ModuleList([copy.deepcopy(encoder_layer), copy.deepcopy(encoder_layer)]))
            elif i == 'cross':
                self.layers.append(copy.deepcopy(encoder_layer))
        self.layers = nn.ModuleList(self.layers)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat_image, feat_point, pose_image=None, pose_point=None, mask0=None, mask1=None):
        """
        Args:
            feat_image (torch.Tensor): [N, L, C]
            feat_point (torch.Tensor): [N, S, C]
            point_embeddings (torch.Tensor):

            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        assert self.d_model == feat_image.size(
            2), "the feature number of src and transformer must be equal"
        assert self.d_model == feat_point.size(
            2), "the feature number of src and transformer must be equal"

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                feat_image = layer[0](feat_image, feat_image, pose_image, mask0, mask0)
                feat_point = layer[1](feat_point, feat_point, pose_point, mask1, mask1)
            elif name == 'cross':
                feat_image = layer(feat_image, feat_point, pose_point, mask0, mask1)
                feat_point = layer(feat_point, feat_image, pose_image, mask1, mask0)
            else:
                raise KeyError

        return feat_image, feat_point
