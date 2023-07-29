import math
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from einops.einops import rearrange

class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, shape=(30, 40)):
        super().__init__()

        pe = torch.zeros((d_model, *shape))  # [C, H, W]
        y_position = torch.ones(shape).cumsum(0).float().unsqueeze(0)  # [1, H, W]
        x_position = torch.ones(shape).cumsum(1).float().unsqueeze(0)  # [1, H, W]

        div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() * (-math.log(10000.0) / (d_model // 2)))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.sin(y_position * div_term)
        pe[2::4, :, :] = torch.cos(x_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer('pe', rearrange(pe.unsqueeze(0), 'n c h w -> n (h w) c'), persistent=False)  # [1, H*W, C]

    def forward(self):
        return self.pe

class PositionEncodingSine_xy(nn.Module):

    def __init__(self, d_model):
        """
        Args:
        """
        super().__init__()
        self.d_model = d_model

    def forward(self, point, data):

        pe = torch.zeros((self.d_model, point.shape[1]), device=point.device)  # [C, N]
        x_position = point[..., 0] / data['scale'][0] / data['scale_i_c'] +  data['hw_c'][1]  # [1, N]
        y_position = -point[..., 1] / data['scale'][0] / data['scale_i_c'] + data['hw_c'][0]  # [1, N] 

        div_term = torch.exp(torch.arange(0, self.d_model // 2, 2).float() * (-math.log(10000.0) / (self.d_model // 2))).to(point.device)
        div_term = div_term[:, None]  # [C//4, 1]
        pe[0::4, :] = torch.sin(x_position * div_term)
        pe[1::4, :] = torch.sin(y_position * div_term)
        pe[2::4, :] = torch.cos(x_position * div_term)
        pe[3::4, :] = torch.cos(y_position * div_term)

        return pe.unsqueeze(0).transpose(1, 2)  # [B, C, N] -> [B, N, C]


class PositionEncodingLearn(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, shape=(30, 40)):
        super(PositionEncodingLearn, self).__init__()

        y_position = torch.ones(shape).cumsum(0).float()  # [1, max_shape]  
        x_position = torch.ones(shape).cumsum(1).float()

        self.xy = (torch.stack([x_position, y_position], dim=-1).view(1, -1, 2).contiguous() - 0.5).detach()

        self.embed = PositionEmbeddingLearned(2, d_model)

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return self.embed(self.xy.to(x))


class GeometricStructureEmbedding(nn.Module):
    def __init__(self, hidden_dim, sigma_d, sigma_a, angle_k, reduction_a='max'):
        super(GeometricStructureEmbedding, self).__init__()
        self.sigma_d = sigma_d
        self.sigma_a = sigma_a
        self.factor_a = 180.0 / (self.sigma_a * np.pi)
        self.angle_k = angle_k

        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.proj_d = nn.Linear(hidden_dim, hidden_dim)
        self.proj_a = nn.Linear(hidden_dim, hidden_dim)

        self.reduction_a = reduction_a
        if self.reduction_a not in ['max', 'mean']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction_a}.')

    @torch.no_grad()
    def get_embedding_indices(self, points):
        r"""Compute the indices of pair-wise distance embedding and triplet-wise angular embedding.

        Args:
            points: torch.Tensor (B, N, 3), input point cloud
        Returns:
            d_indices: torch.FloatTensor (B, N, N), distance embedding indices
            a_indices: torch.FloatTensor (B, N, N, k), angular embedding indices
        """
        batch_size, num_point, _ = points.shape

        dist_map = torch.sqrt(pairwise_distance(points, points))  # (B, N, N)
        d_indices = dist_map / self.sigma_d

        k = self.angle_k
        knn_indices = dist_map.topk(k=k + 1, dim=2, largest=False)[1][:, :, 1:]  # (B, N, k)
        knn_indices = knn_indices.unsqueeze(3).expand(batch_size, num_point, k, 3)  # (B, N, k, 3)
        expanded_points = points.unsqueeze(1).expand(batch_size, num_point, num_point, 3)  # (B, N, N, 3)
        knn_points = torch.gather(expanded_points, dim=2, index=knn_indices)  # (B, N, k, 3)
        ref_vectors = knn_points - points.unsqueeze(2)  # (B, N, k, 3)
        anc_vectors = points.unsqueeze(1) - points.unsqueeze(2)  # (B, N, N, 3)
        ref_vectors = ref_vectors.unsqueeze(2).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
        anc_vectors = anc_vectors.unsqueeze(3).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
        sin_values = torch.linalg.norm(torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)  # (B, N, N, k)
        cos_values = torch.sum(ref_vectors * anc_vectors, dim=-1)  # (B, N, N, k)
        angles = torch.atan2(sin_values, cos_values)  # (B, N, N, k)
        a_indices = angles * self.factor_a

        return d_indices, a_indices

    def forward(self, points):
        d_indices, a_indices = self.get_embedding_indices(points)

        d_embeddings = self.embedding(d_indices)
        d_embeddings = self.proj_d(d_embeddings)

        a_embeddings = self.embedding(a_indices)
        a_embeddings = self.proj_a(a_embeddings)
        if self.reduction_a == 'max':
            a_embeddings = a_embeddings.max(dim=3)[0]
        else:
            a_embeddings = a_embeddings.mean(dim=3)

        embeddings = d_embeddings + a_embeddings

        return embeddings


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(SinusoidalPositionalEmbedding, self).__init__()
        if d_model % 2 != 0:
            raise ValueError(f'Sinusoidal positional encoding with odd d_model: {d_model}')
        self.d_model = d_model
        div_indices = torch.arange(0, d_model, 2).float()
        div_term = torch.exp(div_indices * (-np.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, emb_indices):
        r"""Sinusoidal Positional Embedding.

        Args:
            emb_indices: torch.Tensor (*)

        Returns:
            embeddings: torch.Tensor (*, D)
        """
        input_shape = emb_indices.shape
        omegas = emb_indices.view(-1, 1, 1) * self.div_term.view(1, -1, 1)  # (-1, d_model/2, 1)
        sin_embeddings = torch.sin(omegas)
        cos_embeddings = torch.cos(omegas)
        embeddings = torch.cat([sin_embeddings, cos_embeddings], dim=2)  # (-1, d_model/2, 2)
        embeddings = embeddings.view(*input_shape, self.d_model)  # (*, d_model)
        embeddings = embeddings.detach()
        return embeddings


def pairwise_distance(
        x: torch.Tensor, y: torch.Tensor, normalized: bool = False, channel_first: bool = False
) -> torch.Tensor:
    r"""Pairwise distance of two (batched) point clouds.

    Args:
        x (Tensor): (*, N, C) or (*, C, N)
        y (Tensor): (*, M, C) or (*, C, M)
        normalized (bool=False): if the points are normalized, we have "x2 + y2 = 1", so "d2 = 2 - 2xy".
        channel_first (bool=False): if True, the points shape is (*, C, N).

    Returns:
        dist: torch.Tensor (*, N, M)
    """
    if channel_first:
        channel_dim = -2
        xy = torch.matmul(x.transpose(-1, -2).contiguous(), y)
    else:
        channel_dim = -1
        xy = torch.matmul(x, y.transpose(-1, -2).contiguous())  # (*, N, C) x [(*, M, C) -> (*, N, M)]

    if normalized:
        sq_distances = 2.0 - 2.0 * xy
    else:
        x2 = torch.sum(x ** 2, dim=channel_dim).unsqueeze(-1)  # (*, N, C) or (*, C, N) -> (*, N) -> (*, N, 1)
        y2 = torch.sum(y ** 2, dim=channel_dim).unsqueeze(-2)  # (*, M, C) or (*, C, M) -> (*, M) -> (*, 1, M)
        sq_distances = x2 - 2 * xy + y2
    sq_distances = sq_distances.clamp(min=0.0)

    return sq_distances


class PositionEmbeddingCoordsSine(nn.Module):
    """Similar to transformer's position encoding, but generalizes it to
    arbitrary dimensions and continuous coordinates.

    Args:
        n_dim: Number of input dimensions, e.g. 2 for image coordinates.
        d_model: Number of dimensions to encode into
        temperature:
        scale:
    """

    def __init__(self, n_dim: int = 1, d_model: int = 256, temperature=10000.0, scale=1.0):
        super().__init__()

        self.n_dim = n_dim
        self.num_pos_feats = d_model // n_dim // 2 * 2  #
        self.temperature = temperature
        self.padding = d_model - self.num_pos_feats * self.n_dim

        self.scale = scale * 2 * math.pi

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: Point positions (*, d_in)

        Returns:
            pos_emb (*, d_out)
            [sin]
        """
        assert xyz.shape[-1] == self.n_dim

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=xyz.device)  # (0, 1, 2,..., num_pos_feats)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)

        xyz = xyz * self.scale
        pos_divided = xyz.unsqueeze(-1) / dim_t
        pos_sin = pos_divided[..., 0::2].sin()
        pos_cos = pos_divided[..., 1::2].cos()
        pos_emb = torch.stack([pos_sin, pos_cos], dim=-1).reshape(*xyz.shape[:-1], -1)

        # Pad unused dimensions with zeros
        pos_emb = F.pad(pos_emb, (0, self.padding))
        return pos_emb


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, n_dim: int = 1, d_model: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, d_model)
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return self.mlp(xyz)


class PositionEmbeddingLearned_pointNet(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, n_dim: int = 1, d_model: int = 256):
        super().__init__()
        # T
        self.stn = STNkd(n_dim)
        self.fstn = STNkd(256)
        #
        self.l1 = nn.Linear(3, d_model)
        #
        self.l2 = nn.Linear(d_model, 256)
        self.l3 = nn.Linear(256, d_model)

        #
        self.l1_0 = nn.Linear(d_model * 2, d_model * 2)
        self.l2_0 = nn.Linear(d_model * 2, d_model)
        self.l3_0 = nn.Linear(d_model, d_model)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """

        """
        # t
        batchsize, n_pts, _ = xyz.size()  # B, N, C

        T = self.stn(xyz)  # B, C, C
        x = torch.bmm(xyz, T)

        x = F.relu(self.l1(x))

        T_feature = self.fstn(x)
        x = torch.bmm(x, T_feature)

        # 
        pointfeat = x

        # 
        x = F.relu(self.l2(x))
        x = self.l3(x)
        x = torch.max(x, 1, keepdim=True)[0]
        x = x.view(batchsize, -1).repeat(1, n_pts, 1)

        x = torch.cat([x, pointfeat], 2)
        x = F.relu(self.l1_0(x))
        x = F.relu(self.l2_0(x))
        x = self.l3_0(x)

        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()

        self.l_T1 = nn.Linear(k, 64)
        self.l_T2 = nn.Linear(64, 128)
        self.l_T3 = nn.Linear(128, 1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.k = k

    def forward(self, x):
        batchsize, _, _ = x.size()  # B, N, C

        T = F.relu(self.l_T1(x))  # B, N, C
        T = F.relu(self.l_T2(T))  # B, N, C
        T = F.relu(self.l_T3(T))  # B, N, C
        T = torch.max(T, 1, keepdim=True)[0]  # B, 1, C
        T = T.view(batchsize, -1)  # B, C

        T = F.relu(self.fc1(T))
        T = F.relu(self.fc2(T))
        T = self.fc3(T)  # B, k*k

        iden = torch.eye(self.k).view(1, -1).repeat(batchsize, 1)
        T = T + iden.to(T)
        T = T.view(-1, self.k, self.k)
        return T


if __name__ == '__main__':
    PositionEncodingLearn(256)
