import torch
import torch.nn as nn
from models.backbone.occ.encoder.pointnet import LocalPoolPointnet
from models.backbone.occ.models.decoder import LocalDecoder
from models.backbone.kpconv.modules import ConvBlock, ResidualBlock, UnaryBlock, LastUnaryBlock, nearest_upsample


class KPConvFPN_Kitti(nn.Module):
    def __init__(self, input_dim, init_dim, block_dims, kernel_size, init_radius, init_sigma, group_norm):
        super(KPConvFPN_Kitti, self).__init__()

        self.encoder1_1 = ConvBlock(input_dim, init_dim, kernel_size, init_radius, init_sigma, group_norm)
        self.encoder1_2 = ResidualBlock(init_dim, init_dim * 2, kernel_size, init_radius, init_sigma, group_norm)
        self.layer1_out = nn.Linear(init_dim * 2, init_dim * 2, bias=False)  # TODO

        self.encoder2_1 = ResidualBlock(
            init_dim * 4, 
            block_dims[0], 
            kernel_size, 
            init_radius, 
            init_sigma, 
            group_norm, 
            strided=True
        )
        self.encoder2_2 = ResidualBlock(
            block_dims[0], block_dims[0] * 2, kernel_size, init_radius * 2, init_sigma * 2, group_norm
        )
        self.encoder2_3 = ResidualBlock(
            block_dims[0] * 2, block_dims[0] * 2, kernel_size, init_radius * 2, init_sigma * 2, group_norm
        )
        self.layer2_out = nn.Linear(block_dims[0] * 2, block_dims[0] * 2, bias=False)  # TODO

        self.encoder3_1 = ResidualBlock(
            block_dims[0] * 4, 
            block_dims[1], 
            kernel_size, 
            init_radius * 2, 
            init_sigma * 2, 
            group_norm,  strided=True
        )
        self.encoder3_2 = ResidualBlock(
            block_dims[1], block_dims[1] * 2, kernel_size, init_radius * 4, init_sigma * 4, group_norm
        )
        self.encoder3_3 = ResidualBlock(
            block_dims[1] * 2, block_dims[1] * 2, kernel_size, init_radius * 4, init_sigma * 4, group_norm
        )
        self.layer3_out = nn.Linear(block_dims[1] * 2, block_dims[1] * 2, bias=False)  # TODO

        self.encoder4_1 = ResidualBlock(
            block_dims[1] * 4, 
            block_dims[2], 
            kernel_size, 
            init_radius * 4, 
            init_sigma * 4, 
            group_norm, 
            strided=True
        )
        self.encoder4_2 = ResidualBlock(
            block_dims[2], block_dims[2] * 2, kernel_size, init_radius * 8, init_sigma * 8, group_norm
        )
        self.encoder4_3 = ResidualBlock(
            block_dims[2] * 2, block_dims[2] * 2, kernel_size, init_radius * 8, init_sigma * 8, group_norm
        )
        self.layer4_out = nn.Linear(block_dims[2] * 2, block_dims[2] * 2, bias=False)  # TODO

        self.encoder5_1 = ResidualBlock(
            block_dims[2] * 4,  
            block_dims[3], 
            kernel_size, 
            init_radius * 8, 
            init_sigma * 8, 
            group_norm, 
            strided=True,
        )
        self.encoder5_2 = ResidualBlock(
            block_dims[3], block_dims[3] * 2, kernel_size, init_radius * 16, init_sigma * 16, group_norm
        )
        self.encoder5_3 = ResidualBlock(
            block_dims[3] * 2, block_dims[3] * 2, kernel_size, init_radius * 16, init_sigma * 16, group_norm
        )
        self.layer5_out = nn.Linear(block_dims[3] * 2, 256, bias=False)  # TODO

        self.decoder4_0 = UnaryBlock(block_dims[3] * 2 + block_dims[2] * 4, block_dims[2] * 2, group_norm)
        self.decoder4_1 = UnaryBlock(block_dims[2] * 2, block_dims[2], group_norm)
        self.decoder4_2 = LastUnaryBlock(block_dims[2],  256)

    def forward(self, feats, data_dict):
        points_list = data_dict['points']
        neighbors_list = data_dict['neighbors']
        subsampling_list = data_dict['subsampling']
        upsampling_list = data_dict['upsampling']

        feats_s1 = feats
        feats_s1 = self.encoder1_1(feats_s1, points_list[0], points_list[0], neighbors_list[0])
        feats_s1 = self.encoder1_2(feats_s1, points_list[0], points_list[0], neighbors_list[0])
        feats_s1_out = self.layer1_out(feats_s1)
        feats_s1 = torch.cat([feats_s1, feats_s1_out], dim=1)

        feats_s2 = self.encoder2_1(feats_s1, points_list[1], points_list[0], subsampling_list[0])
        feats_s2 = self.encoder2_2(feats_s2, points_list[1], points_list[1], neighbors_list[1])
        feats_s2 = self.encoder2_3(feats_s2, points_list[1], points_list[1], neighbors_list[1])
        feats_s2_out = self.layer2_out(feats_s2)
        feats_s2 = torch.cat([feats_s2, feats_s2_out], dim=1)

        feats_s3 = self.encoder3_1(feats_s2, points_list[2], points_list[1], subsampling_list[1])
        feats_s3 = self.encoder3_2(feats_s3, points_list[2], points_list[2], neighbors_list[2])
        feats_s3 = self.encoder3_3(feats_s3, points_list[2], points_list[2], neighbors_list[2])
        feats_s3_out = self.layer3_out(feats_s3)
        feats_s3 = torch.cat([feats_s3, feats_s3_out], dim=1)

        feats_s4 = self.encoder4_1(feats_s3, points_list[3], points_list[2], subsampling_list[2])
        feats_s4 = self.encoder4_2(feats_s4, points_list[3], points_list[3], neighbors_list[3])
        feats_s4 = self.encoder4_3(feats_s4, points_list[3], points_list[3], neighbors_list[3])
        feats_s4_out = self.layer4_out(feats_s4)
        feats_s4 = torch.cat([feats_s4, feats_s4_out], dim=1)

        feats_s5 = self.encoder5_1(feats_s4, points_list[4], points_list[3], subsampling_list[3])
        feats_s5 = self.encoder5_2(feats_s5, points_list[4], points_list[4], neighbors_list[4])
        feats_s5 = self.encoder5_3(feats_s5, points_list[4], points_list[4], neighbors_list[4])

        feats_s5_out = self.layer5_out(feats_s5)

        latent_s4 = nearest_upsample(feats_s5, upsampling_list[0])
        latent_s4 = torch.cat([latent_s4, feats_s4], dim=1)
        latent_s4 = self.decoder4_0(latent_s4)
        latent_s4 = self.decoder4_1(latent_s4)
        feats_s4_out = self.decoder4_2(latent_s4)

        return feats_s5_out.unsqueeze(0), feats_s4_out.unsqueeze(0)

from torch_scatter import scatter_mean, scatter_max
from models.backbone.occ.common import coordinate2index, normalize_coordinate, normalize_3d_coordinate, map2local
from models.backbone.occ.encoder.unet import UNet
class KPConvFPN_Kitti_down_up(nn.Module):
    def __init__(self, input_dim, init_dim, block_dims, kernel_size, init_radius, init_sigma, group_norm):
        super(KPConvFPN_Kitti_down_up, self).__init__()
        self.reso_plane = 16
        self.padding = 99
        scatter_type = 'max'
        if scatter_type == 'max':
            self.scatter = scatter_max
        elif scatter_type == 'mean':
            self.scatter = scatter_mean
        else:
            raise ValueError('incorrect scatter type')

        self.encoder1_1 = ConvBlock(input_dim, init_dim, kernel_size, init_radius, init_sigma, group_norm)
        self.encoder1_2 = ResidualBlock(init_dim, init_dim * 2, kernel_size, init_radius, init_sigma, group_norm)
        self.layer1_out = nn.Linear(init_dim * 2, init_dim * 2, bias=False)  # TODO

        self.encoder2_1 = ResidualBlock(
            init_dim * 4, 
            block_dims[0], 
            kernel_size, 
            init_radius, 
            init_sigma, 
            group_norm, 
            strided=True
        )
        self.encoder2_2 = ResidualBlock(
            block_dims[0], block_dims[0] * 2, kernel_size, init_radius * 2, init_sigma * 2, group_norm
        )
        self.encoder2_3 = ResidualBlock(
            block_dims[0] * 2, block_dims[0] * 2, kernel_size, init_radius * 2, init_sigma * 2, group_norm
        )
        self.layer2_out = nn.Linear(block_dims[0] * 2, block_dims[0] * 2, bias=False)  # TODO

        self.encoder3_1 = ResidualBlock(
            block_dims[0] * 4, 
            block_dims[1], 
            kernel_size, 
            init_radius * 2, 
            init_sigma * 2, 
            group_norm,  strided=True
        )
        self.encoder3_2 = ResidualBlock(
            block_dims[1], block_dims[1] * 2, kernel_size, init_radius * 4, init_sigma * 4, group_norm
        )
        self.encoder3_3 = ResidualBlock(
            block_dims[1] * 2, block_dims[1] * 2, kernel_size, init_radius * 4, init_sigma * 4, group_norm
        )
        self.layer3_out = nn.Linear(block_dims[1] * 2, block_dims[1] * 2, bias=False)  # TODO

        self.encoder4_1 = ResidualBlock(
            block_dims[1] * 4, 
            block_dims[2], 
            kernel_size, 
            init_radius * 4, 
            init_sigma * 4, 
            group_norm, 
            strided=True
        )
        self.encoder4_2 = ResidualBlock(
            block_dims[2], block_dims[2] * 2, kernel_size, init_radius * 8, init_sigma * 8, group_norm
        )
        self.encoder4_3 = ResidualBlock(
            block_dims[2] * 2, block_dims[2] * 2, kernel_size, init_radius * 8, init_sigma * 8, group_norm
        )
        self.layer4_out = nn.Linear(block_dims[2] * 2, block_dims[2] * 2, bias=False)  # TODO

        self.encoder5_1 = ResidualBlock(
            block_dims[2] *4,  
            block_dims[3], 
            kernel_size, 
            init_radius * 8, 
            init_sigma * 8, 
            group_norm, 
            strided=True,
        )
        self.encoder5_2 = ResidualBlock(
            block_dims[3], block_dims[3] * 2, kernel_size, init_radius * 16, init_sigma * 16, group_norm
        )
        self.encoder5_3 = ResidualBlock(
            block_dims[3] * 2, block_dims[3] * 2, kernel_size, init_radius * 16, init_sigma * 16, group_norm
        )
        self.layer5_out = nn.Linear(block_dims[3] * 2, 256, bias=False)  # TODO



        self.decoder4_0 = UnaryBlock(block_dims[3] * 2 + block_dims[2] * 4, block_dims[2] * 2, group_norm)
        self.decoder4_1 = UnaryBlock(block_dims[2] * 2, block_dims[2], group_norm)
        self.decoder4_2 = LastUnaryBlock(block_dims[2],  256)  # todo
    
    def pool_local(self, index, c, reso_plane):
        bs, fea_dim = c.size(0), c.size(2)

        c_out = 0
        fea = self.scatter(c.permute(0, 2, 1), index, dim_size=reso_plane**2)
        if self.scatter == scatter_max:
            fea = fea[0]
        # gather feature back to points
        fea = fea.gather(dim=2, index=index.expand(-1, fea_dim, -1))
        c_out += fea
        return c_out.permute(0, 2, 1)
    
    def generate_plane_features(self, p, c, plane='xz'):
        # acquire indices of features in plane
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        index = coordinate2index(xy, self.reso_plane)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), 256, self.reso_plane**2)
        c = c.permute(0, 2, 1) # B x 512 x T
        fea_plane = scatter_mean(c, index, out=fea_plane) # B x 512 x reso^2
        fea_plane = fea_plane.reshape(p.size(0), self.c_dim, self.reso_plane, self.reso_plane) # sparce matrix (B x 512 x reso x reso)

        fea_plane = self.unet(fea_plane)

        return fea_plane
    
    def forward(self, feats, data_dict):
        points_list = data_dict['points']
        neighbors_list = data_dict['neighbors']
        subsampling_list = data_dict['subsampling']
        upsampling_list = data_dict['upsampling']

        #########################################
        feats_s1 = feats
        feats_s1 = self.encoder1_1(feats_s1, points_list[0], points_list[0], neighbors_list[0])
        feats_s1 = self.encoder1_2(feats_s1, points_list[0], points_list[0], neighbors_list[0])
        
        feats_s1_out = self.layer1_out(feats_s1).unsqueeze(0)
        index_1 = coordinate2index(normalize_coordinate(points_list[0].clone().unsqueeze(0), plane='xy', padding=self.padding), self.reso_plane * 8)
        pooled_s1 = self.pool_local(index_1, feats_s1_out, self.reso_plane * 8).squeeze(0)
        feats_s1 = torch.cat([feats_s1, pooled_s1], dim=1)

        #########################################
        feats_s2 = self.encoder2_1(feats_s1, points_list[1], points_list[0], subsampling_list[0])
        feats_s2 = self.encoder2_2(feats_s2, points_list[1], points_list[1], neighbors_list[1])
        feats_s2 = self.encoder2_3(feats_s2, points_list[1], points_list[1], neighbors_list[1])

        feats_s2_out = self.layer2_out(feats_s2).unsqueeze(0)
        index_2 = coordinate2index(normalize_coordinate(points_list[1].clone().unsqueeze(0), plane='xy', padding=self.padding), self.reso_plane * 4)
        pooled_s2 = self.pool_local(index_2, feats_s2_out, self.reso_plane * 4).squeeze(0)
        feats_s2 = torch.cat([feats_s2, pooled_s2], dim=1)

        ##########################################
        feats_s3 = self.encoder3_1(feats_s2, points_list[2], points_list[1], subsampling_list[1])
        feats_s3 = self.encoder3_2(feats_s3, points_list[2], points_list[2], neighbors_list[2])
        feats_s3 = self.encoder3_3(feats_s3, points_list[2], points_list[2], neighbors_list[2])

        feats_s3_out = self.layer3_out(feats_s3).unsqueeze(0)
        index_3 = coordinate2index(normalize_coordinate(points_list[2].clone().unsqueeze(0), plane='xy', padding=self.padding), self.reso_plane  * 2)
        pooled_s3 = self.pool_local(index_3, feats_s3_out, self.reso_plane  * 2).squeeze(0)
        feats_s3 = torch.cat([feats_s3, pooled_s3], dim=1)

        ##########################################
        feats_s4 = self.encoder4_1(feats_s3, points_list[3], points_list[2], subsampling_list[2])
        feats_s4 = self.encoder4_2(feats_s4, points_list[3], points_list[3], neighbors_list[3])
        feats_s4 = self.encoder4_3(feats_s4, points_list[3], points_list[3], neighbors_list[3])

        feats_s4_out = self.layer4_out(feats_s4).unsqueeze(0)
        index_4 = coordinate2index(normalize_coordinate(points_list[3].clone().unsqueeze(0), plane='xy', padding=self.padding), self.reso_plane)
        pooled_s4 = self.pool_local(index_4, feats_s4_out, self.reso_plane).squeeze(0)
        feats_s4 = torch.cat([feats_s4, pooled_s4], dim=1)

        ############################################
        feats_s5 = self.encoder5_1(feats_s4, points_list[4], points_list[3], subsampling_list[3])
        feats_s5 = self.encoder5_2(feats_s5, points_list[4], points_list[4], neighbors_list[4])
        feats_s5 = self.encoder5_3(feats_s5, points_list[4], points_list[4], neighbors_list[4])

        feats_s5_out = self.layer5_out(feats_s5).unsqueeze(0)

        ############################################ up
        latent_s4 = nearest_upsample(feats_s5, upsampling_list[0])
        latent_s4 = torch.cat([latent_s4, feats_s4], dim=1)
        latent_s4 = self.decoder4_0(latent_s4)
        latent_s4 = self.decoder4_1(latent_s4)
        feats_s4_out = self.decoder4_2(latent_s4).unsqueeze(0)    

        return feats_s5_out, feats_s4_out


class KPConvFPN_Kitti_down_up_PointNet(nn.Module):
    def __init__(self):
        super(KPConvFPN_Kitti_down_up_PointNet, self).__init__()
        unet_kwargs = {'depth': 5, 'merge_mode': 'concat',  'start_filts': 32}
        self.encoder = LocalPoolPointnet(c_dim=256, dim=3, hidden_dim=256, scatter_type='max', 
                                         unet=True, unet_kwargs=unet_kwargs, plane_resolution=128, padding=99, plane_type='xy')
        self.decoder = LocalDecoder(dim=3, c_dim=256, hidden_size=256)
        

    def forward(self, feats, data_dict):
        points_list = data_dict['points']

        feats = self.encoder(points_list[0].unsqueeze(0))

        feats_s5_out = self.decoder(points_list[-1].unsqueeze(0), feats)
        feats_s4_out = self.decoder(points_list[-2].unsqueeze(0), feats)

        return feats_s5_out, feats_s4_out
    



if __name__ == '__main__':
    KPConvFPN_Kitti_down_up(init_dim=64, input_dim=1, block_dims= [ 128, 256, 512, 1024 ], group_norm=32, kernel_size=15, init_radius=1.6, init_sigma=1.2)
