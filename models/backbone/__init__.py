from .resnet_fpn import ResNetFPN_8_2, ResNetFPN_16_4 #, ResNetFPN_16_4_deform
from .kpconv_fpn import  KPConvFPN_Kitti, KPConvFPN_Kitti_down_up_PointNet, KPConvFPN_Kitti_down_up


def build_backbone_2d(config):
    if config['backbone_type_2d'] == 'ResNetFPN':
        if config['resolution_2d'] == '8_2':
            return ResNetFPN_8_2(config['resnetfpn_2d'])
        elif config['resolution_2d'] == '16_4':
            return ResNetFPN_16_4(config['resnetfpn_2d'])
    # elif config['backbone_type_2d'] == 'BasicBlock_deform':
    #     return ResNetFPN_16_4_deform(config['resnetfpn_2d'])
    else:
        raise ValueError(f"LOFTR.BACKBONE_TYPE {config['backbone_type_2d']} not supported.")


def build_backbone_3d(config):
    if config['backbone_type_3d'] == 'Kitti':
        return KPConvFPN_Kitti(**config['kpCovfpn'])
    elif config['backbone_type_3d'] == 'KPConvFPN_Kitti_down_up_PointNet':
        return KPConvFPN_Kitti_down_up_PointNet()
    elif config['backbone_type_3d'] == 'KPConvFPN_Kitti_down_up':
        return KPConvFPN_Kitti_down_up(**config['kpCovfpn'])
    else:
        raise ValueError(f"LOFTR.BACKBONE_TYPE {config['backbone_type']} not supported.")
