from math import log
from loguru import logger

import torch
from einops import repeat
from kornia.utils import create_meshgrid


##############  ↓  Coarse-Level supervision  ↓  ##############


@torch.no_grad()
def mask_pts_at_padded_regions(grid_pt, mask):
    """For megadepth dataset, zero-padding exists in images"""
    mask = repeat(mask, 'n h w -> n (h w) c', c=2)
    grid_pt[~mask.bool()] = 0
    return grid_pt


def compute_supervision_coarse(data, config):
    """ 计算粗匹配监督信息
    Update:


    NOTE:
        - for scannet dataset, there're 3 kinds of resolution {i, c, f}
        - for megadepth dataset, there're 4 kinds of resolution {i, i_resize, c, f}
    """

    def out_bound_mask(pt, w, h, b=0):
        return (pt[..., 0] < 0 + b) + (pt[..., 0] >= w - b) + (pt[..., 1] < 0 + b) + (pt[..., 1] >= h - b)

    # 1. misc
    device = data['image'].device
    N, _, H, W = data['image'].shape  # N = 1
    scale = config['model']['backbone']['scale']  # 输入图片与粗匹配特征图的比例
    h, w = map(lambda x: x // scale, [H, W])  # 粗特征大小

    point_c = data['points'][-1]
    num_p = point_c.shape[0]

    rotation = data['rotation'].squeeze()
    translation = data['translation'].squeeze()
    cam_K = data['cam_K'].squeeze()
    depth = data['depth'].squeeze()

    points_c = torch.mm(point_c, rotation.T) + translation
    points_u_v_z = torch.mm(points_c, cam_K.T)
    u = points_u_v_z[..., 0] / points_u_v_z[..., 2]
    v = points_u_v_z[..., 1] / points_u_v_z[..., 2]
    z = points_u_v_z[..., 2]
    pt_uv = torch.stack([u, v], dim=-1).round().long()

    index = ~out_bound_mask(pt_uv, W, H)
    point_id = torch.where(index)[0]
    image_point = pt_uv[index]
    z = z[index]

    z_gt = depth[image_point[..., 1], image_point[..., 0]]
    index_z = torch.where(torch.abs(z_gt - z) < 0.1)[0]  # TODO
    point_id = point_id[index_z]
    image_point = image_point[index_z]

    image_point = image_point // scale
    image_id = image_point[..., 0] + image_point[..., 1] * w

    if config['model']['crossMatch']['match_type'] == 'sinkhorn':
        # pt_score =  torch.norm(torch.stack([u, v], dim=-1) / scale - pt - 0.5, dim=-1)

        conf_matrix_gt = torch.zeros(h * w + 1, num_p, device=device)
        conf_matrix_gt[image_id, point_id] = 1
        sum_conf = torch.sum(conf_matrix_gt, dim=0)
        conf_matrix_gt[h * w, torch.where(sum_conf == 0)[0]] = 1  # 垃圾袋
    else:
        conf_matrix_gt = torch.zeros(h * w, num_p, device=device)
        conf_matrix_gt[image_id, point_id[0]] = 1

    data.update({'conf_matrix_gt': conf_matrix_gt.unsqueeze(0)})

    # 用于fine
    index_ = ~out_bound_mask(pt_uv, W, H, b=scale)  # 去边缘
    point_id = torch.where(index_)[0]
    image_point = pt_uv[index_] // scale
    image_id = image_point[..., 0] + image_point[..., 1] * w

    data.update({
        'spv_b_ids': torch.zeros(image_id.shape[0], device=image_id.device),
        'spv_i_ids': image_id,
        'spv_p_ids': point_id
    })

    data.update({
        'spv_uv_i': pt_uv,
    })


def compute_supervision_coarse_kitti(data, config):
    """ 计算粗匹配监督信息
    Update:


    NOTE:
        - for scannet dataset, there're 3 kinds of resolution {i, c, f}
        - for megadepth dataset, there're 4 kinds of resolution {i, i_resize, c, f}
    """

    def out_bound_mask(pt, w, h, b=0):
        return (pt[..., 0] < 0 + b) + (pt[..., 0] >= w - b) + (pt[..., 1] < 0 + b) + (pt[..., 1] >= h - b)

    # 1. misc
    device = data['image'].device
    N, _, H, W = data['image'].shape  # N = 1
    scale = config['model']['backbone']['scale']  # 输入图片与粗匹配特征图的比例
    h, w = map(lambda x: x // scale, [H, W])  # 粗特征大小

    point_c = data['points'][-1]
    num_p = point_c.shape[0]

    rotation = data['rotation'].squeeze()
    translation = data['translation'].squeeze()
    scale_m_pt = data['scale'][0]
    focal = data['focal'][0]

    point_c = torch.mm(point_c, rotation.T) + translation
    u = point_c[..., 0] / scale_m_pt + focal[0]
    v = -point_c[..., 1] / scale_m_pt + focal[1]
    pt_uv = torch.stack([u, v], dim=-1).round().long()

    # import numpy as np
    # import cv2
    # map_show = data['image_rgb'][0].cpu().numpy().astype(np.uint8)
    # cv2.circle(map_show, (int(focal[0]), int(focal[1])), 4, [0, 0, 255], thickness=4)
    # for p in pt_uv.cpu().numpy():
    #         cv2.circle(map_show, (int(p[0]), int(p[1])), 1, [0, 0, 0], thickness=2)
    # cv2.imshow("dd", map_show)
    # cv2.waitKey()

    index = ~out_bound_mask(pt_uv, W, H)
    point_id = torch.where(index)[0]
    image_point = pt_uv[index] // scale
    image_id = image_point[..., 0] + image_point[..., 1] * w

    if config['model']['crossMatch']['match_type'] == 'sinkhorn':
        # pt_score =  torch.norm(torch.stack([u, v], dim=-1) / scale - pt - 0.5, dim=-1)

        conf_matrix_gt = torch.zeros(h * w + 1, num_p, device=device)
        conf_matrix_gt[image_id, point_id] = 1
        sum_conf = torch.sum(conf_matrix_gt, dim=0)
        conf_matrix_gt[h * w, torch.where(sum_conf == 0)[0]] = 1  # 垃圾袋
    else:
        conf_matrix_gt = torch.zeros(h * w, num_p, device=device)
        conf_matrix_gt[image_id, point_id[0]] = 1

    data.update({'conf_matrix_gt': conf_matrix_gt.unsqueeze(0)})

    # uu, vv = u / scale, v / scale
    # uu_i, vv_i = uu.floor(), vv.floor()
    # weight = torch.stack([(uu - uu_i) * (vv - vv_i), \
    #                                               (uu_i + 1 -uu) * (vv - vv_i), \
    #                                               (uu - uu_i) * (vv_i + 1 - vv),  \
    #                                               (uu_i + 1 -uu) * (vv_i + 1 - vv)]).reshape(-1)
    # xy = torch.stack([torch.stack([uu_i, vv_i], dim=-1), \
    #                                     torch.stack([uu_i + 1, vv_i], dim=-1), \
    #                                     torch.stack([uu_i, vv_i + 1], dim=-1),  \
    #                                     torch.stack([uu_i + 1, vv_i + 1], dim=-1)]).long().reshape(-1, 2)

    # index_xy = ~out_bound_mask(xy, w, h)
    # point_id = torch.where(index_xy)[0]  % point_c.shape[0]
    # image_point = xy[index_xy]
    # weight = weight[index_xy]
    # image_id = image_point[..., 0] + image_point[..., 1] * w

    # if config['model']['crossMatch']['match_type'] == 'sinkhorn':
    #     # pt_score =  torch.norm(torch.stack([u, v], dim=-1) / scale - pt - 0.5, dim=-1)

    #     conf_matrix_gt = torch.zeros(h * w + 1, num_p, device=device)
    #     conf_matrix_gt[image_id, point_id] = weight
    #     sum_conf = torch.sum(conf_matrix_gt, dim=0)
    #     conf_matrix_gt[h * w, :] = 1 -  sum_conf  # 垃圾袋
    # else:
    #     conf_matrix_gt = torch.zeros(h * w, num_p, device=device)
    #     conf_matrix_gt[image_id, point_id[0]] = 1

    # data.update({'conf_matrix_gt': conf_matrix_gt.unsqueeze(0)})

    # # 用于fine
    # pt_uv = torch.stack([u, v], dim=-1).round().long()

    # 用于fine
    index_ = ~out_bound_mask(pt_uv, W, H, b=scale)  # 去边缘
    point_id = torch.where(index_)[0]
    image_point = pt_uv[index_] // scale
    image_id = image_point[..., 0] + image_point[..., 1] * w

    data.update({
        'spv_b_ids': torch.zeros(image_id.shape[0], device=image_id.device),
        'spv_i_ids': image_id,
        'spv_p_ids': point_id
    })

    data.update({
        'spv_uv_i': pt_uv,
    })


##############  ↓  Fine-Level supervision  ↓  ##############

@torch.no_grad()
def compute_supervision_fine(data, config):
    """
    Update:
        data (dict):{
            "expec_f_gt": [M, 2]}
    """
    # 1. misc
    # w_pt0_i, pt1_i = data.pop('spv_w_pt0_i'), data.pop('spv_pt1_i')
    w_pt0_i, pt1_i = data['spv_w_pt0_i'], data['spv_pt1_i']
    scale = config['LOFTR']['RESOLUTION'][1]  # 输入图片到f的尺度
    radius = config['LOFTR']['FINE_WINDOW_SIZE'] // 2

    # 2. get coarse prediction
    b_ids, i_ids, j_ids = data['b_ids'], data['i_ids'], data['j_ids']  # 粗匹配

    # 3. compute gt
    scale = scale * data['scale1'][b_ids] if 'scale0' in data else scale
    # `expec_f_gt` might exceed the window, i.e. abs(*) > 1, which would be filtered later
    expec_f_gt = (w_pt0_i[b_ids, i_ids] - pt1_i[b_ids, j_ids]
                  ) / scale / radius  # [M, 2]  （-1， 1）
    data.update({"expec_f_gt": expec_f_gt})


def compute_supervision_fine_kitti(data, config):
    """
    Update:
        data (dict):{
            "expec_f_gt": [M, 2]}
    """
    # 1. misc
    # w_pt0_i, pt1_i = data.pop('spv_w_pt0_i'), data.pop('spv_pt1_i')
    w_uv_i, w_uv_c = data['point_c_match_gt'], data['uv_c_in_i']
    scale = data['scale_i_f']  # 输入图片到f的尺度
    radius = config['model']['fine_preprocess']['fine_window_size'] // 2

    # 2. compute gt
    # `expec_f_gt` might exceed the window, i.e. abs(*) > 1, which would be filtered later
    #  [M, 2]  （-1， 1）
    expec_f_gt = (w_uv_i - w_uv_c) / scale / radius
    data.update({"expec_f_gt": expec_f_gt})
