import numpy as np
import torch
from utils.ops import (
    grid_subsample,
    radius_search
)


def precompute_data_stack_mode(points, lengths, num_stages, voxel_size, radius, neighbor_limits):
    assert num_stages == len(neighbor_limits)

    points_list = []
    lengths_list = []
    neighbors_list = []
    subsampling_list = []
    upsampling_list = []

    # grid subsampling 点云下采样
    for i in range(num_stages):
        if i > 0:
            points, lengths = grid_subsample(points, lengths, voxel_size=voxel_size)
        points_list.append(points)
        lengths_list.append(lengths)
        voxel_size *= 2

    # radius search
    for i in range(num_stages):
        cur_points = points_list[i]
        cur_lengths = lengths_list[i]

        neighbors = radius_search(
            cur_points,
            cur_points,
            cur_lengths,
            cur_lengths,
            radius,
            neighbor_limits[i],
        )
        neighbors_list.append(neighbors)

        if i < num_stages - 1:
            sub_points = points_list[i + 1]
            sub_lengths = lengths_list[i + 1]

            subsampling = radius_search(
                sub_points,
                cur_points,
                sub_lengths,
                cur_lengths,
                radius,
                neighbor_limits[i],
            )
            subsampling_list.append(subsampling)
            
            if i == num_stages -2:
                upsampling = radius_search(
                    cur_points,
                    sub_points,
                    cur_lengths,
                    sub_lengths,
                    radius * 2,
                    neighbor_limits[i + 1],
                )
                upsampling_list.append(upsampling)

        radius *= 2

    return {
        'points': points_list,
        'lengths': lengths_list,
        'neighbors': neighbors_list,
        'subsampling': subsampling_list,
        'upsampling': upsampling_list,
    }


def registration_collate_fn_stack_mode(
        data_dicts, num_stages, voxel_size, search_radius, neighbor_limits, precompute_data=True
):
    r"""Collate function for registration in stack mode.

    Points are organized in the following order: [ref_1, ..., ref_B, src_1, ..., src_B].
    The correspondence indices are within each point cloud without accumulation.
    处理batch的函数
    Args:
        data_dicts (List[Dict])
        num_stages (int)
        voxel_size (float)
        search_radius (float)
        neighbor_limits (List[int])
        precompute_data (bool)

    Returns:
        collated_dict (Dict)
    """
    batch_size = len(data_dicts)
    # merge data with the same key from different samples into a list 合并
    collated_dict = {}
    for data_dict in data_dicts:
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)  # 转torch
            if key not in collated_dict:
                collated_dict[key] = []
            collated_dict[key].append(value)

    # image T
    collated_dict['image'] = torch.stack(collated_dict.pop('image')).permute(0, 3, 1, 2).contiguous()  # [B, C ,H, W]
    collated_dict['rotation'] = torch.stack(collated_dict.pop('rotation'))
    collated_dict['translation'] = torch.stack(collated_dict.pop('translation'))

    if 'cam_K' in collated_dict.keys():
        collated_dict['cam_K'] = torch.stack(collated_dict.pop('cam_K'))

    if 'depth' in collated_dict.keys():
        collated_dict['depth'] = torch.stack(collated_dict.pop('depth'))
    if 'image_rgb' in collated_dict.keys():
        collated_dict['image_rgb'] = torch.stack(collated_dict.pop('image_rgb'))

    # handle special keys: [ref_feats, src_feats] -> feats, [ref_points, src_points] -> points, lengths
    # point
    feats = torch.cat(collated_dict.pop('points_feats'), dim=0)
    collated_dict['features'] = feats
    if 'points_color' in collated_dict.keys():
        points_color = torch.cat(collated_dict.pop('points_color'), dim=0)
        collated_dict['points_color'] = points_color

    points_list = collated_dict.pop('points')
    lengths = torch.LongTensor([points.shape[0] for points in points_list])
    points = torch.cat(points_list, dim=0)

    # if batch_size == 1:
    #     # remove wrapping brackets if batch_size is 1
    #     for key, value in collated_dict.items():
    #         collated_dict[key] = value[0]

    if precompute_data:
        input_dict = precompute_data_stack_mode(points, lengths, num_stages, voxel_size, search_radius, neighbor_limits)
        collated_dict.update(input_dict)
    else:
        collated_dict['points'] = points
        collated_dict['lengths'] = lengths

    collated_dict['batch_size'] = batch_size

    return collated_dict


def calibrate_neighbors_stack_mode(
        dataset, collate_fn, num_stages, voxel_size, search_radius, keep_ratio=0.8, sample_threshold=20000
):
    # Compute higher bound of neighbors number in a neighborhood
    hist_n = int(np.ceil(4 / 3 * np.pi * (search_radius / voxel_size + 1) ** 3))  # 园体积公式
    neighbor_hists = np.zeros((num_stages, hist_n), dtype=np.int32)
    max_neighbor_limits = [hist_n] * num_stages  # list

    # Get histogram of neighborhood sizes i in 1 epoch max.
    for i in range(len(dataset)):
        data_dict = collate_fn(
            [dataset[i]], num_stages, voxel_size, search_radius, max_neighbor_limits, precompute_data=True
        )

        # update histogram
        counts = [np.sum(neighbors.numpy() < neighbors.shape[0], axis=1) for neighbors in data_dict['neighbors']]
        hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
        neighbor_hists += np.vstack(hists)

        if np.min(np.sum(neighbor_hists, axis=1)) > sample_threshold:  # 超过设置的采样阈值
            break

    cum_sum = np.cumsum(neighbor_hists.T, axis=0)
    neighbor_limits = np.sum(cum_sum < (keep_ratio * cum_sum[hist_n - 1, :]), axis=0)

    return neighbor_limits
