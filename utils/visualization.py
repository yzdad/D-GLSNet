import cv2
import open3d as o3d
import numpy as np


def show_match_2d(batch, index_i, index_p, scale=8, bound=200):
    image = batch['image_rgb'].squeeze(0).cpu().numpy()
    point_list = batch['points']
    point = point_list[0].cpu().numpy()
    color = batch['points_color'].cpu().numpy()
    point_c = point_list[-1].cpu().numpy()

    rotation = batch['rotation'].squeeze().cpu().numpy()
    translation = batch['translation'].squeeze().cpu().numpy()
    cam_K = batch['cam_K'].squeeze().cpu().numpy()

    #
    if image.shape[-1] == 1:
        image = image.repeat(3, axis=-1)
    if color.shape[1] == 1:
        color = np.zeros_like(color)
    # color
    match_num = index_i.shape[0]
    color_match = np.random.random((match_num, 3))

    # images
    H, W = image.shape[0], image.shape[1]
    h, w = H // scale, W // scale
    point_i = []
    for i, c in zip(index_i, color_match):
        x = i % w * scale + scale / 2
        y = i // w * scale + scale / 2
        point_i.append((x, y))
        cv2.circle(image, (int(x), int(y)), 1, c[::-1], thickness=2)
    point_i = np.stack(point_i)

    # point
    image_ = np.ones((image.shape[0] + 2 * bound,
                     image.shape[1], image.shape[2]))
    image_[bound:bound + image.shape[0], :] = image
    image_p = np.ones(
        (image.shape[0] + 2 * bound, image.shape[1] + 2 * bound, image.shape[2]))
    image = image_

    # show all point
    point = point @ rotation.T + translation
    points_u_v_z = point @ cam_K.T
    points_u_v_z[..., 0] = points_u_v_z[..., 0] / points_u_v_z[..., 2]
    points_u_v_z[..., 1] = points_u_v_z[..., 1] / points_u_v_z[..., 2]
    points_u_v = points_u_v_z[..., 0:2].round().astype(np.int32) + bound
    index = (points_u_v[..., 0] < 1) + (points_u_v[..., 1] < 1) + \
            (points_u_v[..., 0] >= image_p.shape[1] - 1) + \
        (points_u_v[..., 1] >= image_p.shape[0] - 1)
    flag = ~index
    points_u_v = points_u_v[flag]
    color = color[flag]
    image_p[points_u_v[..., 1], points_u_v[..., 0], :] = color
    image_p[points_u_v[..., 1] - 1, points_u_v[..., 0] - 1, :] = color
    image_p[points_u_v[..., 1] + 1, points_u_v[..., 0] + 1, :] = color
    image_p[points_u_v[..., 1] - 1, points_u_v[..., 0] + 1, :] = color
    image_p[points_u_v[..., 1] + 1, points_u_v[..., 0] - 1, :] = color
    image_p[points_u_v[..., 1] + 1, points_u_v[..., 0], :] = color
    image_p[points_u_v[..., 1] - 1, points_u_v[..., 0], :] = color
    image_p[points_u_v[..., 1], points_u_v[..., 0] - 1, :] = color
    image_p[points_u_v[..., 1], points_u_v[..., 0] + 1, :] = color

    # point_c
    point_c = point_c[index_p] @ rotation.T + translation
    points_u_v_z = point_c @ cam_K.T
    u = points_u_v_z[..., 0] / points_u_v_z[..., 2]
    v = points_u_v_z[..., 1] / points_u_v_z[..., 2]
    pt_c = np.stack([u, v], axis=-1)

    error = np.abs(point_i - pt_c)
    print(error)

    for i, c in zip(pt_c, color_match):
        cv2.circle(image_p, (int(i[0] + bound),
                   int(i[1] + bound)), 1, c[::-1], thickness=2)

    # match line
    show_img = np.hstack([image, image_p])
    for i, j, c in zip(point_i, pt_c, color_match):
        cv2.line(show_img, (int(i[0]), int(i[1] + bound)), (int(j[0] + W + bound), int(j[1] + bound)), c[::-1],
                 thickness=1)

    return show_img


def show_match_kitti_2d(batch, bound=10):
    image = batch['image_rgb'].squeeze(0).cpu().numpy() / 255.0
    point_list = batch['points']
    point = point_list[0].cpu().numpy()

    uv_i_i = batch['uv_f_in_i'].cpu().numpy()

    rotation = batch['rotation'].squeeze().cpu().numpy()
    translation = batch['translation'].squeeze().cpu().numpy()

    pt_c = batch['point_c_match_gt'].cpu().numpy()

    distance = np.linalg.norm(uv_i_i - pt_c, axis=-1)
    index = np.where(distance < 15)
    uv_i_i, pt_c = uv_i_i[index], pt_c[index]

    #
    if image.shape[-1] == 1:
        image = image.repeat(3, axis=-1)

    # color
    match_num = uv_i_i.shape[0]
    color_match = np.random.random((match_num, 3)) - 0.2  # 匹配点的颜色

    # images
    H, W = image.shape[0], image.shape[1]
    for p, c in zip(uv_i_i, color_match):
        cv2.circle(image, (int(p[0]), int(p[1])), 1, c[::-1], thickness=2)

    # point
    image_ = np.ones((image.shape[0] + 2 * bound,
                     image.shape[1], image.shape[2]))
    image_[bound:bound + image.shape[0], :] = image
    image_p = np.ones(
        (image.shape[0] + 2 * bound, image.shape[1] + 2 * bound, image.shape[2]))
    # image_p[bound:bound + image.shape[0], bound:bound + image.shape[1]] = image
    image = image_

    # show all point
    scale_m_pt = batch['scale'][0].cpu().numpy()
    focal = batch['focal'][0].cpu().numpy()
    point = point @ rotation.T + translation
    z_sort = np.argsort(point[:, 2]) 
    point = point[z_sort] 
    u = point[..., 0] / scale_m_pt +  H // 2
    v = -point[..., 1] / scale_m_pt +  W // 2
    points_u_v = np.stack([u, v], axis=-1).round() + bound
    index = (points_u_v[..., 0] < 1) + (points_u_v[..., 1] < 1) + \
            (points_u_v[..., 0] >= image_p.shape[1] - 1) + \
        (points_u_v[..., 1] >= image_p.shape[0] - 1)
    flag = ~index
    points_u_v = points_u_v[flag]

    # 根据高度生成色彩
    colors = np.zeros([point.shape[0], 3])
    height_max = np.max(point[:, 2]) * 0.9
    height_min = np.min(point[:, 2]) * 0.9
    z_median = abs(height_max - height_min) / 2

    for j in range(point.shape[0]):
        color_n = point[j, 2] - height_min
        if color_n > z_median:
            colors[j, :] = [(color_n -z_median) / z_median + 0.3, 1 - ((color_n -z_median) / z_median), 0]
        else:
            colors[j, :] = [0, color_n / z_median, 1-color_n / z_median + 0.3]

    for p, c in zip(points_u_v, colors):
        cv2.circle(image_p, (int(p[0]), int(p[1])), 1, c[::-1], thickness=1)

    # point_c
    for p, c in zip(pt_c + bound, color_match):
        cv2.circle(image_p, (int(p[0] - focal[0] + H // 2), int(p[1]- focal[1] + W // 2)), 1, c[::-1], thickness=2)

    # # match line
    show_img = np.hstack([image, image_p])
    for i, j, c in zip(uv_i_i, pt_c, color_match):
        cv2.line(show_img, (int(i[0]), int(i[1] + bound)), (int(j[0] + W + bound- focal[0] + H // 2), int(j[1] + bound- focal[1] + W // 2)), c[::-1],
                 thickness=1)

    return show_img


def show_match_kitti_2d_h(batch, bound=10):
    image = batch['image_rgb'].squeeze(0).cpu().numpy() / 255.0
    point_list = batch['points']
    point = point_list[0].cpu().numpy()

    uv_i_i = batch['uv_c_in_i'].cpu().numpy()

    rotation = batch['rotation'].squeeze().cpu().numpy()
    translation = batch['translation'].squeeze().cpu().numpy()

    pt_c = batch['point_c_match_gt'].cpu().numpy()

    distance = np.linalg.norm(uv_i_i - pt_c, axis=-1)
    index = np.where(distance < 12)
    uv_i_i, pt_c = uv_i_i[index], pt_c[index]

    #
    if image.shape[-1] == 1:
        image = image.repeat(3, axis=-1)

    # color
    match_num = uv_i_i.shape[0]
    color_match = np.random.random((match_num, 3)) - 0.2  # 匹配点的颜色

    # images
    H, W = image.shape[0], image.shape[1]
    for p, c in zip(uv_i_i, color_match):
        cv2.circle(image, (int(p[0]), int(p[1])), 1, c[::-1], thickness=2)

    # point
    image_ = np.ones((image.shape[0], image.shape[1] + 2 * bound, image.shape[2]))
    image_[:, bound:bound + image.shape[1]] = image
    image_p = np.ones(
        (image.shape[0] + 2 * bound, image.shape[1] + 2 * bound, image.shape[2]))
    # image_p[bound:bound + image.shape[0], bound:bound + image.shape[1]] = image
    image = image_

    # show all point
    scale_m_pt = batch['scale'][0].cpu().numpy()
    focal = batch['focal'][0].cpu().numpy()
    point = point @ rotation.T + translation
    u = point[..., 0] / scale_m_pt + focal[0]
    v = -point[..., 1] / scale_m_pt + focal[1]
    points_u_v = np.stack([u, v], axis=-1).round() + bound
    index = (points_u_v[..., 0] < 1) + (points_u_v[..., 1] < 1) + \
            (points_u_v[..., 0] >= image_p.shape[1] - 1) + \
        (points_u_v[..., 1] >= image_p.shape[0] - 1)
    flag = ~index
    points_u_v = points_u_v[flag]

    # 根据高度生成色彩
    colors = np.zeros([point.shape[0], 3])
    height_max = np.max(point[:, 2]) * 0.9
    height_min = np.min(point[:, 2]) * 0.9
    z_median = abs(height_max - height_min) / 2

    for j in range(point.shape[0]):
        color_n = point[j, 2] - height_min
        if color_n > z_median:
            colors[j, :] = [(color_n -z_median) / z_median + 0.3, 1 - ((color_n -z_median) / z_median), 0]
        else:
            colors[j, :] = [0, color_n / z_median, 1-color_n / z_median + 0.3]

    for p, c in zip(points_u_v, colors):
        cv2.circle(image_p, (int(p[0]), int(p[1])), 1, c[::-1], thickness=1)

    # point_c
    for p, c in zip(pt_c + bound, color_match):
        cv2.circle(image_p, (int(p[0]), int(p[1])), 1, c[::-1], thickness=2)

    # # match line
    show_img = np.vstack([image_p, image])
    for i, j, c in zip(uv_i_i, pt_c, color_match):
        cv2.line(show_img, (int(i[0] + bound), int(i[1] + H + 2 * bound)), (int(j[0] + bound), int(j[1] + bound)), c[::-1],
                 thickness=1)

    return show_img