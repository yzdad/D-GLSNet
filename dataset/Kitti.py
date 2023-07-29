import open3d as o3d
import h5py
import matplotlib.pyplot as plt
import cv2
import torch.utils.data
import torch
import numpy as np
from typing import Dict
import random
from pathlib import Path
import os
import sys
from imgaug import augmenters as iaa
from scipy.spatial.transform import Rotation
import time

pp = [os.path.dirname((os.path.abspath(__file__))),
      os.path.dirname((os.path.abspath('.'))),
      os.path.dirname((os.path.abspath('..')))]
for p in pp:
    if p not in sys.path:
        sys.path.append(p)

from utils.pointcloud import (
    random_sample_rotation,
    random_sample_rotation_v2,
    random_sample_rotation_z,
    get_transform_from_rotation_translation,
)

class Kitti(torch.utils.data.Dataset):

    def __init__(self, seed=None, task='test', transform=None, **config):
        super(Kitti, self).__init__()
        self.config = config
        # 
        self.dataset_root = Path(config['dataset_root'])
        self.dataset_path = self.dataset_root
        self.datast_split = self.dataset_root / (task + '.txt')
        # 'train','val','test'
        self.subset = task
        #  half_size
        self.hs = config['image_size'] // 2
        # 
        self.point_limit = config['point_limit']
        # 
        self.use_augmentation = config['use_augmentation']
        self.aug_noise = config['augmentation_noise']
        self.aug_rotation = config['augmentation_rotation']
        # image aug
        self.aug = iaa.Sequential([
            iaa.Sometimes(0.5, iaa.Add((-30, 30))),
            iaa.Sometimes(0.5, iaa.LinearContrast((0.7, 1.3))),  # 
            iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(0, 8))),  #
            iaa.Sometimes(0.5, iaa.ImpulseNoise(p=(0, 0.003))),  # 
            iaa.Sometimes(0.5, iaa.MotionBlur(3)),  # 
            iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 1.5))),  # 
        ])


        self.metadata_list = []
        self.source_point_pkg = open(
            self.datast_split).read().split('\n')[0:-1]
        for i in self.source_point_pkg:
            path_pkg = self.dataset_path / i
            point_paths = sorted(list((path_pkg / 'data').glob('*.npz')))
            for point_path in point_paths:
                self.metadata_list.append({'pair_path': point_path, 'scene_name': i, 'name': point_path.stem})

    def __len__(self):
        return len(self.metadata_list)

    def _augment_point_cloud(self, points, rotation, translation):
        r"""
        Augment point clouds.  点云增强
        1. Random rotation to one point cloud.
        2. Random noise.
        ref_points = src_points @ rotation.T + translation
        Args:

        """
        aug_rotation = random_sample_rotation_z(self.aug_rotation)
        aug_translation = (np.random.randn(3) - 0.5) * 1.0

        points += aug_translation
        translation -= aug_translation @ rotation.T 

        points = np.matmul(points, aug_rotation.T)
        rotation = np.matmul(rotation, aug_rotation.T)

        points += (np.random.randn(points.shape[0], 3) - 0.5) *2* self.aug_noise

        return points, rotation, translation

    def __getitem__(self, index):
        data_dict = {}
        # metadata
        metadata = self.metadata_list[index]
        pair_data = np.load(metadata['pair_path'])

        ########################################## image ####################################
        image_rgb = pair_data['map_']
        focal = pair_data['focal']

        x, y = 0, 0
        rotation_z = Rotation.from_euler('xyz', [0, 0, 0]).as_matrix()
        if self.use_augmentation:
            roz = np.random.uniform(0, 360)
            rotation_z = Rotation.from_euler('xyz', [0, 0, roz / 360.0 * np.pi * 2]).as_matrix()
            center = (int(focal[0]), int(focal[1]))
            M = cv2.getRotationMatrix2D(center, roz, 1.0)
            image_rgb = cv2.warpAffine(image_rgb, M, (image_rgb.shape[1], image_rgb.shape[0]))
            
            x = int(np.random.uniform(-100, 100))
            y = int(np.random.uniform(-100, 100))
        #     
        image_rgb = image_rgb[focal[1] - y -  self.hs:focal[1] - y +  self.hs, focal[0] + x -  self.hs:focal[0] + x +  self.hs]
        focal[0], focal[1] = (self.hs - x), (self.hs + y)
        # 
        image = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)[..., np.newaxis]  # 
        if self.use_augmentation:
            image = self.aug.augment_image(image)
            image = image.astype(np.float32) / 255.0
        else:
            image = image / 255.0
        
        # show
        # plt.imshow(cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB))
        # plt.show()
        # plt.imshow(image)
        # plt.show()
        ################################################################################################################
        points = pair_data['point']
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud = point_cloud.voxel_down_sample(0.3)
        points = np.array(point_cloud.points)

        len = np.linalg.norm(points, axis=1)
        index = np.where((len < 50) & (len > 4) & (points[..., 2] > -2)) 
        points = points[index[0]]
        if self.point_limit is not None and points.shape[0] > self.point_limit:
            indices = np.random.permutation(points.shape[0])[:self.point_limit]
            points = points[indices]

        # point augmentation
        rotation = rotation_z  # np.eye(3) @ rotation_z
        translation = np.zeros((3, ))  # (3,)
        if self.use_augmentation:
            points, rotation, translation = self._augment_point_cloud(points, rotation, translation)
        
        # show
        # point_cloud = o3d.geometry.PointCloud()
        # point_cloud.points = o3d.utility.Vector3dVector(points)
        # FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        # o3d.visualization.draw_geometries([FOR1, point_cloud])
        ######################################### check #############################################
        # point = points @ rotation.T + translation
        # map_show = image_rgb.copy()
        # cv2.circle(map_show, (int(focal[0]), int(focal[1])), 4, [0, 0, 255], thickness=4)
        # for p in point:
        #     cv2.circle(map_show, (int(p[0] / pair_data['scale'] + focal[0]), int(-p[1] / pair_data['scale'] + focal[1])), 1,
        #                [0, 0, 0], thickness=1)
        # print(metadata['name'])
        # cv2.imshow("dd", map_show)
        # cv2.waitKey()
        ####################################### return ###############################################
        # infomation
        data_dict['scene_name'] = metadata['scene_name']
        data_dict['name'] = metadata['name']

        # input data
        data_dict['image'] = image.astype(np.float32)  # [H, W, C]
        data_dict['points'] = points.astype(np.float32)  # [N, 3]
        data_dict['points_feats'] = np.ones((points.shape[0], 1), dtype=np.float32)

        # for show
        data_dict['image_rgb'] = image_rgb.astype(np.float32)

        # trans
        data_dict['rotation'] = rotation.astype(np.float32)  # [3, 3]
        data_dict['translation'] = translation.astype(np.float32)  # [3, ]
        data_dict['scale'] = pair_data['scale'].astype(np.float32)  # [3, 3]
        data_dict['focal'] = focal.astype(np.float32)

        return data_dict


if __name__ == '__main__':
    cfg = {
        'dataset_root': '/media/yzdad/datasets/数据包/Kitti',
        'point_limit': 30000,
        'use_augmentation': False,
        'augmentation_noise': 0.02,
        'augmentation_rotation': 1.0,
        'image_size': 480,
    }
    data_test = Kitti(task='watch', **cfg)
    print(len(data_test))
    start = time.time()

    count = 0
    for i in range(len(data_test)):
        if(i >= 1000):
            break
        print(i)
        a = data_test[i]
        count += a['points'].shape[0]

    end = time.time()
    print((end - start) / 1000.0, 's')
    print(count / len(data_test))
