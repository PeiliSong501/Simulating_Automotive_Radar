import math
import pandas as pd
import numpy as np
import os
import cv2
import sys
sys.path.append("/workspace/code/RadarSimulator2")
import utils
import random
from sklearn.model_selection import train_test_split
import re
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from scipy.interpolate import griddata
from shapely.geometry import MultiPoint, Polygon, MultiLineString
from shapely.ops import unary_union, polygonize
from scipy.spatial import Delaunay
from shapely import geometry
#import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib import cm


def farthest_point_sampling(points, num_samples):
    """
    最远点采样算法（Farthest Point Sampling）
    :param points: 输入点云，形状为 (N, 3)
    :param num_samples: 采样点数 M
    :return: 下采样后的点云，形状为 (M, 3)
    """
    N, _ = points.shape
    if num_samples >= N:
        return points

    sampled_indices = [np.random.randint(N)]
    distances = np.full(N, np.inf)

    for _ in range(num_samples - 1):
        # 获取最近点到采样集合的距离
        last_sampled_point = points[sampled_indices[-1]]
        dist_to_last = np.linalg.norm(points - last_sampled_point, axis=1)
        distances = np.minimum(distances, dist_to_last)

        next_index = np.argmax(distances)
        sampled_indices.append(next_index)

    return points[sampled_indices]


def generate_depth_map(l_in_2d, points_3d_l2c, image_width, image_height):
    """
    生成深度图并完成双线性插值
    :param points_3d_l2c: 相机坐标系下的激光雷达点 (N, 3), 每行是 [x, y, z]
    :param K: 相机内参矩阵 (3, 3)
    :param image_width: 图像宽度
    :param image_height: 图像高度
    :return: 插值前的深度图, 插值后的深度图 (H, W)
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # 提取点云的 x, y, z 坐标
    x = l_in_2d[:, 0]
    y = l_in_2d[:, 1]
    z = points_3d_l2c[:, 2]

    u, v = x,y
    v = v - 571
    # 确保点在图像范围内
    valid_mask = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height) & (z > 0)
    u = u[valid_mask]
    v = v[valid_mask]
    depth = z[valid_mask]


    # 将 (u, v) 离散化为像素索引
    u = np.round(u).astype(int)
    v = np.round(v).astype(int)

    # 初始化深度图
    depth_map = np.zeros((image_height, image_width))
    depth_map[v, u] = depth  # 填充已知深度值

    # 插值前的深度图
    depth_map_before = depth_map.copy()

    # 构造网格
    grid_x, grid_y = np.meshgrid(np.arange(image_width), np.arange(image_height))

    # 双线性插值
    valid_points = np.array([u, v]).T
    depth_interpolated = griddata(
        valid_points, depth, (grid_x, grid_y), method='linear', fill_value=0
    )

    return depth_map_before, depth_interpolated


def calculate_coverage_and_count(depth_map):
    """
    计算深度图中具有深度值的像素点数量和百分比
    :param depth_map: 深度图
    :return: 非零像素点数量, 百分比
    """
    total_pixels = depth_map.size
    non_zero_pixels = np.count_nonzero(depth_map)
    percentage = (non_zero_pixels / total_pixels) * 100
    return non_zero_pixels, percentage


def visualize_depth_map(depth_map, title):
    depth_normalized = (255 * (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))).astype(
        np.uint8)

    plt.imshow(depth_normalized, cmap='gray')
    plt.colorbar(label='Depth Value')
    plt.title(title)
    plt.axis('off')
    plt.show()

def save_depth_map_as_image(depth_map, file_path):

    depth_normalized = (255 * (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))).astype(np.uint8)
    cv2.imwrite(file_path, depth_normalized)

def check_and_regenerate_depth_maps(start_frame, end_frame, depth_map_dir, regen_dir, image_width, image_height):
    """
    检查深度图文件是否有效，并在无效时重新生成。
    :param start_frame: 开始帧
    :param end_frame: 结束帧
    :param depth_map_dir: 深度图文件目录
    :param regen_dir: 重新生成的深度图保存目录
    :param image_width: 图像宽度
    :param image_height: 图像高度
    """
    for frame in range(start_frame, end_frame + 1):
        depth_map_file = os.path.join(depth_map_dir, f"{frame}.npy")
        regen_file_path = os.path.join(regen_dir, f"{frame}.npy")
        
        try:
            if not os.path.exists(depth_map_file):
                print(f"File missing for frame {frame}, regenerating...")
                raise ValueError("Depth map file missing.")
            
            # 检查深度图
            depth_map = np.load(depth_map_file)
            if depth_map.shape != (image_height, image_width) or np.all(depth_map == 0):
                raise ValueError("Invalid depth map dimensions or all zeros.")
            
            print(f"Depth map for frame {frame} is valid.")
        
        except (ValueError, FileNotFoundError) as e:
            print(f"Frame {frame} failed validation: {e}")
            try:
                # 重新生成深度图
                radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(frame)
                if not (os.path.exists(radar_file) and os.path.exists(calib_file)):
                    print(f"Missing input files for frame {frame}, skipping regeneration.")
                    continue
                
                l_in_2d = np.load(f'/workspace/data/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/lidar_2d/{frame}.npy')
                l_in = np.load(f'/workspace/data/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/lidar2d_corr_3dpoints/{frame}.npy')
                K = utils.get_intrinsic_matrix(calib_file)
                T_lidar = utils.get_lidar2cam(lidar_calib_file)
                points_3d_l2c = utils.trans_point_coor(l_in, T_lidar)
                
                depth_map_before, depth_map_after = generate_depth_map(l_in_2d, points_3d_l2c, image_width, image_height)
                
                # 保存新生成的深度图
                np.save(regen_file_path, depth_map_after)
                print(f"Regenerated depth map for frame {frame}.")
            
            except Exception as regen_error:
                print(f"Failed to regenerate depth map for frame {frame}: {regen_error}")



if __name__ == '__main__':
    start_frame = 0
    end_frame = 9930
    depth_map_dir = '/workspace/data/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/lidar_inter2d_matrix/'
    regen_dir = '/workspace/data/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/lidar_inter2d_matrix/'
    image_width, image_height = 1935, 644

    check_and_regenerate_depth_maps(start_frame, end_frame, depth_map_dir, regen_dir, image_width, image_height)

    # frame = 0
    # for frame in range(0,9931):
    #     radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(frame)
    #     if not (os.path.exists(radar_file) and os.path.exists(calib_file)):
    #         print(f'File not exist for frame {frame}!')
    #         continue
    #     else:
    #         print(f'dealing with frame {frame}')
    #     l_in_2d = np.load(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/lidar_2d/{frame}.npy')
    #     l_in = np.load(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/lidar2d_corr_3dpoints/{frame}.npy')
    #     K = utils.get_intrinsic_matrix(calib_file)
    #     T_lidar = utils.get_lidar2cam(lidar_calib_file)
    #     fx, cx = K[0, 0], K[0, 2]
    #     fy, cy = K[1, 1], K[1, 2]
    #     #l_in_downsampled = farthest_point_sampling(l_in,4096)
    #     points_3d_l2c = utils.trans_point_coor(l_in, T_lidar)
    #     #l_in_2d_downsampled = utils.project_3d_to_2d(points_3d_l2c, fx, fy, cx, cy)
    #     image_width, image_height = 1935,644

    #     # 生成深度图
    #     depth_map_before, depth_map_after = generate_depth_map(l_in_2d, points_3d_l2c, image_width, image_height)
    #     np.save(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/lidar_inter2d_matrix/{frame}.npy',depth_map_after)
    #     save_path = f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/lidar_inter2d_pic/{frame}.jpg'
    #     save_depth_map_as_image(depth_map_after, save_path)
    #     # non_zero_before, coverage_before = calculate_coverage_and_count(depth_map_before)
    #     # non_zero_after, coverage_after = calculate_coverage_and_count(depth_map_after)

    #     # print(f"插值前具有深度值的像素点数量: {non_zero_before}")
    #     # print(f"插值前具有深度值的像素点百分比: {coverage_before:.2f}%")
    #     # print(f"插值后具有深度值的像素点数量: {non_zero_after}")
    #     # print(f"插值后具有深度值的像素点百分比: {coverage_after:.2f}%")
    #     #
    #     # visualize_depth_map(depth_map_before, "Depth Map Before Interpolation (Inverted)")
    #     # visualize_depth_map(depth_map_after, "Depth Map After Interpolation (Inverted)")

    # # image = cv2.imread(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/image_2/{str(frame).zfill(5)}.jpg')
    # # image1 = utils.visualize_with_image_color(image,l_in_2d_downsampled,[255,0,0])
    # # cv2.imshow('projected',image1)
    # # cv2.waitKey(0)