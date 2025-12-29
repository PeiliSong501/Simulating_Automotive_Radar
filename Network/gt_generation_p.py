import math

import pandas as pd
import numpy as np
import os
import cv2
# import utils
import random
# from sklearn.model_selection import train_test_split
import re
from scipy.spatial import KDTree
from collections import defaultdict
from scipy.ndimage import gaussian_filter
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
# import joblib
# import matplotlib.pyplot as plt
# from sklearn.cluster import DBSCAN
# from scipy.spatial.distance import euclidean
from concurrent.futures import ProcessPoolExecutor
# from joblib import Parallel, delayed
import concurrent.futures

def gen_range_image(local_points, fov_up, fov_down, yaw1, yaw2, proj_H, proj_W, dst):
    #print("len of local_points:",len(local_points))
    fov_up = fov_up / 180.0 * np.pi
    fov_down = fov_down / 180.0 * np.pi
    #fov = abs(fov_down) + abs(fov_up)
    fov = fov_up-fov_down
    depth = np.linalg.norm(local_points[:, :3], 2, axis=1)
    scan_x = local_points[:, 0]
    scan_y = local_points[:, 1]
    scan_z = local_points[:, 2]
    # yaw = -np.arctan2(scan_y, scan_x)
    yaw = np.arctan2(scan_y, scan_x)
    #pitch = np.arcsin(scan_z / depth)
    pitch = np.arctan2(scan_z, scan_x)
    print(np.degrees(yaw),np.degrees(pitch))
    #proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    # proj_x = 0.5 * (np.degrees(yaw) / (abs(yaw1) + abs(yaw2)) + 1.0)
    proj_x = 1.0 - (np.degrees(yaw) - yaw2) / (yaw1 - yaw2)
    #proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch - fov_down) / fov

    proj_x *= proj_W  # in [0.0, W]
    proj_y *= proj_H  # in [0.0, H]

    print(proj_x,proj_y)


    proj_x = np.floor(proj_x)
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]

    order = np.argsort(depth)[::-1]
    depth = depth[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    proj_range = np.full((proj_H, proj_W), -1,
                         dtype=np.float32)
    #proj_range[proj_y, proj_x] = depth*100  #multiply depth by 100, or the color will not be able tell
    proj_range[proj_y, proj_x] = depth*100
    cv2.imwrite(dst, proj_range)

    return


def gen_range_image_rcs_translation(local_points, proj_H, proj_W, r_l, dst):
    print("len of local_points:", len(local_points))

    # 计算深度
    depth = local_points[:, 2]

    scan_x = local_points[:, 0]
    scan_y = local_points[:, 1]

    # 将点投影到图像坐标
    # proj_x = 1 + (scan_x - r_l / 2 * r_l)  # in [0.0, 1.0]
    # proj_y = 1 + (scan_y - r_l / 2 * r_l)  # in [0.0, 1.0]
    proj_x = 0.5 * (-scan_x / r_l + 1.0)  # in [0.0, 1.0]
    proj_y = 1 - (scan_y + r_l / 2 * r_l)  # in [0.0, 1.0]

    proj_x *= proj_W  # in [0.0, W]
    proj_y *= proj_H  # in [0.0, H]
    proj_x = np.floor(proj_x).astype(np.int32)
    proj_y = np.floor(proj_y).astype(np.int32)

    proj_x = np.clip(proj_x, 0, proj_W - 1)
    proj_y = np.clip(proj_y, 0, proj_H - 1)

    # 对深度值进行排序并根据排序重新排列投影坐标
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    # 初始化投影矩阵和计数矩阵
    proj_range = np.zeros((proj_H, proj_W), dtype=np.float32)
    proj_count = np.zeros((proj_H, proj_W), dtype=np.int32)

    # 遍历所有点，将深度值累加并计数
    for i in range(len(depth)):
        x, y = proj_x[i], proj_y[i]
        proj_range[y, x] += depth[i]
        proj_count[y, x] += 1

    # 计算每个像素的平均深度值，避免除以零
    valid_mask = proj_count > 0
    proj_range[valid_mask] /= proj_count[valid_mask]

    # 归一化到 [0, 255] 范围内
    #range_norm = np.minimum((1 + (proj_range - r_l) / (2 * r_l)) * 255, 254)
    range_norm = np.minimum((depth / r_l) * 255, 254)
    proj_range[valid_mask] = range_norm[valid_mask]

    # 保存图像
    cv2.imwrite(dst, proj_range)

    return proj_range


def gen_range_image_rcs_translation_correct(local_points, original_points, proj_H, proj_W, target_point, r_l, dst):
    print("len of local_points:", len(local_points))

    # Calculate depths
    depth = np.linalg.norm(local_points[:, :3], 2, axis=1)
    range_original = np.linalg.norm(original_points[:, :3], 2, axis=1)
    range_target = np.linalg.norm(target_point[:3], 2, axis=0)
    scan_x = local_points[:, 1]
    scan_y = local_points[:, 2]

    # Project points to image coordinates
    proj_x = 0.5 * (-scan_x / r_l + 1.0)  # in [0.0, 1.0]
    proj_y = 1 - (scan_y + r_l / 2 * r_l)  # in [0.0, 1.0]

    proj_x *= proj_W  # in [0.0, W]
    proj_y *= proj_H  # in [0.0, H]
    proj_x = np.floor(proj_x).astype(np.int32)
    proj_y = np.floor(proj_y).astype(np.int32)

    proj_x = np.clip(proj_x, 0, proj_W - 1)
    proj_y = np.clip(proj_y, 0, proj_H - 1)

    order = np.argsort(depth)[::-1]
    depth = depth[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]
    range_original = range_original[order]
    #range_target = range_target[0]  # Single target range

    # Initialize projection matrix
    proj_range = np.full((proj_H, proj_W), -1, dtype=np.float32)
    # Calculate grayscale values according to the given condition
    for i in range(len(depth)):
        dist_diff = np.linalg.norm(local_points[i, :3]) / (2 * r_l)
        if range_original[i] >= range_target:
            grayscale_value = 127 + np.ceil(dist_diff * 255).astype(int)
        else:
            grayscale_value = 127 - np.floor(dist_diff * 255).astype(int)

        proj_range[proj_y[i], proj_x[i]] = np.clip(grayscale_value, 0, 255)

    # Save the image
    success = cv2.imwrite(dst, proj_range)
    print(f"Image saved at {dst}: {success}")

    return proj_range


def gen_range_image_rcs(local_points, fov_up, fov_down, proj_H, proj_W, dst):
    print("len of local_points:",len(local_points))
    fov_up = fov_up / 180.0 * np.pi
    fov_down = fov_down / 180.0 * np.pi
    fov = abs(fov_down) + abs(fov_up)
    depth = np.linalg.norm(local_points[:, :3], 2, axis=1)
    scan_x = local_points[:, 0]
    scan_y = local_points[:, 1]
    scan_z = local_points[:, 2]
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

    proj_x *= proj_W  # in [0.0, W]
    proj_y *= proj_H  # in [0.0, H]
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]

    order = np.argsort(depth)[::-1]
    depth = depth[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    proj_range = np.full((proj_H, proj_W), -1,
                         dtype=np.float32)
    proj_range[proj_y, proj_x] = depth*100  #multiply depth by 100, or the color will not be able tell

    cv2.imwrite(dst, proj_range)

    return


def frame_index(filename):
    match = re.search(r'(\d+)_(\d+)\.jpg', filename)
    if not match:
        raise ValueError(f"Filename {filename} does not match the expected pattern 'frame_index.jpg'")
    frame = int(match.group(1))
    index = int(match.group(2))
    return frame, index


def radar2image(pcd_radar,T_radar,intrinsic):
    shape_image  = [1936, 1216]
    geo_radar = pcd_radar[:, 0:3]
    points_3d_r2c = utils.trans_point_coor(geo_radar, T_radar)
    fx, cx = intrinsic[0, 0], intrinsic[0, 2]
    fy, cy = intrinsic[1, 1], intrinsic[1, 2]
    points_2d_radar = utils.project_3d_to_2d(points_3d_r2c, fx, fy, cx, cy)
    x_coords = points_2d_radar[:, 0]
    y_coords = points_2d_radar[:, 1]
    mask_x = (x_coords >= 0) & (x_coords < shape_image[0])
    mask_y = (y_coords >= 0) & (y_coords < shape_image[1])
    mask = mask_x & mask_y
    filtered_points_2d = points_2d_radar[mask]
    return filtered_points_2d


def radar_in_image(coor, pcd_radar, T_radar, intrinsic):
    x1, y1, x2, y2 = coor
    geo_radar = pcd_radar[:, 0:3]
    # 转换点云坐标系
    points_3d_r2c = utils.trans_point_coor(geo_radar, T_radar)

    # 相机内参
    fx, cx = intrinsic[0, 0], intrinsic[0, 2]
    fy, cy = intrinsic[1, 1], intrinsic[1, 2]

    # 将3D点投影到2D图像平面
    points_2d_radar = utils.project_3d_to_2d(points_3d_r2c, fx, fy, cx, cy)

    # 仅保留在给定坐标范围内的点
    x_coords = points_2d_radar[:, 0]
    y_coords = points_2d_radar[:, 1]
    mask_x = (x_coords >= x1) & (x_coords < x2)
    mask_y = (y_coords >= y1) & (y_coords < y2)
    mask = mask_x & mask_y

    # 筛选出对应的 r_in 和 points_2d_radar
    r_in = pcd_radar[mask]
    r_in_2d = points_2d_radar[mask]

    # 将2D点转换为整数坐标
    r_in_2d_int = np.floor(r_in_2d).astype(int)

    # 创建一个字典用于存储不重复的点，键为坐标，值为对应的深度和索引
    unique_points = {}

    for i, (x, y) in enumerate(r_in_2d_int):
        depth = r_in[i, 0]  # r_in 的第 0 列为深度信息
        key = (x, y)

        if key in unique_points:
            # 如果坐标已存在，保留深度较小的点
            if depth < unique_points[key][0]:
                unique_points[key] = (depth, i)
        else:
            unique_points[key] = (depth, i)

    # 从字典中提取最终的 r_in 和 r_in_2d
    indices = [v[1] for v in unique_points.values()]
    r_in_final = r_in[indices]
    r_in_2d_final = r_in_2d_int[indices]


    return r_in_final, r_in_2d_final


def radar_in_lidarfov(pcd_radar, T_radar, T_lidar):
    mask = []
    for i in range(len(pcd_radar)):
        geo = pcd_radar[i][0:3]
        geo_lidarcoor = transform_coor_to_radar(geo, np.dot(np.linalg.inv(T_lidar), T_radar))
        # yaw = np.arctan2(geo_lidarcoor[1], geo_lidarcoor[0])
        pitch = np.arctan2(geo_lidarcoor[2], np.linalg.norm(geo_lidarcoor[:2]))
        # yaw_deg = np.degrees(yaw)
        pitch_deg = np.degrees(pitch)
        if -23 < pitch_deg < 7:
            mask.append(i)
    radar = pcd_radar[mask,:]
    return radar

def lidar_in_radarfov(pcd_lidar, T_radar, T_lidar):
    mask = []
    pcd_lidar = utils.filter_vlp16(pcd_lidar)
    pcd_lidar = pcd_lidar[pcd_lidar[:,0] > 0]
    for i in range(len(pcd_lidar)):
        geo = pcd_lidar[i][0:3]
        geo_radarcoor = transform_coor_to_radar(geo, np.dot(np.linalg.inv(T_radar), T_lidar))
        yaw = np.arctan2(geo_radarcoor[1], geo_radarcoor[0])
        pitch = np.arctan2(geo_radarcoor[2], np.linalg.norm(geo_radarcoor[:2]))
        yaw_deg = np.degrees(yaw)
        pitch_deg = np.degrees(pitch)
        if -15 < pitch_deg < 15 and -60 < yaw_deg < 60:
            mask.append(i)


    lidar = pcd_lidar[mask, :]
    return lidar


def lidar2image(pcd_lidar,T_lidar,intrinsic):
    shape_image  = [1936, 1216]
    geo_lidar = pcd_lidar[:, 0:3]
    points_3d_r2c = utils.trans_point_coor(geo_lidar, T_lidar)
    fx, cx = intrinsic[0, 0], intrinsic[0, 2]
    fy, cy = intrinsic[1, 1], intrinsic[1, 2]
    points_2d_lidar = utils.project_3d_to_2d(points_3d_r2c, fx, fy, cx, cy)
    return points_2d_lidar


def lidar2uv_depth(pcd_lidar,T_lidar,intrinsic, coor):
    point1, point2 = coor
    x1, y1 = point1
    x2, y2 = point2
    geo_lidar = pcd_lidar[:, 0:3]
    points_3d_r2c = utils.trans_point_coor(geo_lidar, T_lidar)
    fx, cx = intrinsic[0, 0], intrinsic[0, 2]
    fy, cy = intrinsic[1, 1], intrinsic[1, 2]
    points_2d_lidar = utils.project_3d_to_2d(points_3d_r2c, fx, fy, cx, cy)
    x_coords = points_2d_lidar[:, 0]
    y_coords = points_2d_lidar[:, 1]
    mask_x = (x_coords >= x1) & (x_coords < x2)
    mask_y = (y_coords >= y1) & (y_coords < y2)
    mask = mask_x & mask_y
    lidar_uv_depth = np.full((100, 100), -1)
    cnt_depth = 0
    for i in range(len(mask)):
        if mask[i] == True:
            u,v = points_2d_lidar[i]
            depth = geo_lidar[i,0]
            u_new, v_new = int(np.floor(u-x1)), int(np.floor(v-y1))
            if lidar_uv_depth[u_new,v_new] == -1:
                lidar_uv_depth[u_new,v_new] = depth
                cnt_depth += 1
            else:
                if depth < lidar_uv_depth[u_new,v_new]:
                    lidar_uv_depth[u_new,v_new] = depth
    return lidar_uv_depth, cnt_depth



def pixel_to_radar_angles(u, v, K, T):
    K_inv = np.linalg.inv(K)

    v_camera = K_inv @ np.array([u, v, 1])

    R = T[:3, :3]
    R_inv = np.linalg.inv(R)

    v_radar = R_inv @ v_camera
    yaw = np.arctan2(v_radar[1], v_radar[0])
    pitch = np.arctan2(v_radar[2], np.linalg.norm(v_radar[:2]))
    yaw_deg = np.degrees(yaw)
    pitch_deg = np.degrees(pitch)

    return yaw_deg, pitch_deg


def random_crop(image, crop_size, num_crops, Ru_min, Ru_max, Rv_min, Rv_max, seed=None):
    h, w, _ = image.shape
    crops = []
    coordinates = []
    if seed is not None:
        random.seed(seed)

    cnt = 0
    while cnt < num_crops:
        x_start = random.randint(0, w - crop_size)
        y_start = random.randint(0, h - crop_size)

        crop = image[y_start:y_start + crop_size, x_start:x_start + crop_size]
        u_min,v_min = (x_start, y_start)
        u_max,v_max = (x_start + crop_size, y_start + crop_size)
        if u_min >= Ru_min and u_max <= Ru_max and v_min >= Rv_min and v_max <= Rv_max:
            cnt += 1
            crops.append(crop)
            top_left = [x_start, y_start]
            bottom_right = [x_start + crop_size, y_start + crop_size]
            coordinates.append([top_left, bottom_right])

    return crops, coordinates


def sliding_window_crop(image, crop_size, Ru_min, Ru_max, Rv_min, Rv_max):
    h, w, _ = image.shape
    crops = []
    coordinates = []

    step_size = 2*crop_size

    for y_start in range(0, h - crop_size + 1, step_size):
        for x_start in range(0, w - crop_size + 1, step_size):
            crop = image[y_start:y_start + crop_size, x_start:x_start + crop_size]

            u_min, v_min = (x_start, y_start)
            u_max, v_max = (x_start + crop_size, y_start + crop_size)

            if Ru_min <= u_min < u_max <= Ru_max and Rv_min <= v_min < v_max <= Rv_max:
                crops.append(crop)
                top_left = [x_start, y_start]
                bottom_right = [x_start + crop_size, y_start + crop_size]
                coordinates.append([top_left, bottom_right])

    return crops, coordinates


def p_in(u1,v1,u2,v2,radar_uv):
    sign = 0
    for i in range(len(radar_uv)):
        u,v = radar_uv[i]
        if u1 <= u <= u2 and v1 <= v <= v2:
            sign = 1
            break
    return sign

def frame_index(filename):
    name, _ = filename.split('.')
    frame, index = map(int, name.split('_'))
    return frame, index



def remove_point_if_exists(local_radar_pcd, radarpoint):
    for i, point in enumerate(local_radar_pcd):
        if np.allclose(point, radarpoint[0:3]):
            return np.delete(local_radar_pcd, i, axis=0), True
    return local_radar_pcd, False

def get_new_point(x, y, z, r):
    point = np.array([x, y, z])
    vector = point - np.array([0, 0, 0])
    unit_vector = vector / np.linalg.norm(vector)
    new_point = point - unit_vector * r
    return new_point


def compute_pitch_angles(points):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    horizontal_distance = np.sqrt(x**2 + y**2)

    pitch_angles = np.degrees(np.arctan2(z, horizontal_distance))

    return pitch_angles

def generate_pitch_mask(pitch_angles, min_angle=-15, max_angle=15):
    mask = (pitch_angles >= min_angle) & (pitch_angles <= max_angle)
    return mask

def transform_coor_to_radar(location, radar_transform_matrix):
    # T_inv = np.linalg.inv(radar_transform_matrix)
    location_homogeneous = np.array([location[0], location[1], location[2], 1])
    location_radar_homogeneous = np.dot(radar_transform_matrix, location_homogeneous)
    location_radar = location_radar_homogeneous[:3]
    return location_radar


def spatial_partition(width, max_range, h_fov, v_fov, T_radar, T_lidar, intrinsic):
    #VLP-16 v_fov:-15~15
    max_y = max_range * np.tan(np.radians(h_fov / 2))
    max_z = max_range * np.tan(np.radians(v_fov / 2))
    tany = np.tan(np.radians(h_fov / 2))
    tanz = np.tan(np.radians(v_fov / 2))
    num_voxel_x = int(np.ceil(max_range/width))
    num_voxel_y = 2*int(np.ceil(max_y/width))
    num_voxel_z = 2*int(np.ceil(max_z/width))
    center_list = []
    pixel_center_list = []
    for i in range(num_voxel_x):
        for j in range(num_voxel_y):
            for k in range(num_voxel_z):
                x_center = i * width + 0.5 * width      # back to front
                y_center = j * width + 0.5 * width - max_y      # right to left
                z_center = k * width + 0.5 * width - max_z      # bottom to top
                ymax = (x_center + 0.5*width)*tany
                zmax = (x_center + 0.5*width)*tanz
                if y_center + width < -ymax or y_center - width > ymax:
                    continue
                elif z_center + width < -zmax or z_center - width > zmax:
                    continue
                elif x_center > 100:
                    continue
                else:
                    centerpoint = np.array([x_center, y_center, z_center])
                    #vertices = gen_voxel_vertices(np.array([x_center, y_center, z_center]),width)
                    centerpoint_2d = radar2image(np.array([centerpoint]), T_radar, intrinsic)
                    if len(centerpoint_2d) == 1:
                        centerpoint = np.array([x_center, y_center, z_center])
                        cpoint_lidarcoor = transform_coor_to_radar(centerpoint, np.dot(np.linalg.inv(T_lidar), T_radar))
                        x,y,z = cpoint_lidarcoor
                        pitch = np.degrees(np.arctan(z / x))
                        if -15 <= pitch <= 15:
                            center_list.append(centerpoint)
                            u,v = int(np.floor(centerpoint_2d[0,0])),int(np.floor(centerpoint_2d[0,1]))
                            pixel_center_list.append(np.array([u,v]))
                            print(pixel_center_list[-1])
    return np.array(center_list), np.array(pixel_center_list)

def gen_voxel_vertices(center, width):
    x, y, z = center
    w = width * 0.5
    vertices = [
        [x-w, y-w, z-w],
        [x+w, y-w, z-w],
        [x-w, y+w, z-w],
        [x+w, y+w, z-w],
        [x-w, y-w, z+w],
        [x+w, y-w, z+w],
        [x-w, y+w, z+w],
        [x+w, y+w, z+w]
    ]
    vertices_array = np.array(vertices)
    return vertices_array


def crop_patch(image, center, patch_size=50):
    u, v = center
    img_h, img_w = image.shape[:2]
    half_patch = patch_size // 2

    # Calculate initial crop boundaries
    u_min = u - half_patch
    u_max = u + half_patch
    v_min = v - half_patch
    v_max = v + half_patch

    # Adjust the boundaries if they exceed image borders
    if u_min < 0:
        u_max = min(u_max + abs(u_min), img_w)
        u_min = 0
    if u_max > img_w:
        u_min = max(u_min - (u_max - img_w), 0)
        u_max = img_w

    if v_min < 0:
        v_max = min(v_max + abs(v_min), img_h)
        v_min = 0
    if v_max > img_h:
        v_min = max(v_min - (v_max - img_h), 0)
        v_max = img_h

    # Crop the patch
    cropped_patch = image[v_min:v_max, u_min:u_max]

    return cropped_patch, np.array([u_min, u_max, v_min, v_max])


def sample_voxels(center_list, num_frames=9931, frame_interval=15, voxels_per_frame=200,
                  save_dir="D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/voxel_dataset/sample_voxel/"):
    np.random.seed(42)  # For reproducibility
    total_frames = num_frames // frame_interval + 1
    total_voxels = len(center_list)

    # Convert center_list to NumPy array if it's not already
    if not isinstance(center_list, np.ndarray):
        all_centers = np.array(center_list)
    else:
        all_centers = center_list

    # Create directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)

    # Initialize variable to store previous frame's selected voxel centers
    prev_selected_centers = None

    for frame_idx in range(total_frames):
        frame_number = frame_idx * frame_interval
        print(f"Processing frame {frame_number}...")

        if prev_selected_centers is None:
            # For the first frame, randomly select voxels
            selected_indices = np.random.choice(total_voxels, size=voxels_per_frame, replace=False)
        else:
            # Compute distances from all voxels to previously selected voxel centers
            distances = np.linalg.norm(all_centers[:, np.newaxis, :] - prev_selected_centers[np.newaxis, :, :], axis=2)
            min_distances = distances.min(axis=1)  # Minimal distance to any of the previously selected voxels

            # Get indices of voxels sorted by maximal minimal distance
            sorted_indices = np.argsort(-min_distances)

            # Select top voxels_per_frame voxels that are farthest from previous selections
            # Also ensure that we do not select the same voxels again
            selected_indices = []
            selected_set = set(prev_selected_indices)  # To avoid re-selection
            for idx in sorted_indices:
                if idx not in selected_set:
                    selected_indices.append(idx)
                if len(selected_indices) == voxels_per_frame:
                    break
            selected_indices = np.array(selected_indices)

        # Save the indices for the current frame
        save_path = os.path.join(save_dir, f'{frame_number}.npy')
        np.save(save_path, selected_indices)

        # Update previous selections
        prev_selected_centers = all_centers[selected_indices]
        prev_selected_indices = selected_indices

        print(f'Frame {frame_number}: {len(selected_indices)} voxels sampled and saved to {save_path}.')


def sample_points_from_point_cloud(point_cloud, point_cloud_2d, n):
    np.random.seed(2024)
    points = point_cloud[:, :3]
    num_points = points.shape[0]
    #
    # # 随机选择第一个点
    # sampled_indices = [np.random.randint(num_points)]
    # distances = np.linalg.norm(points - points[sampled_indices[-1]], axis=1)
    #
    # for _ in range(1, n):
    #     farthest_point_index = np.argmax(distances)
    #     sampled_indices.append(farthest_point_index)
    #     # 更新距离
    #     new_distances = np.linalg.norm(points - points[farthest_point_index], axis=1)
    #     distances = np.minimum(distances, new_distances)
    #
    # sampled_indices = np.array(sampled_indices)
    # sampled_points = point_cloud[sampled_indices, :]
    # sampled_points_2d = point_cloud_2d[sampled_indices, :]
    if n > num_points:
        n = num_points
    sampled_indices = np.random.choice(num_points, size=n, replace=False)

    # 选取对应的点和它们在2D投影中的位置
    sampled_points = point_cloud[sampled_indices, :]
    sampled_points_2d = point_cloud_2d[sampled_indices, :]
    return sampled_points, sampled_points_2d


def image_in_fov(image, T, K, h_fov, v_fov_min, v_fov_max):
    height, width = image.shape[:2]

    def is_in_fov(u, v, K, T, h_fov, v_fov_min, v_fov_max):
        yaw_deg, pitch_deg = pixel_to_radar_angles(u, v, K, T)
        return -h_fov / 2 <= yaw_deg <= h_fov / 2 and v_fov_min <= pitch_deg <= v_fov_max

    x_min, y_min, x_max, y_max = width, height, 0, 0

    for x in range(width):
        for y in range(height):
            if is_in_fov(x, y, K, T, h_fov, v_fov_min, v_fov_max):
                x_min = x
                break
        if x_min < width:
            break

    for x in range(width - 1, -1, -1):
        for y in range(height):
            if is_in_fov(x, y, K, T, h_fov, v_fov_min, v_fov_max):
                x_max = x
                break
        if x_max > 0:
            break

    for y in range(height):
        for x in range(width):
            if is_in_fov(x, y, K, T, h_fov, v_fov_min, v_fov_max):
                y_min = y
                break
        if y_min < height:
            break

    for y in range(height - 1, -1, -1):
        for x in range(width):
            if is_in_fov(x, y, K, T, h_fov, v_fov_min, v_fov_max):
                y_max = y
                break
        if y_max > 0:
            break

    return [x_min, y_min, x_max, y_max]


def radar_to_image_resolution(angle_resolution_deg_h, angle_resolution_deg_v, K):
    # 将角度分辨率从度转换为弧度
    angle_resolution_rad_h = np.radians(angle_resolution_deg_h)
    angle_resolution_rad_v = np.radians(angle_resolution_deg_v)

    # 计算雷达点的空间分辨率
    # spatial_resolution_h = depth * angle_resolution_rad_h
    # spatial_resolution_v = depth * angle_resolution_rad_v

    # 从相机内参矩阵提取焦距
    fx = K[0, 0]
    fy = K[1, 1]

    # 将空间分辨率转换为图像上的像素分辨率
    pixel_resolution_x = (angle_resolution_rad_h * fx)
    pixel_resolution_y = (angle_resolution_rad_v * fy)

    return pixel_resolution_x, pixel_resolution_y



def convert_rcs_dbsm_to_m2(rcs_dbsm):
    rcs_m2 = 10 ** (rcs_dbsm / 10)
    return rcs_m2


def model_gaussian(center, sigma_x, sigma_y):
    #print('sigma_x,sigma_y',sigma_x,sigma_y)
    # Mean is the center of the radar point in image coordinates
    mean = center

    # Define the 2D Gaussian distribution with the given parameters
    def gaussian(x, y):
        return (1 / (2 * np.pi * sigma_x * sigma_y)) * np.exp(-0.5 * (((x - mean[0]) / sigma_x) ** 2 + ((y - mean[1]) / sigma_y) ** 2))

    return gaussian




def update_radar_map_with_gaussian(radar_2d_map, r_in, r_in_2d, std_var):
    """
    遍历雷达图像中的点，对周边像素点应用高斯分布
    :param radar_2d_map: 雷达图像二维矩阵
    :param r_in: 点云列表，其中第0列是深度信息
    :param r_in_2d: 投影到二维平面的点云列表
    :param pixel_resolution: 像素分辨率 [res_x, res_y]
    :return: 更新后的概率密度矩阵
    """
    # 获取图像的尺寸
    rows, cols = radar_2d_map.shape

    radar_2d_map_pro = np.zeros((rows, cols))
    # rcs_list = r_in[:,3]
    # rcs_list = convert_rcs_dbsm_to_m2(rcs_list)
    # max_rcs = np.max(rcs_list)


    for i in range(len(r_in_2d)):
        #print(i)
        u, v = int(np.floor(r_in_2d[i, 0])), int(np.floor(r_in_2d[i, 1]))
        #depth = r_in[i, 0]  # 获取深度信息
        #rcs = rcs_list
        #print(u,v)
        v = v-571
        if 0 <= u < cols and 0 <= v < rows:
            # 计算标准差
            #print(i)

            # sigma_x, sigma_y = sigma_x_list[i], sigma_y_list[i]
            sigma_x, sigma_y = std_var[0], std_var[1]
            # 标准差的范围
            # range_x = min(int(np.ceil(3 * sigma_x)),500)
            # range_y = min(int(np.ceil(3 * sigma_y)),500)
            left_range = max(-int(np.ceil(3 * sigma_x)), -u)
            right_range = min(int(np.ceil(3 * sigma_x)), cols - u - 1)
            top_range = max(-int(np.ceil(3 * sigma_y)), -v)
            bottom_range = min(int(np.ceil(3 * sigma_y)), rows - v - 1)

            gaussian_fn = model_gaussian((u, v), sigma_x, sigma_y)

            # 对周边像素点应用高斯分布
            for di in range(top_range, bottom_range + 1):
                for dj in range(left_range, right_range + 1):
                    ni, nj = v + di, u + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        # 计算高斯分布的概率值
                        prob = gaussian_fn(nj, ni)  # 注意，传入参数应为 (y, x)
                        radar_2d_map_pro[ni, nj] += prob
                        radar_2d_map_pro[ni, nj] = radar_2d_map_pro[ni, nj] + prob
    total = np.sum(radar_2d_map_pro)
    for i in range(radar_2d_map_pro.shape[0]):
        for j in range(radar_2d_map_pro.shape[1]):
            radar_2d_map_pro[i,j] /= total
    return radar_2d_map_pro


def calculate_yaw_pitch(x1, y1, z1, x2, y2, z2):
    yaw1 = np.arctan2(y1, x1)
    pitch1 = np.arctan2(z1, np.sqrt(x1 ** 2 + y1 ** 2))

    yaw2 = np.arctan2(y2, x2)
    pitch2 = np.arctan2(z2, np.sqrt(x2 ** 2 + y2 ** 2))

    yaw_diff = np.degrees(yaw2 - yaw1)
    pitch_diff = np.degrees(pitch2 - pitch1)

    return yaw_diff, pitch_diff


def generate_colors(num_colors):
    np.random.seed(42)  # 固定种子以确保可重复性
    return np.random.randint(0, 255, size=(num_colors, 3), dtype=np.uint8)

def calculate_distances_to_bbox_edges(points, bbox):
    xmin, ymin, xmax, ymax = bbox
    distances = {
        'xmin': np.min(abs(xmin - points[:, 0])),
        'ymin': np.min(abs(ymin - points[:, 1])),
        'xmax': np.min(abs(points[:, 0] - xmax)),
        'ymax': np.min(abs(points[:, 1] - ymax))
    }
    return distances


def process_frame(frame):
    if os.path.exists(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/pmap_image/{frame}.jpg'):
        return

    print(f'Processing frame {frame}')
    radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(frame)

    if not (os.path.exists(radar_file) and os.path.exists(calib_file)):
        print(f'File not exist for frame {frame}!')
        return

    K = utils.get_intrinsic_matrix(calib_file)
    T_radar = utils.get_radar2cam(calib_file)
    T_lidar = utils.get_lidar2cam(lidar_calib_file)

    radar_pcd = np.fromfile(radar_file, dtype=np.float32).reshape(-1, 7)
    lidar_pcd = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
    radar_pcd = radar_in_lidarfov(radar_pcd, T_radar, T_lidar)
    radar_pcd = radar_pcd[radar_pcd[:, 0] > 0]

    image = cv2.imread(cam_file)
    coor = [0, 571, 1935, 1215]  # 这个坐标是你代码中固定的

    r_in, r_in_2d = radar_in_image(coor, radar_pcd, T_radar, K)
    np.save(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/r_in/{frame}.npy',r_in)
    np.save(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/r_in_2d/{frame}.npy', r_in_2d)


    radar_2d_map = np.zeros((644, 1935))

    standard_variance = [100, 100]
    radar_2d_map_pro = update_radar_map_with_gaussian(radar_2d_map, r_in, r_in_2d, standard_variance)

    np.save(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/radar_2d_map_pro/{frame}.npy',
            radar_2d_map_pro)

    normalized_array = cv2.normalize(radar_2d_map_pro, None, 0, 255, cv2.NORM_MINMAX)
    gray_image_pro = normalized_array.astype(np.uint8)
    cv2.imwrite(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/pmap_image/{frame}.jpg',
                gray_image_pro)





if __name__ == '__main__':
    # frame = 0
    # radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(frame)
    # K = utils.get_intrinsic_matrix(calib_file)
    # T_radar = utils.get_radar2cam(calib_file)
    # T_lidar = utils.get_lidar2cam(lidar_calib_file)
    # image = cv2.imread(cam_file)
    # coor = image_in_fov(image,T_lidar,K,180,-23,7)
    # print(coor)
#----------------------------------------------convert radar RCS to m^2-------------------------------------------------
    # pcd_radar = np.fromfile(radar_file,dtype=np.float32).reshape(-1,7)
    # rcs = pcd_radar[:,3]
    # print(rcs[0:5])
    # print(np.min(rcs),np.max(rcs))
    # rcs_m2 = convert_rcs_dbsm_to_m2(rcs)
    # print(rcs_m2[0:5])
    # print(np.min(rcs_m2),np.max(rcs_m2))




#------------------------------------------generate radar probabilistic image------------------------------------------
    #frames = range(0, 9931)

    #使用joblib并行处理
    #Parallel(n_jobs=8)(delayed(process_frame)(frame) for frame in frames)
    # for frame in range(0,9931, 1):
    #     # if os.path.exists(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/pmap_image/{frame}.jpg'):
    #     #     continue
    #     print(f'frame {frame}')
    #     radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(frame)
    #     if not (os.path.exists(radar_file) and os.path.exists(calib_file)):
    #         print('file not exist!')
    #         continue
    #     K = utils.get_intrinsic_matrix(calib_file)
    #     T_radar = utils.get_radar2cam(calib_file)
    #     T_lidar = utils.get_lidar2cam(lidar_calib_file)
    #
    #     radar_pcd = np.fromfile(radar_file, dtype=np.float32).reshape(-1, 7)
    #
    #     lidar_pcd = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
    #     lidar_pcd = utils.filter_vlp16(lidar_pcd)
    #     lidar_pcd =  lidar_pcd[lidar_pcd[:,0]>0]
    #     radar_pcd = radar_in_lidarfov(radar_pcd,T_radar,T_lidar)
    #     radar_pcd = radar_pcd[radar_pcd[:,0] > 0]
    #     image = cv2.imread(cam_file)
    #
    #
    #     coor = [0, 571, 1935, 1215]
    #     r_in, r_in_2d = radar_in_image(coor, radar_pcd,T_radar,K)
    #     #np.save(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/radar_in/{frame}.npy',r_in)
    #     rcs_list = r_in[:,3]
    #     imgage = cv2.imread(cam_file)
    #
    #     #radar_2d_map = np.zeros((896,1935))
    #
    #     if os.path.exists(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/radar_2d_map_pro/{frame}.npy'):
    #         radar_2d_map_pro = np.load(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/radar_2d_map_pro/{frame}.npy')
    #         radar_2d_map_pro = radar_2d_map_pro[252:,0:1935]
    #
    #     # standard_variance = [100,100]
    #     # radar_2d_map_pro = update_radar_map_with_gaussian(radar_2d_map, r_in, r_in_2d, standard_variance)
    #     #np.save(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/radar_2d_map_pro/{frame}.npy', radar_2d_map_pro)
    #
    #         # normalized_array = cv2.normalize(radar_2d_map_pro, None, 0, 255, cv2.NORM_MINMAX)
    #         # gray_image_pro = normalized_array.astype(np.uint8)
    #
    #         # cv2.imwrite(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/pmap_image/{frame}.jpg', gray_image_pro)
    #         gray_image_pro = cv2.imread(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/pmap_image/{frame}.jpg')
    #         gray_image_pro = gray_image_pro[252:,0:1935]
    #         cv2.imwrite(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/pmap_image/{frame}.jpg', gray_image_pro)
    #
    #
    #
    # # radar_2d_map_pro_image = cv2.imread('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/pmap.jpg', cv2.IMREAD_GRAYSCALE)
    # # radar_2d_map_pro_image = cv2.applyColorMap(radar_2d_map_pro_image, cv2.COLORMAP_JET)
    # # alpha = 0.5
    # # overlay = cv2.addWeighted(radar_2d_map_pro_image, alpha, image, 1 - alpha, 0)
    # # cv2.imshow('Overlay', overlay)
    # # cv2.waitKey(0)
    #




#--------------------------------------pointinside& larger input image--------------------------------------------------


    # column_names = ['Frame', 'PointNum', 'Velocity']
    # new_df = pd.DataFrame(columns=column_names)
    # ego_velo = np.load('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/radar_velo.npy')
    # for frame in range(0,9931,1):
    #     radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(frame)
    #     if not (os.path.exists(radar_file) and os.path.exists(calib_file)):
    #         print('file not exist!')
    #         continue
    #     if os.path.exists(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/pmap_image/{frame}.jpg'):
    #         r_in = np.load(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/r_in/{frame}.npy')
    #         pointnum = len(r_in)
    #     vx, vy, vz = ego_velo[frame]
    #     v = math.sqrt(vx**2+vy**2+vz**2)
    #     new_data = pd.DataFrame({'Frame':[frame], 'PointNum':[pointnum], 'Velocity':[v]})
    #     new_df = pd.concat([new_df, new_data], ignore_index=True)
    # # for frame in range(7,9931,15):
    # #     radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(frame)
    # #     if not (os.path.exists(radar_file) and os.path.exists(calib_file)):
    # #         print('file not exist!')
    # #         continue
    # #     if os.path.exists(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/pmap_image/{frame}.jpg'):
    # #         r_in = np.load(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/r_in/{frame}.npy')
    # #         pointnum = len(r_in)
    # #     vx, vy, vz = ego_velo[frame]
    # #     v = math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    # #     new_data = pd.DataFrame({'Frame': [frame], 'PointNum': [pointnum], 'Velocity': [v]})
    # #     new_df = pd.concat([new_df, new_data], ignore_index=True)
    # new_df.to_csv('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/prob_dataset.csv', index=False)
    # print(new_df)

    # df = pd.read_csv('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/prob_dataset.csv')
    # print(df.values.shape)

#-------------------------------------------generate lidar target image-------------------------------------------------
    # lidar_base = 'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/lidar_inter2d_pic/'
    # for frame in range(0,9931,15):
    #     new_coords = []
    #     print(f'frame {frame}')
    #     radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(frame)
    #     if not (os.path.exists(radar_file) and os.path.exists(calib_file)):
    #         print('file not exist!')
    #         continue
    #     K = utils.get_intrinsic_matrix(calib_file)
    #     T_radar = utils.get_radar2cam(calib_file)
    #     T_lidar = utils.get_lidar2cam(lidar_calib_file)
    #     #lidar_pcd = np.fromfile(radar_file, dtype=np.float32).reshape(-1, 4)
    #     coords = np.load(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/cropped_image_coor/{frame}.npy')
    #     local_lidar_depth = cv2.imread(lidar_base+f'{frame}.jpg',cv2.IMREAD_GRAYSCALE)
    #     for i in range(len(coords)):
    #         coor = coords[i]
    #         u1,v1 = coor[0]
    #         u2,v2 = coor[1]
    #         local_lidar_depth_tmp = local_lidar_depth.copy()
    #         local_lidar_depth_tmp = local_lidar_depth_tmp[v1 - 319:v2 - 319, u1:u2]
    #         cv2.imwrite(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/target_lidar/{frame}_{i}.jpg',local_lidar_depth_tmp)









#--------------------------------------------------generate LiDAR depth map---------------------------------------------
    # def process_frame_lidar(frame):
    # #for frame in range(0,9931):
    #
    #     print(f'frame {frame}')
    #     radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(frame)
    #     if not (os.path.exists(radar_file) and os.path.exists(calib_file)):
    #         print('file not exist!')
    #         return
    #         #continue
    #     K = utils.get_intrinsic_matrix(calib_file)
    #     T_radar = utils.get_radar2cam(calib_file)
    #     T_lidar = utils.get_lidar2cam(lidar_calib_file)
    #
    #     lidar_pcd = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
    #     lidar_pcd = utils.filter_vlp16(lidar_pcd)
    #     lidar_pcd = lidar_pcd[lidar_pcd[:,0]> 0]
    #     lidar_pcd = lidar_pcd[:,0:3]
    #     coor = [0, 571, 1935, 1215]
    #     l_in, l_in_2d = radar_in_image(coor, lidar_pcd, T_lidar,K)
    #     np.save(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/lidar_2d/{frame}.npy',l_in_2d)
    #     np.save(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/lidar2d_corr_3dpoints/{frame}.npy',l_in)
    #
    #
    #
    # frames = range(0, 9931)
    #
    # Parallel(n_jobs=8)(delayed(process_frame_lidar)(frame) for frame in frames)



    #-------------------------------------------generate (u,v) coor for radar points------------------------------------
    # for frame in range(0,9931):
    #     radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(frame)
    #     # image = cv2.imread(cam_file)
    #     K = utils.get_intrinsic_matrix(calib_file)
    #     if K is None:
    #         print(f'frame {frame} missing intrinsic matrix.')
    #         continue
    #     T_radar = utils.get_radar2cam(calib_file)
    #     pcd_radar = np.fromfile(radar_file, dtype=np.float32).reshape(-1, 7)[:,0:4]
    #     filtered_points_2d =  radar2image(pcd_radar,T_radar,K)
    #     np.save(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/velodyne_pixel/{frame}.npy',filtered_points_2d)
    #     if frame % 10 == 0:
    #         print(f'frame {frame} saved.')

    # -------------------------------------------generate (u,v) coor for lidar points------------------------------------
    # for frame in range(0,9931):
    #     radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(frame)
    #     # image = cv2.imread(cam_file)
    #     K = utils.get_intrinsic_matrix(calib_file)
    #     if K is None:
    #         print(f'frame {frame} missing intrinsic matrix.')
    #         continue
    #     T_lidar = utils.get_radar2cam(calib_file)
    #     T_radar = utils.get_radar2cam(calib_file)
    #     pcd_lidar = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)[:,0:4]
    #     geo_lidar = utils.filter_vlp16(pcd_lidar)[:,3]
    #     geo_lidar = geo_lidar[geo_lidar[:,0]>0]
    #     lidar_points = utils.trans_point_coor(geo_lidar, np.dot(np.linalg.inv(T_radar), T_lidar))
    #     filtered_points_2d =  radar2image(pcd_lidar,T_lidar,K)
    #     np.save(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/lidar_pixel/{str(frame)}.npy',filtered_points_2d)
    #     if frame % 10 == 0:
    #         print(f'frame {frame} saved.')




    #---------------------------------------------------make dataset----------------------------------------------------
    # column_names = ['Frame', 'Index']
    # df = pd.DataFrame(columns=column_names)
    # for frame in range(0,9931,15):
    #     print(f'frame {frame}')
    #     radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(frame)
    #
    #     image = cv2.imread(cam_file)
    #     K = utils.get_intrinsic_matrix(calib_file)
    #     if K is None:
    #         print(f'frame {frame} missing intrinsic matrix.')
    #         continue
    #     T_radar = utils.get_radar2cam(calib_file)
    #     T_lidar = utils.get_lidar2cam(lidar_calib_file)
    #
    #     crop_size = 40
    #
    #     Ru_min, Ru_max, Rv_min, Rv_max = 0,1935,319,1215
    #     crops, coordinates = sliding_window_crop(image, crop_size, Ru_min, Ru_max, Rv_min, Rv_max)
    #
    #
    #     np.save(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/cropped_image_coor/{frame}.npy',coordinates)
    #     for i in range(len(crops)):
    #         print(f'index {i}')
    #         c = crops[i]
    #         cv2.imwrite(
    #             f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/cropped_images/{frame}_{i}.jpg',
    #             c)
    #         coor = coordinates[i]
    #         u1,v1 = coor[0]
    #         u2,v2 = coor[1]
    #         target_image = cv2.imread(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/pmap_image/{frame}.jpg')
    #         crop_target = target_image[v1-319:v2-319, u1:u2]
    #         print('crop_target:',u1,v1-319,u2,v2-319)
    #         cv2.imwrite(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/target_image/{frame}_{i}.jpg', crop_target)
    #         print('coor_1:',u1,v1,'coor_2:',u2,v2)
    #         new_data = pd.DataFrame({'Frame':[frame], 'Index':[i]})
    #         df = pd.concat([df, new_data], ignore_index=True)
    # df.to_csv('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/prob_dataset.csv')





#-----------------------------------------------split dataset-----------------------------------------------------------
    # random_seed = 2024
    # np.random.seed(random_seed)
    #
    # # 读取数据集
    # df = pd.read_csv('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/prob_dataset.csv')
    # sequences = {
    #     "Sequence 1": (0, 543),
    #     "Sequence 2": (544, 1311),
    #     "Sequence 3": (1312, 1802),
    #     "Sequence 4": (1803, 2199),
    #     "Sequence 5": (2200, 2531),
    #     "Sequence 6": (2532, 2797),
    #     "Sequence 7": (2798, 3574),
    #     "Sequence 8": (3575, 4047),
    #     "Sequence 9": (4049, 4386),  # Frame 04048 is missing
    #     "Sequence 10": (4387, 5085),
    #     "Sequence 11": (6334, 6570),  # Skipping Frame 05085-06334 as missing
    #     "Sequence 12": (6571, 6758),
    #     "Sequence 13": (6759, 7542),
    #     "Sequence 14": (7543, 7899),
    #     "Sequence 15": (7900, 8197),
    #     "Sequence 16": (8198, 8480),
    #     "Sequence 17": (8481, 8748),
    #     "Sequence 18": (8749, 9095),
    #     "Sequence 19": (9096, 9517),
    #     "Sequence 20": (9518, 9775),
    #     "Sequence 21": (9776, 9930),
    # }
    #
    # # 创建空的训练集和测试集
    # train_data = pd.DataFrame(columns=df.columns)
    # test_data = pd.DataFrame(columns=df.columns)
    #
    # # 遍历每个序列，按比例划分数据
    # for seq_name, (start_frame, end_frame) in sequences.items():
    #     # 获取当前序列的数据
    #     seq_data = df[(df['Frame'] >= start_frame) & (df['Frame'] <= end_frame)]
    #
    #     # 随机打乱数据
    #     seq_data = seq_data.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    #
    #     # 计算训练集和测试集大小
    #     train_size = int(0.7 * len(seq_data))
    #
    #     # 划分训练集和测试集
    #     train_seq = seq_data[:train_size]
    #     test_seq = seq_data[train_size:]
    #
    #     # 将结果追加到总的训练集和测试集中
    #     train_data = pd.concat([train_data, train_seq], ignore_index=True)
    #     test_data = pd.concat([test_data, test_seq], ignore_index=True)
    #
    # # 对最终的训练集和测试集进行全局打乱
    # train_data = train_data.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    # test_data = test_data.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    #
    # # 保存训练集和测试集到 CSV 文件
    # train_data.to_csv('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/train_dataset.csv', index=False)
    # test_data.to_csv('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/test_dataset.csv', index=False)


# -----------------------------------------------split dataset----------------------------------------------------------










#------------------------------------------edge&corner detection--------------------------------------------------------
# # 根据 Frame 列对数据集进行排序（如果尚未排序）
#     df = pd.read_csv('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/prob_dataset.csv')
#     df = df.sort_values(by='Frame').reset_index(drop=True).values
#     for i in range(len(df)):
#         frame, pointnum,v = df[i]
#         cam_dir = 'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/image_2/'+str(int(frame)).zfill(5)+".jpg"
#         des_dir = 'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/edge_image/'+str(int(frame))+".jpg"
#         image = cv2.imread(cam_dir)
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#         # 边缘检测
#         edges = cv2.Canny(gray, 50, 150)
#
#         # 角点检测 (Harris)
#         dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
#         dst = cv2.dilate(dst, None)
#
#         # 创建一个全黑图像，用于存放边缘和角点信息
#         combined = np.zeros_like(gray)
#
#         # 将边缘信息添加到 combined 图像中
#         combined[edges > 0] = 128  # 用128表示边缘
#
#         # 将角点信息添加到 combined 图像中
#         combined[dst > 0.01 * dst.max()] = 255  # 用255表示角点
#
#         # 显示和保存结果
#         # cv2.imshow('Combined Edges and Corners', combined)
#         # cv2.imwrite('combined_output.jpg', combined)
#         # cv2.waitKey(0)
#         # cv2.destroyAllWindows()
#         cropped_edges = combined[571:1215, 0: 1935]
#         cv2.imwrite(des_dir,cropped_edges)
#         print(des_dir)
#         print(edges.shape)



    # # 读取图像并转换为灰度图
    # image = cv2.imread('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/image_2/00240.jpg')
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #
    # # 边缘检测
    # edges = cv2.Canny(gray, 50, 150)
    #
    # # 角点检测 (Harris)
    # dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    # dst = cv2.dilate(dst, None)
    #
    # # 创建一个全黑图像，用于存放边缘和角点信息
    # combined = np.zeros_like(gray)
    #
    # # 将边缘信息添加到 combined 图像中
    # combined[edges > 0] = 128  # 用128表示边缘
    #
    # # 将角点信息添加到 combined 图像中
    # combined[dst > 0.01 * dst.max()] = 255  # 用255表示角点
    #
    # # 显示和保存结果
    # cv2.imshow('Combined Edges and Corners', combined)
    # cv2.imwrite('combined_output.jpg', combined)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # img = cv2.imread('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/image_2/09902.jpg', 0)
    # fast = cv2.FastFeatureDetector_create()
    # keypoints = fast.detect(img, None)
    # img_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imshow('FAST Corners', img_keypoints)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

#------------------------------------------edge&corner detection--------------------------------------------------------





#--------------------------------------deciding covariance by 2D clustering---------------------------------------------
    # x_max, y_max = 0, 0
    # x_mean, y_mean = 0,0
    # cnt = 0
    # for frame in range(0,9931):
    #     #print(f'frame {frame}')
    #     radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(frame)
    #
    #     #image = cv2.imread(cam_file)
    #     K = utils.get_intrinsic_matrix(calib_file)
    #     if not (os.path.exists(radar_file) and os.path.exists(calib_file)):
    #         #print('file not exist!')
    #         continue
    #     if K is None:
    #         print(f'frame {frame} missing intrinsic matrix.')
    #         continue
    #     T_radar = utils.get_radar2cam(calib_file)
    #     T_lidar = utils.get_lidar2cam(lidar_calib_file)
    #     # # r_in_final, r_in_2d_final = radar_in_image(coor,pcd_radar,T_radar,K)
    #     # # np.save(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/r_in_2d/{frame}.npy',r_in_2d_final)
    #     # # np.save(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/r_in/{frame}.npy', r_in_final)
    #     annotations = utils.read_annotation(txt_base_dir+f'{str(frame).zfill(5)}.txt')
    #     if annotations is None:
    #         continue
    #     r_in_2d = np.load(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/r_in_2d/{frame}.npy')
    #     r_in = np.load(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/r_in/{frame}.npy')
    #     for anno in annotations:
    #         xmin, ymin, xmax, ymax = anno['Bbox']
    #         boundary=[0,1935,319,319+896]
    #         if xmin <= 0 or xmax >= 1935 or ymin <= 319 or ymax >= 319+896:
    #             continue
    #         x_diff = xmax-xmin
    #         y_diff = ymax-ymin
    #         cl = anno['Class']
    #
    #         bbox_points = r_in_2d[(r_in_2d[:, 0] >= xmin) & (r_in_2d[:, 0] <= xmax) &
    #                               (r_in_2d[:, 1] >= ymin) & (r_in_2d[:, 1] <= ymax)]
    #
    #         if bbox_points.shape[0] > 0:
    #             # 计算距离bbox边界的最小距离
    #             distances = calculate_distances_to_bbox_edges(bbox_points, (xmin, ymin, xmax, ymax))
    #             p_xmin, p_xmax, p_ymin, p_ymax = distances["xmin"], distances["xmax"], distances["ymin"], distances["ymax"]
    #             flag = 0
    #             x_mean += max(p_xmax, p_xmin)
    #             y_mean += max(p_xmax, p_xmin)
    #             cnt += 1
    #             if max(p_xmax, p_xmin) > x_max:
    #                 print('update x_max to',max(p_xmax, p_xmin))
    #                 x_max = max(p_xmax, p_xmin)
    #                 flag = 1
    #             if max(p_ymax, p_ymin) > y_max:
    #                 print('update y_max to',max(p_ymax, p_ymin))
    #                 y_max = max(p_ymax, p_ymin)
    #                 flag = 1
    #             if flag == 1:
    #                 print('frame:',frame)
    #                 print(f'class:{cl}, x_diff = {x_diff}, y_diff = {y_diff}')
    #                 print(xmin, ymin, xmax, ymax)
    #                 print(bbox_points)
    #                 print(f'  Distance to bbox edges:')
    #                 print(f'    Distance to xmin edge = {distances["xmin"]:.2f}')
    #                 print(f'    Distance to ymin edge = {distances["ymin"]:.2f}')
    #                 print(f'    Distance to xmax edge = {distances["xmax"]:.2f}')
    #                 print(f'    Distance to ymax edge = {distances["ymax"]:.2f}')
    #     # pcd_radar = np.fromfile(radar_file,dtype=np.float32).reshape(-1,7)
    #     # coor = [0,319,1935,1215]
    # x_mean /= cnt
    # y_mean /= cnt
    # print('x_mean, y_mean:',x_mean, y_mean)






# --------------------------------------------fix lidar depth-----------------------------------------------------------
#     df = pd.read_csv('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/RNet_dataset.csv').values[:,1:]
#     for i in range(len(df)):
#         lidar_uv_depth = np.full((100, 100), -1)
#         frame = df[i, 0]
#         index = df[i, 1]
#         if df[i,3] > 0:
#             lidar_points = np.load(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/cropped_lidar_radarcoor/{frame}_{index}.npy')
#             radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(frame)
#             #image = cv2.imread(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/cropped_images/{frame}_{index}.jpg')
#             K = utils.get_intrinsic_matrix(calib_file)
#             if K is None:
#                 continue
#             T_radar = utils.get_radar2cam(calib_file)
#             T_lidar = utils.get_lidar2cam(lidar_calib_file)
#             coor = np.load(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/cropped_image_coor/{frame}.npy')[index]
#             lidar_uv_depth, cnt_depth = lidar2uv_depth(lidar_points,T_radar,K,coor)
#             print(f'saved {frame}_{index}.npy, cnt_depth = {cnt_depth}')
#         np.save(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/lidar_uv_depth/{frame}_{index}.npy',lidar_uv_depth)

# --------------------------------------------fix lidar depth-----------------------------------------------------------





#--------------------------------------------projection & test----------------------------------------------------------
    # image = cv2.imread('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/image_2/00000.jpg')
    # cropped_image = cv2.imread('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/cropped_images/0_76.jpg')
    # radar_pcd = np.load('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/velodyne_pixel_radar_coor/0_76.npy')
    # print(radar_pcd)
    # radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(3)
    # K = utils.get_intrinsic_matrix(calib_file)
    # T_radar = utils.get_radar2cam(calib_file)
    # T_lidar = utils.get_lidar2cam(lidar_calib_file)
    # pcd_lidar = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
    # geo_lidar = utils.filter_vlp16(pcd_lidar)[:, 0:3]
    # lidar_pcl = np.load('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/cropped_lidar/0_76.npy')
    # lidar_points = utils.trans_point_coor(geo_lidar, np.dot(np.linalg.inv(T_radar), T_lidar))
    # #project radar points to 2d image frame
    # geo_radar = radar_pcd[:, 0:3]
    # points_3d_r2c = utils.trans_point_coor(geo_radar, T_radar)
    # y_max, x_max = image.shape[:2]
    # fx, cx = K[0, 0], K[0, 2]
    # fy, cy = K[1, 1], K[1, 2]
    # points_2d_radar = utils.project_3d_to_2d(points_3d_r2c, fx, fy, cx, cy)
    # x_coords = points_2d_radar[:, 0]
    # y_coords = points_2d_radar[:, 1]
    # mask_x = (x_coords >= 0) & (x_coords < x_max)
    # mask_y = (y_coords >= 0) & (y_coords < y_max)
    # mask = mask_x & mask_y
    # filtered_points_2d = points_2d_radar[mask]
    # #print(filtered_points_2d)
    # new_image = utils.visualize_with_image_color(image, filtered_points_2d,[0,0,255])
    # cv2.imshow('new_image',new_image)
    # cv2.waitKey(0)
    # cv2.imshow('cropped image',cropped_image)
    # cv2.waitKey(0)

    #-------------------------------project lidar points to radar coor-------------------------------------------------
    # src_dir = 'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/cropped_lidar/'
    # des_dir = 'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/cropped_lidar_radarcoor/'
    # df = pd.read_csv('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/RNet_dataset.csv').values[:,1:]
    #
    # for i in range(len(df)):
    #     frame, index = df[i,0], df[i,1]
    #     radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(frame)
    #     K = utils.get_intrinsic_matrix(calib_file)
    #     if K is None:
    #         print(frame)
    #         continue
    #     T_radar = utils.get_radar2cam(calib_file)
    #     T_lidar = utils.get_lidar2cam(lidar_calib_file)
    #     filename = str(frame)+"_"+str(index)+".npy"
    #     geo_lidar = np.load(src_dir+filename)
    #     # if len(geo_lidar) > 0:
    #     lidar_R = utils.trans_point_coor(geo_lidar, np.dot(np.linalg.inv(T_radar), T_lidar))
    #     #print(lidar_R)
    #     np.save(des_dir+filename,lidar_R)



#-----------------------------project lidar_uv_depth on cropped image-------------------------------------------
    # frame = 30
    # index = 4
    # image = cv2.imread(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/cropped_images/{frame}_{index}.jpg')
    # pixel_matrix = np.load(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/lidar_uv_depth/{frame}_{index}.npy')
    #
    # for i in range(pixel_matrix.shape[0]):
    #     for j in range(pixel_matrix.shape[1]):
    #         if pixel_matrix[i, j] != -1:
    #             cv2.circle(image, (j, i), 1, (0, 0, 255), -1)  # 红色点
    #
    # # 使用 OpenCV 显示结果图像
    # cv2.imshow('Visualized Image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
#-----------------------------project lidar_uv_depth on cropped image-------------------------------------------






# --------------------------------------------------generate range image------------------------------------------------

    # #print(file_list)
    # dst = 'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/range_image/'
    # for frame in range(0,9931,15):
    #     print(f'frame {frame}')
    #     radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(frame)
    #     K = utils.get_intrinsic_matrix(calib_file)
    #     if K is None:
    #         print(f'frame {frame} missing intrinsic matrix.')
    #         continue
    #     lidar_points = np.fromfile(lidar_file,dtype=np.float32).reshape(-1,4)
    #     lidar_points = utils.filter_vlp16(lidar_points)
    #     lidar_points = lidar_points[lidar_points[:,0]>0][:,0:3]
    #     T_radar = utils.get_radar2cam(calib_file)
    #     T_lidar = utils.get_lidar2cam(lidar_calib_file)
    #     coords = np.load(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/cropped_image_coor/{frame}_expanded.npy')
    #     for i in range(len(coords)):
    #         print(i)
    #         coor = coords[i]
    #         point1, point2 = coor
    #         x1, y1 = point1
    #         x2, y2 = point2
    #         print('x1,y1,x2,y2',x1,y1,x2,y2)
    #         lidar_in, lidar_in_2d = radar_in_image([x1,y1,x2,y2],lidar_points,T_lidar,K)
    #         # if len(lidar_in) > 0:
    #         #     lidar_in = utils.trans_point_coor(lidar_in, np.dot(np.linalg.inv(T_radar), T_lidar))
    #         yaw1, pitch1 = pixel_to_radar_angles(x1, y1, K, T_lidar)
    #         yaw2, pitch2 = pixel_to_radar_angles(x2, y2, K, T_lidar)
    #         yaw3, pitch3 = pixel_to_radar_angles(x1, y2, K, T_lidar)
    #         yaw4, pitch4 = pixel_to_radar_angles(x2, y1, K, T_lidar)
    #         assert yaw1 > yaw2 and pitch1 > pitch2
    #         yaw_min = min(yaw2,yaw4)
    #         yaw_max = max(yaw1,yaw3)
    #         pitch_min = min(pitch2, pitch3)
    #         pitch_max = max(pitch1,pitch4)
    #
    #         # fov_up, fov_down = pitch1, pitch2
    #         fov_up, fov_down = pitch_max, pitch_min
    #         dst_name = dst + str(frame)+"_"+str(i)+".jpg"
    #         # fov_up, fov_down = 15,-15
    #         proj_H, proj_W = 100,100
    #         print('yaw1,yaw2,pitch1,pitch2',yaw_max,yaw_min,pitch_max,pitch_min)
    #         gen_range_image(lidar_in, fov_up, fov_down, yaw_max, yaw_min, proj_H, proj_W, dst_name)


# --------------------------------------------------test lidar modal  limit---------------------------------------------
#     tp,fp,tn,fn = 0,0,0,0
#     acc = 0
#     df = pd.read_csv('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/RNet_dataset.csv').values[:,1:]
#     for i in range(len(df)):
#         PointInside = df[i,2]
#         len_lidar = df[i,3]
#         if PointInside == 0:
#             if len_lidar > 0:
#                 fp += 1
#             elif len_lidar == 0:
#                 tn += 1
#                 acc += 1
#         else:
#             if len_lidar > 0:
#                 tp += 1
#                 acc += 1
#             elif len_lidar == 0:
#                 fn += 1
#     accuracy = acc / len(df)
#     precision = tp / (tp + fp) if (tp + fp) != 0 else 0
#     recall = tp / (tp + fn) if (tp + fn) != 0 else 0
#     print('tp,fp,tn,fn',tp,fp,tn,fn)
#     print('precision:',precision)
#     print('recall:',recall)
#     print('accuracy:',accuracy)
# --------------------------------------------------test lidar modal  limit---------------------------------------------


# ------------------------------------------------len of radar point cloud-------------------------------------------
#     max_len = 0
#     folder_path = 'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/velodyne_pixel_radar_coor/'
#     file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
#     for f in file_list:
#         pcd_file = folder_path + f
#         pcd = np.load(pcd_file)
#         if len(pcd) > max_len:
#             max_len = len(pcd)
#             print('newest max len:',max_len)
#     print(max_len)
# ------------------------------------------------len of radar point cloud-------------------------------------------


# # ----------------------------------------------make RCS regression dataset---------------------------------------------
#     train_df = pd.read_csv("D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/train_dataset.csv").values
#     test_df = pd.read_csv("D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/test_dataset.csv").values
#     column_names = ['Frame', 'Index', 'x', 'y', 'z','u', 'v', 'v_r', 'rcs']
#     new_df_train = pd.DataFrame(columns=column_names)
#     for i in range(len(test_df)):
#         frame, pointnum, ego_velocity = test_df[i]
#         frame, pointnum = int(frame), int(pointnum)
#         print('frame',frame)
#         radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(frame)
#         K = utils.get_intrinsic_matrix(calib_file)
#         if K is None:
#             print(f'frame {frame} missing intrinsic matrix.')
#             continue
#         T_radar = utils.get_radar2cam(calib_file)
#         T_lidar = utils.get_lidar2cam(lidar_calib_file)
#         r_in = np.load(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/r_in/{frame}.npy')
#         r_in_2d = np.load(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/r_in_2d/{frame}.npy')
#         l_in = np.load(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/lidar2d_corr_3dpoints/{frame}.npy')
#         l_in_radar_coor = utils.trans_point_coor(l_in, np.dot(np.linalg.inv(T_radar), T_lidar))
#         np.save(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/l_in_radar_coor/{frame}.npy',l_in_radar_coor)
#         kdtree = KDTree(l_in_radar_coor)
#         n = 50
#         sampled_points, sampled_points_2d = sample_points_from_point_cloud(r_in, r_in_2d, n)
#         for j in range(0,min(n,len(r_in))):
#             x,y,z,rcs,v_r = sampled_points[j,0:5]
#             u,v = sampled_points_2d[j,:]
#             new_data = pd.DataFrame(
#                 {'Frame': [frame], 'Index': [j], 'x':[x], 'y':[y], 'z':[z], 'u':[u], 'v':[v], 'v_r':[v_r], 'rcs':[rcs]})
#             new_df_train = pd.concat([new_df_train, new_data], ignore_index=True)
#             r = 1
#
#             pitch = np.degrees(np.arctan(z/ np.linalg.norm([x, y], 2)))
#
#             local_lidar_index = kdtree.query_ball_point([x, y, z], r)
#             # if len(local_lidar_index) == 0:
#             #     print(x, y, z)
#             #     distance, index = kdtree.query([x, y, z])
#             #     print(distance,l_in_radar_coor[index])
#             local_lidar_radarcoor = l_in_radar_coor[local_lidar_index]
#
#             np.save(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/local_points/{frame}_{j}.npy',
#                     local_lidar_radarcoor)
#             dst = f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/range_image/{frame}_{j}.jpg'
#             virtual_point = get_new_point(x, y, z)
#             fov_down = 7
#             fov_up = -23
#             for k in range(len(local_lidar_radarcoor)):
#                 local_lidar_radarcoor[k, 0] -= virtual_point[0]
#                 local_lidar_radarcoor[k, 1] -= virtual_point[1]
#                 local_lidar_radarcoor[k, 2] -= virtual_point[2]
#                 x1, y1, z1 = local_lidar_radarcoor[k]
#                 pitch = np.degrees(np.arctan(z1 / np.linalg.norm([x1,y1],2)))
#                 if pitch > fov_up:
#                     fov_up = pitch
#                 if pitch < fov_down:
#                     fov_down = pitch
#
#             proj_H, proj_W, = 32, 128
#             gen_range_image_rcs(local_lidar_radarcoor, fov_up, fov_down, proj_H, proj_W,
#                             dst)
#     new_df_train.to_csv('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/RCS_dataset_test.csv')
# ----------------------------------------------make RCS regression dataset---------------------------------------------






#-------------------------------------------make RCS regression dataset without vp--------------------------------------

    # df = pd.read_csv('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/RCS_dataset_train.csv').values[:,1:]
    # for i in range(len(df)):
    #
    #     frame, index, x, y, z, u, v, v_r, rcs = df[i]
    #     l_in_radar_coor = np.load(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/l_in_radar_coor/{int(frame)}.npy')
    #     kdtree = KDTree(l_in_radar_coor)
    #
    #     r = 1
    #
    #     pitch = np.degrees(np.arctan(z / np.linalg.norm([x, y], 2)))
    #
    #     local_lidar_index = kdtree.query_ball_point([x, y, z], r)
    #
    #     local_lidar_radarcoor = l_in_radar_coor[local_lidar_index]
    #
    #     dst = f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/ablation/range_image/{int(frame)}_{int(index)}.jpg'
    #     # virtual_point = get_new_point(x, y, z)
    #     fov_down = -23
    #     fov_up = 7
    #
    #     proj_H, proj_W, = 32, 128
    #     gen_range_image_rcs(local_lidar_radarcoor, fov_up, fov_down, proj_H, proj_W,
    #                         dst)



#-------------------------------make RCS regression dataset for optimal parameter fiding------------------------------
    for r in [0.5,1,2,3,4]:
        def process_data(row):
            frame, index, x, y, z, u, v, v_r, rcs = row
            frame, index = int(frame), int(index)
            print(f'frame {frame}')
        
            l_in_radar_coor = np.load(f'/workspace/data/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/l_in_radar_coor/{int(frame)}.npy')
            kdtree = KDTree(l_in_radar_coor)
            # r = 1

            local_lidar_index = kdtree.query_ball_point([x, y, z], r)
            local_lidar_radarcoor = l_in_radar_coor[local_lidar_index]
            if not os.path.exists(f'/workspace/data/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/ablation/range_image_{r}/'):
                os.mkdir(f'/workspace/data/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/ablation/range_image_{r}/')
            
            dst = f'/workspace/data/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/ablation/range_image_{r}/{int(frame)}_{int(index)}.jpg'
            if os.path.exists(dst):
                return
            original_points = local_lidar_radarcoor.copy()
            
            for k in range(len(local_lidar_radarcoor)):
                local_lidar_radarcoor[k, 0] -= x
                local_lidar_radarcoor[k, 1] -= y
                local_lidar_radarcoor[k, 2] -= z
            proj_H, proj_W = 32, 128
            gen_range_image_rcs_translation_correct(local_lidar_radarcoor,original_points, proj_H, proj_W,  [x,y,z],r,
                                            dst)


        df = pd.read_csv('/workspace/data/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/RCS_dataset_train.csv').values[:, 1:]
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            executor.map(process_data, df)
        df = pd.read_csv('/workspace/data/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/RCS_dataset_test.csv').values[:,
            1:]
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            executor.map(process_data, df)




# ----------------------------------------------ego velocity and coor--------------------------------------------------

    # velo_last = None
    # velo = []
    # for frame in range(0, 9931):
    #     print(f'frame {frame}')
    #     frame0, frame1 = frame, frame + 1
    #     if frame + 1 > 9930:
    #         velo.append(velo_last)
    #     else:
    #         radar_file1, lidar_file1, cam_file1, calib_file1, lidar_calib_file1, txt_base_dir1 = utils.get_vod_dir(
    #             frame0)
    #         radar_file2, lidar_file2, cam_file2, calib_file2, lidar_calib_file2, txt_base_dir2 = utils.get_vod_dir(
    #             frame1)
    #         T_radar1 = utils.get_radar2cam(calib_file1)
    #         T_radar2 = utils.get_radar2cam(calib_file2)
    #         odom_transform, _, _ = utils.compute_transform(frame0, frame1, T_radar1, T_radar2)
    #         if odom_transform is None and velo_last is not None:
    #             velo.append(velo_last)
    #         else:
    #             v_x, v_y, v_z = odom_transform[0, 3] / 0.1, odom_transform[1, 3] / 0.1, odom_transform[2, 3] / 0.1
    #             print(v_x, v_y, v_z)
    #             velo_last = np.array([v_x, v_y, v_z])
    #             v_magnitude, v_pitch, v_yaw = utils.cartesian_to_spherical(velo_last)
    #             velo.append([v_magnitude, v_pitch, v_yaw])
    #             print(v_magnitude, v_pitch, v_yaw)
    #
    # assert len(velo) == 9931
    # np.save('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/ego_velocity_spherical.npy',
    #         np.array(velo))

    # ego_velocity = np.load('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/ego_velocity.npy')
    # #v_magnitude, pitch, yaw
    # df = pd.read_csv('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/RNet_dataset.csv')
    # # 添加新的列
    # df['v_magnitude'] = np.nan
    # df['v_pitch'] = np.nan
    # df['v_yaw'] = np.nan
    #
    # # 更新数据框
    # for index, row in df.iterrows():
    #     frame = int(row['Frame'])
    #     if frame < len(ego_velocity):
    #         v_magnitude, v_pitch, v_yaw = ego_velocity[frame]
    #         df.at[index, 'v_magnitude'] = v_magnitude
    #         df.at[index, 'v_pitch'] = v_pitch
    #         df.at[index, 'v_yaw'] = v_yaw
    #
    # # 保存新的CSV文件
    # df.to_csv('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/RNet_dataset.csv',
    #           index=False)
    #
    # print(df)
    # df['pitch1'] = np.nan
    # df['pitch2'] = np.nan
    # df['yaw1'] = np.nan
    # df['yaw2'] = np.nan
    #
    # for index, row in df.iterrows():
    #     frame = int(row['Frame'])
    #     coordinates = np.load(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/cropped_image_coor/{str(frame)}.npy')
    #     coor = coordinates[int(row['Index'])]
    #     u1,v1 = coor[0]
    #     u2,v2 = coor[1]
    #     radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(frame)
    #     T_radar = utils.get_radar2cam(calib_file)
    #     K = utils.get_intrinsic_matrix(calib_file)
    #     if K is None:
    #         print(f'frame {frame} missing intrinsic matrix.')
    #         continue
    #     yaw1, pitch1 = pixel_to_radar_angles(u1, v1, K, T_radar)
    #     yaw2, pitch2 = pixel_to_radar_angles(u2, v2, K, T_radar)
    #     if frame < len(ego_velocity):
    #         v_magnitude, v_pitch, v_yaw = ego_velocity[frame]
    #         df.at[index, 'pitch1'] = pitch1
    #         df.at[index, 'pitch2'] = pitch2
    #         df.at[index, 'yaw1'] = yaw1
    #         df.at[index, 'yaw2'] = yaw2
    #
    # # 保存新的CSV文件
    # df.to_csv('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/RNet_dataset.csv',
    #           index=False)








