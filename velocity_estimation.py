import utils
import numpy as np
import pandas as pd
import random
import math
import os
import csv
from collections import defaultdict
import open3d as o3d
from scipy.spatial import KDTree
from sample_vod_stats import trackid2pcd, track_id_f2bbox
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle
import gc
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import multivariate_normal
from sklearn.linear_model import Ridge, RANSACRegressor
from sklearn.linear_model import LinearRegression


def remove_existed_pcd(pcd_A, pcd_B):
    unique_mask = np.isin(pcd_A, pcd_B).all(axis=1)
    unique_points = pcd_A[~unique_mask]

    return unique_points
def get_object_velo(track_id, frame):
    radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(frame)
    label_path = 'D:/VoD_dataset/label_2_with_track_ids/label_2/'
    file_path = os.path.join(label_path, f'{frame:05d}.txt')
    loc0 = None
    annotations = utils.read_annotation(file_path)
    T_radar = utils.get_radar2cam(calib_file)
    for anno in annotations:
        if anno['Track_ID'] == track_id:
            loc0 = anno['Location']
            loc0 = utils.transform_coor_to_radar(loc0, T_radar)
    new_frame = frame + 1
    file_path1 = os.path.join(label_path,f'{new_frame:05d}.txt')

    radar_file1, lidar_file1, cam_file1, calib_file1, lidar_calib_file1, txt_base_dir1 = utils.get_vod_dir(frame + 1)
    annotations1 = utils.read_annotation(file_path1)
    loc1 = None
    T_radar1 = utils.get_radar2cam(calib_file1)
    for anno in annotations1:
        if anno['Track_ID'] == track_id:
            loc1 = anno['Location']
            loc1 = utils.transform_coor_to_radar(loc1, T_radar1)
    if loc0 is None or loc1 is None:
        return None

    odom_transform_radar, _,_ = utils.compute_transform(frame, frame + 1, T_radar, T_radar1)
    loc0_1coor = utils.transform_coor_to_radar(loc0, odom_transform_radar)
    #print(loc0, loc1)
    velo = loc1 - loc0_1coor
    velo = np.array(velo)
    return velo


def estimate_radar_velocity_ransac(pcd_static, min_samples=10, residual_threshold= 1e-2, max_trials=3000, alpha = 1.0):
    """
    使用RANSAC从静态雷达点估计雷达自身的速度。

    参数:
    - pcd_static: 静态雷达点云，形状为 (N, 7) 的 NumPy 数组，每一行表示一个点 [x, y, z, RCS, v_r, v_r_compensated, time]。
    - min_samples: 每次RANSAC拟合使用的最小样本数。
    - residual_threshold: RANSAC的残差阈值，用于判断一个点是否是内点。
    - max_trials: RANSAC的最大迭代次数。

    返回:
    - estimated_radar_velocity: 估计的雷达自身速度 [vx, vy, vz]。
    """

    # 提取位置坐标和多普勒速度
    positions = pcd_static[:, :3]  # 获取x, y, z
    vr = pcd_static[:, 4]  # 获取v_r

    # 归一化位置坐标
    normalized_positions = -positions / np.linalg.norm(positions, axis=1)[:, np.newaxis]

    ridge = Ridge(alpha=alpha)
    ransac = RANSACRegressor(min_samples=min_samples, residual_threshold=residual_threshold, max_trials=max_trials)

    # RANSAC拟合
    ransac.fit(normalized_positions, vr)

    # 获取估计的雷达速度
    estimated_radar_velocity = ransac.estimator_.coef_

    return estimated_radar_velocity




def estimate_target_velocity_ransac(radar_data, radar_velocity, min_samples=3, residual_threshold=0.1, max_trials=3000):
    """
    使用RANSAC从动态目标的雷达点云数据估计目标的速度。

    参数:
    - radar_data: 动态目标的雷达点云，形状为 (N, 5) 的 NumPy 数组，每一行表示一个点 [x, y, z, rcs, v_r]。
    - radar_velocity: 已知的雷达速度 [vx, vy, vz]。
    - min_samples: 每次RANSAC拟合使用的最小样本数。
    - residual_threshold: RANSAC的残差阈值，用于判断一个点是否是内点。
    - max_trials: RANSAC的最大迭代次数。

    返回:
    - estimated_target_velocity: 估计的目标速度 [vx, vy, vz]。
    """

    # 提取位置坐标和多普勒速度
    positions = radar_data[:, :3]  # 获取x, y, z
    vr = radar_data[:, 4]  # 获取v_r

    # 归一化位置坐标
    normalized_positions = positions / np.linalg.norm(positions, axis=1)[:, np.newaxis]

    # 修正多普勒速度：v_r_corrected = v_r - dot(normalized_positions, radar_velocity)
    vr_corrected = vr + np.dot(normalized_positions, radar_velocity)
    print('vr',vr)
    print('radar part(projected)',np.dot(normalized_positions, radar_velocity))
    print('object part(projected)',vr_corrected)
    # 使用RANSAC拟合目标速度

    ransac = RANSACRegressor(min_samples=min_samples, residual_threshold=residual_threshold, max_trials=max_trials)

    # RANSAC拟合
    ridge = Ridge(alpha=1.0)
    ransac.fit(normalized_positions, vr_corrected)

    # 获取估计的目标速度
    estimated_target_velocity = ransac.estimator_.coef_

    inlier_mask = ransac.inlier_mask_
    outlier_mask = ~inlier_mask

    # 打印内点和外点
    print("内点：")
    print("位置：", positions[inlier_mask])
    print("多普勒速度：", vr[inlier_mask])

    print("\n外点：")
    print("位置：", positions[outlier_mask])
    print("多普勒速度：", vr[outlier_mask])

    return estimated_target_velocity

def estimate_target_velocity_least_squares_3d(radar_data, radar_velocity):
    positions = radar_data[:, :3]
    vr = radar_data[:, 4]

    normalized_positions = positions / np.linalg.norm(positions, axis=1)[:, np.newaxis]

    vr_corrected = vr + np.dot(normalized_positions, radar_velocity)

    ridge = Ridge(alpha=1.0)

    ridge.fit(normalized_positions, vr_corrected)

    estimated_target_velocity = ridge.coef_

    return estimated_target_velocity

def estimate_target_velocity_least_squares(radar_data, radar_velocity, loc):

    positions = radar_data[:, :3]  # 获取x, y, z
    vr = radar_data[:, 4]  # 获取v_r
    direction_to_radar = loc / np.linalg.norm(loc)
    # 归一化位置坐标
    normalized_positions = positions / np.linalg.norm(positions, axis=1)[:, np.newaxis]

    # 修正多普勒速度：v_r_corrected = v_r + dot(normalized_positions, radar_velocity)
    vr_corrected = vr + np.dot(normalized_positions, radar_velocity)
    if len(radar_data) >= 3:
        ridge = Ridge(alpha=1.0)
        ridge.fit(normalized_positions, vr_corrected)
        estimated_velocity = ridge.coef_
        return np.dot(estimated_velocity, direction_to_radar)

    elif len(radar_data) == 2:
        return np.mean(vr_corrected)
    else:
        return vr_corrected[0]




# def trackid2pcd(track_id,frame):
#     label_path = 'D:/VoD_dataset/label_2_with_track_ids/label_2'
#     file_path = os.path.join(label_path, f'{frame:05d}.txt')
#     radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(frame)
#     T_radar = utils.get_radar2cam(calib_file)
#     annotations = utils.read_annotation(file_path)
#     pcd = None
#     for j in range(len(annotations)):
#         track_tmp = annotations[j]['Track_ID']
#         if track_tmp == track_id:   #found
#             bbox, loc = utils.read_3dbbox(annotations[j])
#             bbox = utils.transform_bbox_to_radar(bbox, T_radar)
#             loc = utils.transform_coor_to_radar(loc, T_radar)
#             dimension = annotations[j]['Dimensions']
#             yaw = annotations[j]['Rotation']
#             pcd = np.fromfile(radar_file, dtype=np.float32).reshape(-1, 7)
#             pcd = utils.filter_points_in_bbox(pcd, loc, dimension, yaw)
#             pcd = pcd[pcd[:, 0] > 0]
#             break
#     if pcd is None:
#         print(f'track id {track_id} not found in frame {frame}')
#     return pcd
#
# def track_id_f2bbox(track_id,frame):
#     label_path = 'D:/VoD_dataset/label_2_with_track_ids/label_2'
#     file_path = os.path.join(label_path, f'{frame:05d}.txt')
#     radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(frame)
#     T_radar = utils.get_radar2cam(calib_file)
#     annotations = utils.read_annotation(file_path)
#     bbox, loc, yaw = None, None, None
#     for j in range(len(annotations)):
#         track_tmp = annotations[j]['Track_ID']
#         if track_tmp == track_id:  # found
#             bbox, loc = utils.read_3dbbox(annotations[j])
#             bbox = utils.transform_bbox_to_radar(bbox, T_radar)
#             loc = utils.transform_coor_to_radar(loc, T_radar)
#             yaw = annotations[j]['Rotation']
#             dimension = annotations[j]['Dimensions']
#             break
#     return bbox, loc, yaw












ego_velocity = np.load('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/radar_velo.npy')
class_list = ['ride_uncertain', 'rider', 'moped_scooter', 'bicycle', 'Cyclist', 'vehicle_other', 'Pedestrian', 'truck',
              'DontCare', 'motor', 'bicycle_rack', 'Car', 'human_depiction', 'ride_other']
#ego_velocity_total = []
column_names = ['Frame', 'Track_ID', 'Class', 'Rotation', 'Location_x', 'Location_y', 'Location_z', 'PointNum',
                'Dimension_x', 'Dimension_y', 'Dimension_z', 'v_r_real']
df = pd.DataFrame(columns=column_names)
#         new_data = pd.DataFrame({'Frame':[frame], 'Index':[i], 'PointInside':[pin], 'LenLidar': [len_lidar]})
#         df = pd.concat([df, new_data], ignore_index=True)

for frame in range(0,9931):
    ego_velo = ego_velocity[frame]
    print(f'frame: {frame}')
    #print('radar velocity:', ego_velo)
    radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(frame)
    T_radar = utils.get_radar2cam(calib_file)

    # if T_radar is None:
    #     print(f'frame {frame} is missing in D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/calib/')
    #     continue
    label_path = 'D:/VoD_dataset/label_2_with_track_ids/label_2/'
    file_path = os.path.join(label_path, f'{frame:05d}.txt')

    if not (os.path.exists(radar_file) and os.path.exists(calib_file) and os.path.exists(file_path)):
        print('file not exist!')
        continue

    pcd_radar = np.fromfile(radar_file,dtype=np.float32).reshape(-1, 7)

    annotations = utils.read_annotation(file_path)
    for anno in annotations:
        track_id = anno['Track_ID']
        location = anno['Location']
        rotation = anno['Rotation']
        class_ = anno['Class']
        #dimension = anno['Dimensions']
        pcd_object = trackid2pcd(track_id, frame)
        bbox, location = utils.read_3dbbox(anno)
        bbox = utils.transform_bbox_to_radar(bbox, T_radar)
        location = utils.transform_coor_to_radar(location, T_radar)
        xmin, xmax, ymin, ymax, zmin, zmax = bbox
        l,w,h = xmax-xmin, ymax-ymin, zmax-zmin
        dimension = [l,w,h]

        # print(track_id, class_, len(pcd_object))
        # print(pcd_object[:3, :])
        # print(location, dimension)
        transformed_points = pcd_object[:, 0:3] - location
        save_path = f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/gaussian_points/{frame}_{track_id}.npy'
        np.save(save_path, transformed_points)

        velo_target = None
        if len(pcd_object) == 0:
            velo_target = 0.
        else:
            velo_target = estimate_target_velocity_least_squares(pcd_object, ego_velo, location)
            #print('Least Squares->velo_target:',velo_target)
        # column_names = ['Frame', 'Track_ID', 'Class', 'Rotation', 'Location_x', 'Location_y', 'Location_z', 'PointNum',
        #                 'Dimension_x', 'Dimension_y', 'Dimension_z', 'v_r_real']
        new_data = pd.DataFrame({'Frame':[frame], 'Track_ID':[track_id], 'Class':[class_],'Rotation':[rotation],
                                 'Location_x':[location[0]], 'Location_y':[location[1]], 'Location_z':[location[2]],
                                 'PointNum': [len(pcd_object)], 'Dimension_x':[l], 'Dimension_y':[w], 'Dimension_z':[h],
                                 'v_r_real':[velo_target]})
        df = pd.concat([df, new_data], ignore_index=True)
df.to_csv('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/dynamic_objects_total.csv')
filtered_df = df[df['PointNum'] > 3]
class_counts = filtered_df['Class'].value_counts()
print(class_counts)


    # ego_velocity = estimate_radar_velocity_ransac(pcd_static)
    # print(f'radar ego velocity of frame {frame}:', ego_velocity)

#np.save('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/radar_velo.npy',np.array(ego_velocity_total))

