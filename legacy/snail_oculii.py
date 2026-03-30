import cv2
import numpy as np
import pandas as pd
import os
import math
import json
import utils
from joblib import Parallel, delayed
import gt_generation_p as utils1
from sklearn.linear_model import Ridge, RANSACRegressor, LinearRegression
from scipy.spatial.transform import Rotation as R
import re
import open3d as o3d
import concurrent.futures

from pypcd4 import PointCloud
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree


def radar_in_lidarfov(pcd_radar, T_radar, T_lidar):
    mask = []
    for i in range(len(pcd_radar)):
        geo = pcd_radar[i][0:3]
        geo_lidarcoor = utils1.transform_coor_to_radar(geo, np.dot(np.linalg.inv(T_lidar), T_radar))
        # yaw = np.arctan2(geo_lidarcoor[1], geo_lidarcoor[0])
        pitch = np.arctan2(geo_lidarcoor[2], np.linalg.norm(geo_lidarcoor[:2]))
        # yaw_deg = np.degrees(yaw)
        pitch_deg = np.degrees(pitch)
        r = np.linalg.norm(geo_lidarcoor,axis=0)
        if -25 < pitch_deg < 15 and 0 <= r <= 120:
            mask.append(i)
    # radar = pcd_radar[mask,:]
    return mask


def get_new_point(x, y, z, r):
    point = np.array([x, y, z])
    vector = point - np.array([0, 0, 0])
    unit_vector = vector / np.linalg.norm(vector)
    new_point = point - unit_vector * r
    return new_point

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


def gen_range_image_rcs(local_points, fov_up, fov_down, proj_H, proj_W, r_l, dst):
    print("len of local_points:",len(local_points))
    fov_up = fov_up / 180.0 * np.pi
    fov_down = fov_down / 180.0 * np.pi
    fov = abs(fov_down) + abs(fov_up)

    #depth = np.linalg.norm(local_points[:, :3], 2, axis=1)
    depth = local_points[:,2]

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
    range_norm = np.minimum(1 + (depth - r_l) / (2 * r_l) * 255, 254)
    # proj_range[proj_y, proj_x] = depth*100  #multiply depth by 100, or the color will not be able tell
    proj_range[proj_y, proj_x] = range_norm

    cv2.imwrite(dst, proj_range)

    return



def gen_range_image_rcs_translation(local_points, proj_H, proj_W, r_l, dst):
    print("len of local_points:", len(local_points))

    # 计算深度
    depth = np.linalg.norm(local_points[:, :3], 2, axis=1)
    #depth = local_points[:, 0]

    scan_x = local_points[:, 1]
    scan_y = local_points[:, 2]

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

    # 归一化到 [0, 255] 范围内
    #range_norm = np.minimum((1 + ((depth - r_l) / (2 * r_l))) * 255+150, 254)
    range_norm = np.minimum((depth / r_l) * 255, 254)
    proj_range[proj_y, proj_x] = range_norm

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


def calculate_height_and_angles(lidar_pcd):
    # 提取高度信息
    heights = lidar_pcd[:, 2]

    max_height = np.max(heights)
    min_height = np.min(heights)

    x = lidar_pcd[:, 0]
    y = lidar_pcd[:, 1]
    z = lidar_pcd[:, 2]

    yaw = np.arctan2(y, x)

    pitch = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))

    yaw_range = (np.min(yaw), np.max(yaw))
    pitch_range = (np.min(pitch), np.max(pitch))

    return max_height, min_height, yaw_range, pitch_range


def get_K():
    cameras = [
        {
            'focal_length': [266.52, 266.8025],
            'principal_point': [345.01, 189.99125]
        },
        {
            'focal_length': [266.35, 266.5125],
            'principal_point': [338.65, 194.67525]
        }
    ]

    # 获取第一个相机的内参矩阵
    camera = cameras[0]
    focal_length = camera['focal_length']
    principal_point = camera['principal_point']

    K = np.array([[focal_length[0], 0, principal_point[0]],
                  [0, focal_length[1], principal_point[1]],
                  [0, 0, 1]])

    return K


def get_frame_data():
    base = 'D:/snail_radar/20231208/data4/'
    cam_left_base_dir = base + 'zed2i/left/'
    cam_right_base_dir = base + 'zed2i/right/'
    radar_arg_base_dir = base + 'ars548/points/'
    radar_eagle_base_dir = base + 'eagleg7/pcl/'
    lidar_base_dir = base + 'xt32/'
    #imu_base_dir = base + 'mti3dk/'
    odom_base_dir = base + 'zed2i/'

    # 获取各个传感器的文件列表并提取时间戳
    cam_left_files = {os.path.splitext(f)[0] for f in os.listdir(cam_left_base_dir) if f.endswith('.jpg')}
    cam_right_files = {os.path.splitext(f)[0] for f in os.listdir(cam_right_base_dir) if f.endswith('.jpg')}
    radar_arg_files = {os.path.splitext(f)[0] for f in os.listdir(radar_arg_base_dir) if f.endswith('.pcd')}
    radar_eagle_files = {os.path.splitext(f)[0] for f in os.listdir(radar_eagle_base_dir) if f.endswith('.pcd')}
    lidar_files = {os.path.splitext(f)[0] for f in os.listdir(lidar_base_dir) if f.endswith('.pcd')}

    odom_data = pd.read_csv(odom_base_dir + 'odom.txt', delim_whitespace=True, header=None, dtype=str)
    odom_data.columns = ['odom_time', 'x', 'y', 'z', 'q1', 'q2', 'q3', 'q4']
    odom_time_list = odom_data.values[:, 0]
    # imu_data = pd.read_csv(imu_base_dir + 'imu.txt', delim_whitespace=True, header=None)
    # imu_data.columns = ['imu_time', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z']

    # 找到每个传感器的最小和最大时间戳
    all_times = (cam_left_files | cam_right_files | radar_arg_files | radar_eagle_files | lidar_files)
    min_time = min(all_times, key=lambda x: float(x))
    max_time = max(all_times, key=lambda x: float(x))

    # 初始化 DataFrame
    # df = pd.DataFrame(
    #     columns=['frame', 'timestamp', 'cam_left_time', 'cam_right_time', 'radar_arg_time', 'radar_eagle_time',
    #              'lidar_time'])
    df = pd.DataFrame(
        columns=['frame', 'timestamp', 'cam_left_time', 'cam_right_time', 'radar_arg_time', 'radar_eagle_time',
                 'lidar_time', 'odom_time'])

    # 设置 frame 间隔为 0.1 秒
    frame_interval = 0.1
    current_frame = 0
    start_time = float(min_time)
    end_time = float(max_time)

    # 遍历从最小时间戳到最大时间戳，每隔 0.1 秒计算一次
    current_time = start_time
    while current_time <= end_time:
        # 找到最接近当前时间戳的传感器数据
        cam_left_time = find_closest_time(cam_left_files, current_time)
        cam_right_time = find_closest_time(cam_right_files, current_time)
        radar_arg_time = find_closest_time(radar_arg_files, current_time)
        radar_eagle_time = find_closest_time(radar_eagle_files, current_time)
        lidar_time = find_closest_time(lidar_files, current_time)

        odom_time = find_closest_time(odom_time_list,current_time)
        # odom_time_row = odom_data.iloc[(odom_data['odom_time'] - current_time).abs().argsort()[:1]]
        # odom_time = odom_time_row['odom_time'].values[0]

        # 计算每个传感器找到的时间与当前时间的误差
        cam_left_error = abs(float(cam_left_time) - current_time)
        cam_right_error = abs(float(cam_right_time) - current_time)
        radar_arg_error = abs(float(radar_arg_time) - current_time)
        radar_eagle_error = abs(float(radar_eagle_time) - current_time)
        lidar_error = abs(float(lidar_time) - current_time)
        odom_error = abs(float(odom_time) - current_time)

        # 检查是否所有传感器的误差都小于 0.05 秒
        if (cam_left_error <= 0.05 and
                cam_right_error <= 0.05 and
                radar_arg_error <= 0.05 and
                radar_eagle_error <= 0.05 and
                lidar_error <= 0.05 and
                odom_error <= 0.05):
            # 创建一行数据
            row = pd.DataFrame({
                'frame': [current_frame],
                'timestamp': [current_time],
                'cam_left_time': [cam_left_time],
                'cam_right_time': [cam_right_time],
                'radar_arg_time': [radar_arg_time],
                'radar_eagle_time': [radar_eagle_time],
                'lidar_time': [lidar_time],
                'odom_time': [odom_time]
            })
            # 使用 pd.concat 来添加新行到 DataFrame
            df = pd.concat([df, row], ignore_index=True)

            # 更新当前 frame
            current_frame += 1

        # 更新当前时间
        current_time += frame_interval
    return df


def find_closest_time(sensor_files, target_time):
    """
    找到最接近目标时间的传感器时间戳。
    """
    sensor_times = [(f, float(f)) for f in sensor_files]

    # 找到最接近的时间戳
    closest_time = min(sensor_times, key=lambda x: abs(x[1] - target_time))

    # 返回原始字符串格式的时间戳
    return closest_time[0]



def get_snail_dir(corres_file, frame,seq='data4'):
    base = f'D:/snail_radar/20231208/{seq}/'
    cam_left_base_dir = base + 'zed2i/left/'
    cam_right_base_dir = base + 'zed2i/right/'
    radar_arg_base_dir = base + 'ars548/points/'
    radar_eagle_base_dir = base + 'eagleg7/pcl/'
    imu_dir = base + 'mti3dk/'
    lidar_base_dir = base+'xt32/'
    radar_arg_file = radar_arg_base_dir + str(corres_file.iloc[frame]['radar_arg_time']) + '.pcd'
    radar_eagle_file = radar_eagle_base_dir + str(corres_file.iloc[frame]['radar_eagle_time']) + '.pcd'
    lidar_file = lidar_base_dir + str(corres_file.iloc[frame]['lidar_time']) + '.pcd'
    cam_left_file = cam_left_base_dir + str(corres_file.iloc[frame]['cam_left_time']) + '.jpg'
    cam_right_file = cam_right_base_dir + str(corres_file.iloc[frame]['cam_right_time']) + '.jpg'
    imu_file = imu_dir + 'imu.txt'

    return radar_arg_file, radar_eagle_file, lidar_file, imu_file, cam_left_file, cam_right_file


def quat2rotm(q):
    """将四元数转换为旋转矩阵"""
    qw, qx, qy, qz = q
    R = np.array([[1 - 2 * qy ** 2 - 2 * qz ** 2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
                  [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx ** 2 - 2 * qz ** 2, 2 * qy * qz - 2 * qx * qw],
                  [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx ** 2 - 2 * qy ** 2]])
    return R




def Body_T_Xt32():
    """Hesai Pandar XT32坐标系到车身坐标系的变换"""
    R_body_xt32 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    p_body_xt32 = np.array([0, 0, 0])
    T_body_xt32 = np.eye(4)
    T_body_xt32[:3, :3] = R_body_xt32
    T_body_xt32[:3, 3] = p_body_xt32
    return T_body_xt32


def Body_T_Zed2Imu():
    """ZED2 IMU到车身坐标系的变换"""
    R_body_zed2 = np.eye(3)
    p_body_zed2 = np.array([0.125, 0.0, -0.195])
    T_body_zed2 = np.eye(4)
    T_body_zed2[:3, :3] = R_body_zed2
    T_body_zed2[:3, 3] = p_body_zed2

    T_Zed2Imu_Xt32 = np.array([[-0.020255, -0.999744, 0.010133, -0.100633],
                               [0.999784, -0.020206, 0.004917, -0.030823],
                               [-0.004711, 0.010231, 0.999937, 0.197481],
                               [0.000000, 0.000000, 0.000000, 1.000000]])
    return Body_T_Xt32() @ np.linalg.inv(T_Zed2Imu_Xt32)


def Body_T_Oculii():
    """Oculii传感器到车身坐标系的变换"""
    R_x36d_oculii = np.array([[0.999829834123514, 0.0183936527785399, -0.001405821470991],
                              [-0.0183512882701082, 0.999498727400647, 0.0257977546176971],
                              [0.00187963171211191, -0.0257675660851007, 0.999666194048132]])
    T_body_x36d = np.eye(4)
    T_body_x36d[:3, :3] = np.eye(3)
    R_body_oculii = T_body_x36d[:3, :3] @ R_x36d_oculii
    p_body_oculii = np.array([0.07, 0, -0.115])

    T_body_oculii = np.eye(4)
    T_body_oculii[:3, :3] = R_body_oculii
    T_body_oculii[:3, 3] = p_body_oculii
    return T_body_oculii


def Body_T_Ars548():
    """ARS548传感器到车身坐标系的变换"""
    R_x36d_ars548 = np.array([[0.998861908053375, 0.0445688670331839, -0.0169854270287433],
                              [-0.0445188967507348, 0.999003062288094, 0.00330898339477966],
                              [0.0171159712569224, -0.0025490449952668, 0.999850261737998]])
    p_body_ars548 = np.array([0, 0, 0.07])
    T_body_x36d = np.eye(4)
    T_body_x36d[:3, :3] = np.eye(3)

    R_body_ars548 = T_body_x36d[:3, :3] @ R_x36d_ars548
    T_body_ars548 = np.eye(4)
    T_body_ars548[:3, :3] = R_body_ars548
    T_body_ars548[:3, 3] = p_body_ars548
    return T_body_ars548


def Zed2LeftCam_T_Zed2RightCam():
    """ZED2左摄像头到右摄像头的变换"""
    return np.array([[0.9999729520475557, 0.0005196052489018924, -0.00733656484209896, 0.1198626857465555],
                     [-0.000506872981087257, 0.9999983625956328, 0.0017372063303895385, 4.457504283335926e-05],
                     [0.007337455490703346, -0.00173344063602287, 0.9999715780613383, 0.00035146891950921987],
                     [0.0, 0.0, 0.0, 1.0]])


def Body_T_Zed2RightCam():
    """ZED2右摄像头到车身坐标系的变换"""
    return Body_T_Zed2LeftCam() @ Zed2LeftCam_T_Zed2RightCam()


def Body_T_Zed2LeftCam():
    """ZED2左摄像头到车身坐标系的变换"""
    tx = -0.0020000000949949026
    ty = -0.023000003769993782
    tz = 0.0002200000308221206
    qx = -0.0008717564051039517
    qy = -0.00139715860132128
    qz = -0.0010711626382544637
    qw = 0.9999980330467224
    q_zed2leftcamx_zed2imu = [qw, qx, qy, qz]

    T_zed2leftcamx_zed2imu = np.eye(4)
    T_zed2leftcamx_zed2imu[:3, :3] = quat2rotm(q_zed2leftcamx_zed2imu)
    T_zed2leftcamx_zed2imu[:3, 3] = [tx, ty, tz]

    T_rdf_zed2leftcamx = np.eye(4)
    T_rdf_zed2leftcamx[:3, :3] = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])

    T_zed2leftcam_imu = T_rdf_zed2leftcamx @ T_zed2leftcamx_zed2imu
    return Body_T_Zed2Imu() @ np.linalg.inv(T_zed2leftcam_imu)


# def get_trans(sensor_A, sensor_B):
#     """获取两个传感器之间的坐标变换矩阵"""
#     T_A = globals()[f"Body_T_{sensor_A}"]()
#     T_B = globals()[f"Body_T_{sensor_B}"]()
#     T_A_to_B = np.linalg.inv(T_B) @ T_A
#     return T_A_to_B

sensor_transformation_funcs = {
    'Ars548': Body_T_Ars548,
    'Xt32':Body_T_Xt32,
    'Zed2Imu':Body_T_Zed2Imu,
    'Oculii':Body_T_Oculii,
    'Zed2RightCam':Body_T_Zed2RightCam,
    'Zed2LeftCam':Body_T_Zed2LeftCam

}

def get_trans(sensor_A, sensor_B):
    """获取两个传感器之间的坐标变换矩阵"""
    T_A = sensor_transformation_funcs[sensor_A]()  # 显式调用
    T_B = sensor_transformation_funcs[sensor_B]()  # 显式调用
    T_A_to_B = np.linalg.inv(T_B) @ T_A
    return T_A_to_B


def read_pcd_to_numpy(pcd_file):
    pc = PointCloud.from_path(pcd_file)
    #data_array = pc.pc_data
    data_array = pc.numpy()
    return data_array

def project_lidar_to_image(lidar_points, transform_matrix, K):
    num_points = lidar_points.shape[0]
    lidar_points_hom = np.hstack((lidar_points, np.ones((num_points, 1))))

    # 使用变换矩阵将雷达点转换到相机坐标系 (N, 4) * (4, 4).T -> (N, 4)
    points_camera_hom = np.dot(lidar_points_hom, transform_matrix.T)

    # 提取相机坐标系下的 x, y, z (N, 3)
    points_camera = points_camera_hom[:, :3]

    # 使用相机内参矩阵将 3D 点投影到 2D 图像平面 (N, 3) * (3, 3).T -> (N, 3)
    points_image_hom = np.dot(points_camera, K.T)

    # 归一化，得到图像坐标 (u, v)
    u = points_image_hom[:, 0] / points_image_hom[:, 2]
    v = points_image_hom[:, 1] / points_image_hom[:, 2]

    # 将结果堆叠为 (N, 2) 的数组
    image_points = np.vstack((u, v)).T

    return image_points

def visualize_lidar_projection(image, image_points):
    image_shape = image.shape
    image_points = np.round(image_points).astype(int)

    # 绘制投影点到图像上
    for point in image_points:
        x, y = point
        if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)  # 绿色点，半径为3

    # 显示图像
    cv2.imshow('Lidar Projection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def lidar_fov_to_image(lidar_to_camera_matrix, K, fov_h, fov_v, image_shape):
    img_height, img_width = image_shape[:2]
    # 定义 LiDAR 水平和竖直 FoV 的边界角度
    half_fov_h = fov_h / 2
    half_fov_v = fov_v / 2

    # 计算 FoV 边界的八个关键点（极值点）
    fov_points = []
    for theta_h in [-half_fov_h, half_fov_h]:
        for theta_v in [-half_fov_v, half_fov_v]:
            y = np.cos(np.radians(theta_v)) * np.sin(np.radians(theta_h))
            x = np.cos(np.radians(theta_v)) * np.cos(np.radians(theta_h))
            z = np.sin(np.radians(theta_v))
            fov_points.append([x, y, z, 1.0])

    # 将点从 LiDAR 坐标系转换到相机坐标系
    fov_points_camera = np.dot(lidar_to_camera_matrix, np.array(fov_points).T).T[:, :3]
    # 使用相机内参矩阵将 3D 点投影到 2D 图像平面
    image_points_hom = np.dot(fov_points_camera, K.T)
    # 归一化得到图像坐标
    u = image_points_hom[:, 0] / image_points_hom[:, 2]
    v = image_points_hom[:, 1] / image_points_hom[:, 2]
    # 将坐标限制在图像范围内
    u = np.clip(u, 0, img_width - 1)
    v = np.clip(v, 0, img_height - 1)

    # 获取重合部分的最小矩形框
    x_min = int(np.min(u))
    y_min = int(np.min(v))
    x_max = int(np.max(u))
    y_max = int(np.max(v))

    return (x_min, y_min, x_max, y_max)


def lidar_fov_to_image_asy(lidar_to_camera_matrix, K, fov_h, fov_v_down, fov_v_up, image_shape):
    img_height, img_width = image_shape[:2]

    # 定义 LiDAR 水平和竖直 FoV 的边界角度
    half_fov_h = fov_h / 2

    # 计算 FoV 边界的八个关键点（极值点）
    fov_points = []
    for theta_h in [-half_fov_h, half_fov_h]:
        for theta_v in [fov_v_down, fov_v_up]:
            y = np.cos(np.radians(theta_v)) * np.sin(np.radians(theta_h))
            x = np.cos(np.radians(theta_v)) * np.cos(np.radians(theta_h))
            z = np.sin(np.radians(theta_v))
            fov_points.append([x, y, z, 1.0])

    # 将点从 LiDAR 坐标系转换到相机坐标系
    fov_points_camera = np.dot(lidar_to_camera_matrix, np.array(fov_points).T).T[:, :3]

    # 使用相机内参矩阵将 3D 点投影到 2D 图像平面
    image_points_hom = np.dot(fov_points_camera, K.T)

    # 归一化得到图像坐标
    u = image_points_hom[:, 0] / image_points_hom[:, 2]
    v = image_points_hom[:, 1] / image_points_hom[:, 2]

    # 将坐标限制在图像范围内
    u = np.clip(u, 0, img_width - 1)
    v = np.clip(v, 0, img_height - 1)

    # 获取重合部分的最小矩形框
    x_min = int(np.min(u))
    y_min = int(np.min(v))
    x_max = int(np.max(u))
    y_max = int(np.max(v))

    return (x_min, y_min, x_max, y_max)



# def radar_fov_to_image(radar_to_camera_matrix, K, fov_h, fov_v, image_shape):
#     img_height, img_width = image_shape[:2]
#
#     # 定义 LiDAR 水平和竖直 FoV 的边界角度
#     half_fov_h = fov_h / 2
#     half_fov_v = fov_v / 2
#
#     # 计算 FoV 边界的八个关键点（极值点）
#     fov_points = []
#     for theta_h in [-half_fov_h, half_fov_h]:
#         for theta_v in [-half_fov_v, half_fov_v]:
#             x = np.cos(theta_v) * np.sin(theta_h)
#             z = np.cos(theta_v) * np.cos(theta_h)
#             y = np.sin(theta_v)
#             fov_points.append([x, y, z, 1.0])
#
#     # 将点从 LiDAR 坐标系转换到相机坐标系
#     fov_points_camera = np.dot(radar_to_camera_matrix, np.array(fov_points).T).T[:, :3]
#
#     # 使用相机内参矩阵将 3D 点投影到 2D 图像平面
#     image_points_hom = np.dot(fov_points_camera, K.T)
#
#     # 归一化得到图像坐标
#     u = image_points_hom[:, 0] / image_points_hom[:, 2]
#     v = image_points_hom[:, 1] / image_points_hom[:, 2]
#
#     # 将坐标限制在图像范围内
#     u = np.clip(u, 0, img_width - 1)
#     v = np.clip(v, 0, img_height - 1)
#
#     # 获取重合部分的最小矩形框
#     x_min = int(np.min(u))
#     y_min = int(np.min(v))
#     x_max = int(np.max(u))
#     y_max = int(np.max(v))
#
#     return (x_min, y_min, x_max, y_max)


def rotate_radar_to_lidar(radar_pcd, R_radar, R_lidar):

    # R_radar_inv = np.linalg.inv(R_radar)
    # R_radar_to_lidar = R_lidar @ R_radar_inv
    R_lidar_inv = np.linalg.inv(R_lidar)
    R_radar_to_lidar = R_lidar_inv @ R_radar

    xyz = radar_pcd[:, :3]  # (N, 3)

    rotated_xyz = (R_radar_to_lidar @ xyz.T).T  # (N, 3)

    rotated_pcd = radar_pcd.copy()
    rotated_pcd[:, :3] = rotated_xyz

    return rotated_pcd

def model_gaussian(center, sigma_x, sigma_y):
    #print('sigma_x,sigma_y',sigma_x,sigma_y)
    # Mean is the center of the radar point in image coordinates
    mean = center

    # Define the 2D Gaussian distribution with the given parameters
    def gaussian(x, y):
        return (1 / (2 * np.pi * sigma_x * sigma_y)) * np.exp(-0.5 * (((x - mean[0]) / sigma_x) ** 2 + ((y - mean[1]) / sigma_y) ** 2))

    return gaussian




def update_radar_map_with_gaussian(radar_2d_map, r_in_2d, std_var):

    rows, cols = radar_2d_map.shape

    radar_2d_map_pro = np.zeros((rows, cols))
    for i in range(len(r_in_2d)):

        u, v = int(np.floor(r_in_2d[i, 0])), int(np.floor(r_in_2d[i, 1]))

        if 0 <= u < cols and 0 <= v < rows:

            sigma_x, sigma_y = std_var[0], std_var[1]

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


def process_frame(frame):
    radar_arg_file, radar_eagle_file, lidar_file, imu_file, cam_left_file, cam_right_file = get_snail_dir(df, frame,
                                                                                                          'data4')
    T_radar = get_trans('Oculii','Zed2LeftCam')
    if os.path.exists(f'D:/snail_radar/20231208/data4/eagle/pmap_image/{frame}.jpg'):
        return

    print(f'Processing frame {frame}')
    K = get_K()


    image = cv2.imread(cam_left_file)
    coor = [0,0,641,317]        #boundary + 1

    r_in = np.load(f'D:/snail_radar/20231208/data4/eagle/r_in/{frame}.npy')
    #r_in_2d = np.load(f'D:/snail_radar/20231208/data4/r_in_2d/{frame}.npy')
    radar_2d_map = np.zeros((316, 640))

    r_in, r_in_2d = utils1.radar_in_image(coor, r_in, T_radar, K)

    standard_variance = [100, 100]
    radar_2d_map_pro = update_radar_map_with_gaussian(radar_2d_map, r_in_2d, standard_variance)

    np.save(f'D:/snail_radar/20231208/data4/eagle/radar_2d_map_pro/{frame}.npy',
            radar_2d_map_pro)

    normalized_array = cv2.normalize(radar_2d_map_pro, None, 0, 255, cv2.NORM_MINMAX)
    gray_image_pro = normalized_array.astype(np.uint8)
    cv2.imwrite(f'D:/snail_radar/20231208/data4/eagle/pmap_image/{frame}.jpg',
                gray_image_pro)





if __name__ == '__main__':
    # df = get_frame_data()
    # df.to_csv('D:/snail_radar/20231208/data4/frames_timestamps.csv', index=False)
    columns_to_read_as_str = ['timestamp', 'cam_left_time', 'cam_right_time',"radar_arg_time","radar_eagle_time",'lidar_time','odom_time']
    df = pd.read_csv('D:/snail_radar/20231208/data4/frames_timestamps.csv', usecols=columns_to_read_as_str, dtype=str)
    # print(Body_T_Xt32())
    # print(get_trans('Zed2LeftCam','Xt32'))


    # for frame in range(1000, 1001):
    #
    #     radar_arg_file, radar_eagle_file, lidar_file, imu_file, cam_left_file, cam_right_file = get_snail_dir(df, frame,'data4')
    #     # T = get_transform_matrix(calib_file, 'radar_6455','lidar_vlp16')
    #     T_radar = get_trans('Oculii','Zed2LeftCam')
    #     T_lidar = get_trans('Xt32','Zed2LeftCam')
    #     radar_pcd = read_pcd_to_numpy(radar_eagle_file)
    #     lidar_pcd = read_pcd_to_numpy(lidar_file)
    #     print(radar_pcd[0,:])
    #
    #
    #     K = get_K()
    #
    #     image = cv2.imread(cam_left_file)
    #     image_points = project_lidar_to_image(radar_pcd[:, 0:3], T_radar, K)
    #     vmin, vmax = min(image_points[:, 1]), max(image_points[:, 1])
    #     # print(vmin, vmax)
    #     visualize_lidar_projection(image, image_points)

    #     T_l2r = get_trans('Xt32','Oculii')
    #     lidar_pcd = lidar_pcd[:, 0:3]
    #     lidar_pcd = utils.trans_point_coor(lidar_pcd,T_l2r)
    #     image_points = project_lidar_to_image(lidar_pcd[:, 0:3], T_radar, K)
    #     vmin,vmax = min(image_points[:,1]), max(image_points[:,1])
    #     # print(vmin, vmax)
    #     visualize_lidar_projection(image, image_points)
    #
    #
    #
    #     coord = lidar_fov_to_image(T_radar, K, 113, 45, image.shape)
    #     print(coord)
    #
    #     coord = lidar_fov_to_image(T_radar, K, 180, 31, image.shape)
    #     print(coord)
    #
    #     coord = lidar_fov_to_image_asy(T_radar, K, 113, -25, 25, image.shape)
    #     print(coord)
    #
    #     coord = lidar_fov_to_image_asy(T_radar, K, 180, -25, 15,  image.shape)
    #     print(coord)
    #
    # lidar_yaw_min, lidar_yaw_max = 1000, -1000
    # lidar_pitch_min, lidar_pitch_max = 1000, -1000
    # for frame in range(0, len(df)):
    #     print('frame', frame)
    #     radar_arg_file, radar_eagle_file, lidar_file, imu_file, cam_left_file, cam_right_file = get_snail_dir(df, frame,'data4')
    #     if not (os.path.exists(radar_eagle_file) and os.path.exists(lidar_file) and os.path.exists(
    #             cam_left_file)):
    #         print('file missing at frame ', frame)
    #         continue
    #
    #     # lidar_pcd = read_pcd_to_numpy(lidar_file)[:,0:3]
    #     # T_l2r = get_trans('Xt32','Ars548')
    #     # lidar_pcd = utils.trans_point_coor(lidar_pcd,T_l2r)
    #     # lidar_pcd = lidar_pcd[lidar_pcd[:,0]>0]
    #     radar_pcd = read_pcd_to_numpy(radar_eagle_file)
    #
    #     max_height, min_height, yaw_range, pitch_range = calculate_height_and_angles(radar_pcd)
    #
    #     # print(f"最大高度: {max_height:.2f}, 最小高度: {min_height:.2f}")
    #     yaw_range_degrees = (np.degrees(yaw_range[0]), np.degrees(yaw_range[1]))
    #     pitch_range_degrees = (np.degrees(pitch_range[0]), np.degrees(pitch_range[1]))
    #     if yaw_range_degrees[0] < lidar_yaw_min:
    #         lidar_yaw_min = yaw_range_degrees[0]
    #     if yaw_range_degrees[1] > lidar_yaw_max:
    #         lidar_yaw_max = yaw_range_degrees[1]
    #     if pitch_range_degrees[0] < lidar_pitch_min:
    #         lidar_pitch_min = pitch_range_degrees[0]
    #     if pitch_range_degrees[1] > lidar_pitch_max:
    #         lidar_pitch_max = pitch_range_degrees[1]
    #     print(f"Yaw 角范围 (度): {yaw_range_degrees[0]:.2f}° 到 {yaw_range_degrees[1]:.2f}°")
    #     print(f"Pitch 角范围 (度): {pitch_range_degrees[0]:.2f}° 到 {pitch_range_degrees[1]:.2f}°")
    # print('Final:')
    # print(f"Yaw 角范围 (度): {lidar_yaw_min:.2f}° 到 {lidar_yaw_max:.2f}°")
    # print(f"Pitch 角范围 (度): {lidar_pitch_min:.2f}° 到 {lidar_pitch_max:.2f}°")

# ----------------------------------------------projection---------------------------------------------------------------
#     for frame in range(0,len(df)):
#         print(f'frame:{frame}')
#         radar_arg_file, radar_eagle_file, lidar_file, imu_file, cam_left_file, cam_right_file = get_snail_dir(df, frame,'data4')
#         coor = [0,0,640,360]
#         T_radar = get_trans('Oculii', 'Zed2LeftCam')
#         T_lidar = get_trans('Xt32', 'Zed2LeftCam')
#         lidar_pcd = read_pcd_to_numpy(lidar_file)[:,0:3]
#         T_l2r = get_trans('Xt32','Oculii')
#         lidar_pcd = utils.trans_point_coor(lidar_pcd,T_l2r)
#         lidar_pcd = lidar_pcd[lidar_pcd[:,0]>0]
#
#         radar_pcd = read_pcd_to_numpy(radar_eagle_file)
#         radar_pcd = radar_pcd[radar_pcd[:,0]<=120]
#
#         K = get_K()
#         image = cv2.imread(cam_left_file)
#         l_in, l_in_2d = utils1.radar_in_image(coor, lidar_pcd, T_radar,K)   #lidar points have already been transformed
#         print(f'len(lidar_pcd):{len(lidar_pcd)}')
#         print(f'len(l_in):{len(l_in)}')
#         np.save(f'D:/snail_radar/20231208/data4/eagle/l_in_2d/{frame}.npy',l_in_2d)
#         np.save(f'D:/snail_radar/20231208/data4/eagle/l_in/{frame}.npy',l_in)
#         r_in, r_in_2d = utils1.radar_in_image(coor, radar_pcd, T_radar,K)
#         np.save(f'D:/snail_radar/20231208/data4/eagle/r_in_2d/{frame}.npy',r_in_2d)
#         np.save(f'D:/snail_radar/20231208/data4/eagle/r_in/{frame}.npy',r_in)
#         #velocity_list = r_in[:,3]
#         print(f'len(radar_pcd):{len(radar_pcd)}')
#         print(f'len(r_in):{len(r_in)}')


#--------------------------------------------make dataset---------------------------------------------------------------
    # odom_dir = 'D:/snail_radar/20231208/data4/zed2i/odom.txt'
    # odom_data = pd.read_csv(odom_dir, delim_whitespace=True, header=None, dtype=str)
    # odom_data.columns = ['odom_time', 'x', 'y', 'z', 'q1', 'q2', 'q3', 'q4']
    # odom_data[['odom_time', 'x', 'y', 'z']] = odom_data[['odom_time', 'x', 'y', 'z']].astype(float)
    # column_names = ['Frame', 'PointNum', 'Velocity']
    # df_array = df.values.astype(float)
    # new_df = pd.DataFrame(columns=column_names)
    # for frame in range(0,len(df)):
    #     print(f'frame:{frame}')
    #     timestamp = df_array[frame,0]
    #     idx = (np.abs(odom_data['odom_time'] - timestamp)).argmin()
    #     if idx == 0:
    #         t1, t2 = odom_data.loc[0, 'odom_time'], odom_data.loc[1, 'odom_time']
    #         x1, y1, z1 = odom_data.loc[0, ['x', 'y', 'z']].values
    #         x2, y2, z2 = odom_data.loc[1, ['x', 'y', 'z']].values
    #     else:
    #         t1, t2 = odom_data.loc[idx - 1, 'odom_time'], odom_data.loc[idx, 'odom_time']
    #         x1, y1, z1 = odom_data.loc[idx - 1, ['x', 'y', 'z']].values
    #         x2, y2, z2 = odom_data.loc[idx, ['x', 'y', 'z']].values
    #
    #     time_diff = t2 - t1
    #     if time_diff == 0:
    #         v_x = v_y = v_z = 0
    #     else:
    #         v_x = (x2 - x1) / time_diff
    #         v_y = (y2 - y1) / time_diff
    #         v_z = (z2 - z1) / time_diff
    #
    #     ego_velo = np.sqrt(v_x ** 2 + v_y ** 2 + v_z ** 2)
    #     print(ego_velo)
    #
    #     radar_arg_file, radar_eagle_file, lidar_file, imu_file, cam_left_file, cam_right_file = get_snail_dir(df, frame,'data4')
    #     T_radar = get_trans('Oculii', 'Zed2LeftCam')
    #     T_lidar = get_trans('Xt32', 'Zed2LeftCam')
    #     lidar_pcd = read_pcd_to_numpy(lidar_file)[:, 0:3]
    #     T_l2r = get_trans('Xt32', 'Oculii')
    #
    #     radar_pcd = np.load(f'D:/snail_radar/20231208/data4/eagle/r_in/{frame}.npy')
    #     num_points = len(radar_pcd)
    #     #dynamic_pcd, static_pcd = classify_points(radar_pcd,data)
    #     #print(f'len(dynamic):{len(dynamic_pcd)},len(static):{len(static_pcd)}')
    #     #ego_velo = estimate_radar_velocity_ransac(static_pcd)
    #     #print(ego_velo,np.linalg.norm(ego_velo))
    #     print('pointnum',num_points)
    #
    #     new_data = pd.DataFrame({'Frame': [frame], 'PointNum': [num_points], 'Velocity': [np.linalg.norm(ego_velo)]})
    #     new_df = pd.concat([new_df, new_data], ignore_index=True)
    # new_df.to_csv('D:/snail_radar/20231208/data4/eagle/prob_dataset.csv', index=False)

    # -------------------------------------------------------edge/corner detection-------------------------------------------
    # for frame in range(0,len(df)):
    #     radar_arg_file, radar_eagle_file, lidar_file, imu_file, cam_left_file, cam_right_file = get_snail_dir(df, frame,'data4')
    #     cam_dir = cam_left_file
    #     des_dir = 'D:/snail_radar/20231208/data4/edge_image/'+str(int(frame))+".jpg"
    #     image = cv2.imread(cam_dir)
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #
    #     # 边缘检测
    #     edges = cv2.Canny(gray, 50, 150)
    #
    #     # 角点检测 (Harris)
    #     dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    #     dst = cv2.dilate(dst, None)
    #
    #     # 创建一个全黑图像，用于存放边缘和角点信息
    #     combined = np.zeros_like(gray)
    #
    #     # 将边缘信息添加到 combined 图像中
    #     combined[edges > 0] = 128  # 用128表示边缘
    #
    #     # 将角点信息添加到 combined 图像中
    #     combined[dst > 0.01 * dst.max()] = 255  # 用255表示角点
    #
    #     # 显示和保存结果
    #     # cv2.imshow('Combined Edges and Corners', combined)
    #     # cv2.imwrite('combined_output.jpg', combined)
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()
    #     cropped_edges = combined[0:316, 0: 640]
    #     cv2.imwrite(des_dir,cropped_edges)
    #     print(des_dir)
    #     print(edges.shape)

    # ----------------------------------------------distribution map-----------------------------------------------------
    # for frame in range(0,len(df)):
    #     print(frame)
    #     radar_arg_file, radar_eagle_file, lidar_file, imu_file, cam_left_file, cam_right_file = get_snail_dir(df, frame,'data4')
    #     image = cv2.imread(cam_left_file)
    #     r_in_2d = np.load(f'D:/snail_radar/20231208/data4/eagle/r_in_2d/{frame}.npy')
    #     new_image = utils.visualize_with_image_color(image, r_in_2d, [0, 0, 255])
    #     cv2.imwrite(f'D:/snail_radar/20231208/data4/eagle/projection_radar/{frame}.jpg',new_image)

    # v_max = 0
    # for frame in range(0, len(df)):
    #     r_in_2d = np.load(f'D:/snail_radar/20231208/data4/eagle/l_in_2d/{frame}.npy')
    #     min_u, max_u = min(r_in_2d[:,0]),max(r_in_2d[:,0])
    #     min_v, max_v = min(r_in_2d[:,1]),max(r_in_2d[:,1])
    #     if 0 <= min_v <= max_v <= 305:
    #         continue
    #     else:
    #         v_max = max(v_max, max_v)
    # print(v_max)    #316





    # frames = range(0, len(df))
    #
    # # 使用joblib并行处理
    # Parallel(n_jobs=8)(delayed(process_frame)(frame) for frame in frames)

    #------------------------------------------------split dataset-----------------------------------------------------
    # random_seed = 2024
    # np.random.seed(random_seed)
    #
    # # 读取数据集
    # df = pd.read_csv('D:/snail_radar/20231208/data4/eagle/prob_dataset.csv').iloc[500:4648,:]
    # df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    #
    # # 自定义划分比例
    # train_ratio = 0.7  # 训练集占 70%
    # test_ratio = 0.3  # 测试集占 30%
    #
    # # 确保比例总和为1
    # assert train_ratio + test_ratio == 1.0
    #
    # # 计算划分索引
    # train_index = int(train_ratio * len(df))
    #
    # # 数据集划分
    # train_df = df[:train_index]
    # test_df = df[train_index:]
    #
    # # 打印每个划分的大小
    # print(f"训练集大小: {len(train_df)}")
    # print(f"测试集大小: {len(test_df)}")
    #
    # # 保存划分后的数据集（可选）
    # train_df.to_csv('D:/snail_radar/20231208/data4/eagle/train_dataset.csv',
    #                 index=False)
    # test_df.to_csv('D:/snail_radar/20231208/data4/eagle/test_dataset.csv', index=False)
    #
    # #检查PointNum列的最小值和最大值
    # print(np.min(train_df['PointNum'].values), np.max(train_df['PointNum'].values))

    # -------------------------------------------------RCS regression--------------------------------------------------------
    # train_df = pd.read_csv("D:/snail_radar/20231208/data4/eagle/train_dataset.csv").values
    # test_df = pd.read_csv("D:/snail_radar/20231208/data4/eagle/test_dataset.csv").values
    # column_names = ['Frame', 'Index', 'x', 'y', 'z', 'u', 'v', 'v_r', 'rcs']
    # new_df_train = pd.DataFrame(columns=column_names)
    # for i in range(len(test_df)):
    #     frame, pointnum, ego_velocity = test_df[i]
    #     frame, pointnum = int(frame), int(pointnum)
    #     print('frame', frame)
    #     radar_arg_file, radar_eagle_file, lidar_file, imu_file, cam_left_file, cam_right_file = get_snail_dir(df, frame,'data4')
    #     K = get_K()
    #     if K is None:
    #         print(f'frame {frame} missing intrinsic matrix.')
    #         continue
    #     #T_lidar2radar = get_trans('Xt32','Ars548')
    #
    #     r_in = np.load(f'D:/snail_radar/20231208/data4/eagle/r_in/{frame}.npy')
    #     r_in_2d = np.load(f'D:/snail_radar/20231208/data4/eagle/r_in_2d/{frame}.npy')
    #     T_radar = get_trans('Oculii', 'Zed2LeftCam')
    #     T_lidar = get_trans('Xt32', 'Zed2LeftCam')
    #     mask = radar_in_lidarfov(r_in,T_radar,T_lidar)
    #     r_in = r_in[mask,:]
    #     r_in_2d = r_in_2d[mask,:]
    #     l_in = np.load(
    #         f'D:/snail_radar/20231208/data4/eagle/l_in/{frame}.npy')
    #     #l_in_radar_coor = utils.trans_point_coor(l_in, T_lidar2radar)
    #     l_in_radar_coor = l_in
    #     # np.save(f'D:/snail_radar/20231208/data4/eagle/l_in_radar_coor/{frame}.npy',
    #     #         l_in_radar_coor)
    #     kdtree = KDTree(l_in_radar_coor)
    #     n = 50
    #     sampled_points, sampled_points_2d = sample_points_from_point_cloud(r_in, r_in_2d, n)
    #     #idx = 0
    #     for j in range(0, min(n,len(r_in))):
    #         x, y, z, v_r, _, rcs = sampled_points[j, 0:6]
    #         u, v = sampled_points_2d[j, :]
    #
    #         r = 1
    #
    #         pitch = np.degrees(np.arctan(z / np.linalg.norm([x, y], 2)))
    #
    #         local_lidar_index = kdtree.query_ball_point([x, y, z], r)
    #         local_lidar_radarcoor = l_in_radar_coor[local_lidar_index]
    #         # if len(local_lidar_index) == 0:
    #         #     continue
    #         new_data = pd.DataFrame(
    #             {'Frame': [frame], 'Index': [j], 'x': [x], 'y': [y], 'z': [z], 'u': [u], 'v': [v], 'v_r': [v_r],
    #              'rcs': [rcs]})
    #         new_df_train = pd.concat([new_df_train, new_data], ignore_index=True)
    #
    #         np.save(f'D:/snail_radar/20231208/data4/eagle/local_points/{frame}_{j}.npy',
    #                 local_lidar_radarcoor)
    #         dst = f'D:/snail_radar/20231208/data4/eagle/ablation/range_image_{r}/{frame}_{j}.jpg'
    #         #virtual_point = get_new_point(x, y, z, 1)
    #         fov_down = -25
    #         fov_up = 15
    #         for k in range(len(local_lidar_radarcoor)):
    #             # local_lidar_radarcoor[k, 0] -= virtual_point[0]
    #             # local_lidar_radarcoor[k, 1] -= virtual_point[1]
    #             # local_lidar_radarcoor[k, 2] -= virtual_point[2]
    #             local_lidar_radarcoor[k, 0] -= x
    #             local_lidar_radarcoor[k, 1] -= y
    #             local_lidar_radarcoor[k, 2] -= z
    #             x1, y1, z1 = local_lidar_radarcoor[k]
    #             # pitch = np.degrees(np.arctan(z1 / np.linalg.norm([x1, y1], 2)))
    #             # if pitch > fov_up:
    #             #     fov_up = pitch
    #             # if pitch < fov_down:
    #             #     fov_down = pitch
    #
    #         proj_H, proj_W, = 32, 128
    #         gen_range_image_rcs_translation(local_lidar_radarcoor,proj_H, proj_W, r,
    #                             dst)
    #
    #         #idx += 1
    # new_df_train.to_csv('D:/snail_radar/20231208/data4/eagle/RCS_dataset_test.csv')

    # --------------------------------make RCS regression dataset wo vp-------------------------------------------------
    # #     train_df = pd.read_csv("D:/Astyx dataset/dataset_astyx_hires2019/train_dataset.csv").values
    # #     test_df = pd.read_csv("D:/Astyx dataset/dataset_astyx_hires2019/test_dataset.csv").values
    # #     column_names = ['Frame', 'Index', 'x', 'y', 'z', 'u', 'v', 'v_r', 'rcs']
    # df = pd.read_csv('D:/snail_radar/20231208/data4/RCS_dataset_test.csv').values[:, 1:]
    # for i in range(len(df)):
    #     frame, index, x, y, z, u, v, v_r, rcs = df[i]
    #     frame, index = int(frame), int(index)
    #     # l_in_radar_coor = np.load(f'D:/snail_radar/20231208/data4/l_in/{int(frame)}.npy')
    #     # kdtree = KDTree(l_in_radar_coor)
    #     #
    #     # r = 1
    #     #
    #     # pitch = np.degrees(np.arctan(z / np.linalg.norm([x, y], 2)))
    #     #
    #     # local_lidar_index = kdtree.query_ball_point([x, y, z], r)
    #     #
    #     # local_lidar_radarcoor = l_in_radar_coor[local_lidar_index]
    #
    #     local_lidar_radarcoor = np.load(f'D:/snail_radar/20231208/data4/eagle/local_points/{frame}_{index}.npy')
    #     dst = f'D:/snail_radar/20231208/data4/eagle/ablation/range_image/{int(frame)}_{int(index)}.jpg'
    #     # virtual_point = get_new_point(x, y, z)
    #     fov_down = -25
    #     fov_up = 15
    #     # for k in range(len(local_lidar_radarcoor)):
    #     #     local_lidar_radarcoor[k, 0] -= virtual_point[0]
    #     #     local_lidar_radarcoor[k, 1] -= virtual_point[1]
    #     #     local_lidar_radarcoor[k, 2] -= virtual_point[2]
    #     #     x1, y1, z1 = local_lidar_radarcoor[k]
    #     #     pitch = np.degrees(np.arctan(z1 / np.linalg.norm([x1, y1], 2)))
    #     #     if pitch > fov_up:
    #     #         fov_up = pitch
    #     #     if pitch < fov_down:
    #     #         fov_down = pitch
    #
    #     proj_H, proj_W, = 32, 128
    #     gen_range_image_rcs(local_lidar_radarcoor, fov_up, fov_down, proj_H, proj_W,
    #                         dst)

# -------------------------------make RCS regression dataset for optimal parameter fiding------------------------------
    for r in [2,3,4]:
        def process_data(row):
            frame, index, x, y, z, u, v, v_r, rcs = row
            l_in_radar_coor = np.load(f'D:/snail_radar/20231208/data4/eagle/l_in/{int(frame)}.npy')
            kdtree = KDTree(l_in_radar_coor)
            if not os.path.exists(f'D:/snail_radar/20231208/data4/eagle/ablation/newest/range_image_{r}'):
                os.mkdir(f'D:/snail_radar/20231208/data4/eagle/ablation/newest/range_image_{r}')

            pitch = np.degrees(np.arctan(z / np.linalg.norm([x, y], 2)))

            local_lidar_index = kdtree.query_ball_point([x, y, z], r)
            local_lidar_radarcoor = l_in_radar_coor[local_lidar_index]

            dst = f'D:/snail_radar/20231208/data4/eagle/ablation/newest/range_image_{r}/{int(frame)}_{int(index)}.jpg'
            fov_down = -25
            fov_up = 15
            #virtual_point = get_new_point(x, y, z, r)
            original_points = local_lidar_radarcoor.copy()
            for k in range(len(local_lidar_radarcoor)):
                # local_lidar_radarcoor[k, 0] -= virtual_point[0]
                # local_lidar_radarcoor[k, 1] -= virtual_point[1]
                # local_lidar_radarcoor[k, 2] -= virtual_point[2]
                local_lidar_radarcoor[k, 0] -= x
                local_lidar_radarcoor[k, 1] -= y
                local_lidar_radarcoor[k, 2] -= z
                x1, y1, z1 = local_lidar_radarcoor[k]
                # pitch = np.degrees(np.arctan(z1 / np.linalg.norm([x1, y1], 2)))
                # if pitch > fov_up:
                #     fov_up = pitch
                # if pitch < fov_down:
                #     fov_down = pitch

            proj_H, proj_W = 32, 128
            #gen_range_image_rcs(local_lidar_radarcoor, fov_up, fov_down, proj_H, proj_W, dst)
            gen_range_image_rcs_translation_correct(local_lidar_radarcoor, original_points, proj_H, proj_W, [x,y,z], r, dst)


        df = pd.read_csv('D:/snail_radar/20231208/data4/eagle/RCS_dataset_test.csv').values[:, 1:]
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            executor.map(process_data, df)
        df = pd.read_csv('D:/snail_radar/20231208/data4/eagle/RCS_dataset_train.csv').values[:, 1:]
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            executor.map(process_data, df)