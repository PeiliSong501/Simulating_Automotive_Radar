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

from pypcd4 import PointCloud


def transform_r2l(radar_pcd, T_r2l):
    xyz_radar = radar_pcd[:, :3]
    num_points = xyz_radar.shape[0]
    xyz_radar_homogeneous = np.hstack((xyz_radar, np.ones((num_points, 1))))

    xyz_lidar_homogeneous = (T_r2l @ xyz_radar_homogeneous.T).T

    radar_pcd[:, :3] = xyz_lidar_homogeneous[:, :3]

    return radar_pcd



#base_dir = 'E:/MSCRadar/RURAL_A0/'
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




def get_msc_dir(frame,seq='RURAL_A0'):
    base = f'E:/MSCRadar/{seq}/'
    cam_base_dir = base + '1_IMAGE/LEFT/'
    radar_base_dir = base + '3_RADAR/PCD/'

    lidar_base_dir = base+'2_LIDAR/PCD/'
    radar_file = radar_base_dir + str(frame).zfill(6) + '.pcd'
    lidar_file = lidar_base_dir + str(frame).zfill(6) + '.pcd'
    cam_file = cam_base_dir + str(frame).zfill(6) + '.png'
    return radar_file,lidar_file,cam_file


def get_transform_matrix(seq, sensor='RADAR'):
    base = f'E:/MSCRadar/{seq}/'
    calib_base_dir = base + f'5_CALIBRATION_RURAL/CALIBRATION_CAMERA_{sensor}.txt'

    with open(calib_base_dir, 'r') as file:
        lines = file.readlines()

    # 提取只包含数值的行
    num_lines = []
    for line in lines:
        # 检查行中是否只包含数值和空格
        if re.match(r'^[-0-9.eE\s]+$', line.strip()):
            num_lines.append(line)

    # 如果没有足够的数值行，返回错误
    if len(num_lines) < 4:
        raise ValueError("文件中没有足够的数值行来构建变换矩阵。")

    # 提取旋转矩阵
    rotation_matrix = []
    for i in range(3):
        rotation_matrix.append([float(x) for x in num_lines[i].split()])
    rotation_matrix = np.array(rotation_matrix)

    # 提取平移向量
    translation_vector = [float(x) for x in num_lines[3].split()]
    translation_vector = np.array(translation_vector).reshape(3, 1)

    # 构造4x4变换矩阵
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = translation_vector.flatten()

    return transform_matrix


def lidar_to_radar_transform_matrix(seq):
    lidar_to_camera = get_transform_matrix(seq, 'LIDAR')
    radar_to_camera = get_transform_matrix(seq, 'RADAR')

    lidar_to_radar = np.linalg.inv(radar_to_camera) @ lidar_to_camera

    return lidar_to_radar



def get_camera_intrinsics(seq, left=True):
    # 文件路径
    base = f'E:/MSCRadar/{seq}/'
    calib_base_dir = base + '5_CALIBRATION_RURAL/CALIBRATION_CAMERA.txt'

    # 打开并读取文件
    with open(calib_base_dir, 'r') as f:
        lines = f.readlines()

    # 定义要查找的关键字
    if left:
        camera_keyword = "Camera1 (LEFT)"
    else:
        camera_keyword = "Camera2 (RIGHT)"

    # 查找关键字所在的行
    start_line = -1
    for idx, line in enumerate(lines):
        if camera_keyword in line:
            start_line = idx + 1  # 内参矩阵的起始行是关键字行的下一行
            break

    if start_line == -1:
        raise ValueError(f"Could not find the keyword '{camera_keyword}' in the calibration file.")

    # 读取内参矩阵
    K = []
    for i in range(1,4):
        # 分割每一行，并转换为浮点数
        K.append(list(map(float, lines[start_line + i].strip().split())))

    return K


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
            x = np.cos(np.radians(theta_v)) * np.sin(np.radians(theta_h))
            y = np.cos(np.radians(theta_v)) * np.cos(np.radians(theta_h))
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


def radar_fov_to_image(radar_to_camera_matrix, K, fov_h, fov_v, image_shape):
    img_height, img_width = image_shape[:2]

    # 定义 LiDAR 水平和竖直 FoV 的边界角度
    half_fov_h = fov_h / 2
    half_fov_v = fov_v / 2

    # 计算 FoV 边界的八个关键点（极值点）
    fov_points = []
    for theta_h in [-half_fov_h, half_fov_h]:
        for theta_v in [-half_fov_v, half_fov_v]:
            x = np.cos(np.radians(theta_v)) * np.sin(np.radians(theta_h))
            z = np.cos(np.radians(theta_v)) * np.cos(np.radians(theta_h))
            y = np.sin(np.radians(theta_v))
            fov_points.append([x, y, z, 1.0])

    # 将点从 LiDAR 坐标系转换到相机坐标系
    fov_points_camera = np.dot(radar_to_camera_matrix, np.array(fov_points).T).T[:, :3]

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


def read_wheel_data(file_path):
    wheel_columns = ['index', 'timestamp_sec', 'timestamp_nsec', 'linear_velocity', 'angular_velocity']
    wheel_data = pd.read_csv(file_path, delim_whitespace=True, header=None, names=wheel_columns)
    return wheel_data


# 根据序号获取线速度
def get_velocity_by_index(wheel_data, target_index):
    # 查找对应的行
    row = wheel_data.iloc[target_index]

    # 检查是否找到对应序号
    if not row.empty:
        return row['linear_velocity']
    else:
        raise ValueError(f"未找到序号为 {target_index} 的数据")



if __name__ == '__main__':
    for frame in range(1000, 1001):
        radar_file, lidar_file, cam_file = get_msc_dir(frame,'RURAL_A0')
        radar_pcd = read_pcd_to_numpy(radar_file)
        lidar_pcd = read_pcd_to_numpy(lidar_file)


        # print(np.min(radar_pcd[:,0]),np.max(radar_pcd[:,0]),np.mean(radar_pcd[:,0]))
        # print(np.min(radar_pcd[:, 1]), np.max(radar_pcd[:, 1]),np.mean(radar_pcd[:,1]))
        # print(np.min(radar_pcd[:, 2]), np.max(radar_pcd[:, 2]),np.mean(radar_pcd[:,2]))

        lidar_pcd = lidar_pcd[lidar_pcd[:, 0] > 0][:, 0:3]
        K = get_camera_intrinsics('RURAL_A0')
        K = np.array(K)
        image = cv2.imread(cam_file)
        # print(image.shape)

        T_lidar = get_transform_matrix( 'RURAL_A0', 'LIDAR')
        T_radar = get_transform_matrix( 'RURAL_A0', 'RADAR')
        #print(T_radar,T_lidar)
        T_r2l = np.linalg.inv(T_lidar) @ T_radar
        #print(T_r2l)
        print(radar_pcd[0:5, :3])
        radar_in_lidar = transform_r2l(radar_pcd,T_r2l)
        print(radar_in_lidar[0:5,:3])
        # print(len(radar_in_lidar))
        # radar_in_lidar = radar_in_lidar[radar_in_lidar[:,0] <= 0]
        # print(len(radar_in_lidar))


        image_points = project_lidar_to_image(radar_in_lidar[:, 0:3], T_lidar, K)
        visualize_lidar_projection(image, image_points)
        coord = radar_fov_to_image(T_lidar, K, 113, 45, image.shape)
        print(coord)

        image_points = project_lidar_to_image(lidar_pcd[:, 0:3], T_lidar, K)
        visualize_lidar_projection(image, image_points)
        coord = lidar_fov_to_image(T_lidar, K, 180, 45, image.shape)
        print(coord)
        #coord = [0, 0, 443, 539]
    lidar_yaw_min, lidar_yaw_max = 1000, -1000
    lidar_pitch_min, lidar_pitch_max = 1000, -1000
    for frame in range(0, 1):
        print('frame', frame)
        radar_file, lidar_file, cam_file = get_msc_dir(frame)
        if not (os.path.exists(radar_file) and os.path.exists(lidar_file) and os.path.exists(
                cam_file)):
            print('file missing at frame ', frame)
            continue

        lidar_pcd = read_pcd_to_numpy(lidar_file)

        lidar_pcd = lidar_pcd[lidar_pcd[:, 0] > 0][:, 0:3]
        print(len(lidar_pcd))
        # radar_pcd = read_pcd_to_numpy(radar_file)[:,0:3]
        # lidar_pcd = utils.trans_point_coor(radar_pcd,T_r2l)

        max_height, min_height, yaw_range, pitch_range = calculate_height_and_angles(lidar_pcd)

        # print(f"最大高度: {max_height:.2f}, 最小高度: {min_height:.2f}")
        yaw_range_degrees = (np.degrees(yaw_range[0]), np.degrees(yaw_range[1]))
        pitch_range_degrees = (np.degrees(pitch_range[0]), np.degrees(pitch_range[1]))
        if yaw_range_degrees[0] < lidar_yaw_min:
            lidar_yaw_min = yaw_range_degrees[0]
        if yaw_range_degrees[1] > lidar_yaw_max:
            lidar_yaw_max = yaw_range_degrees[1]
        if pitch_range_degrees[0] < lidar_pitch_min:
            lidar_pitch_min = pitch_range_degrees[0]
        if pitch_range_degrees[1] > lidar_pitch_max:
            lidar_pitch_max = pitch_range_degrees[1]
        print(f"Yaw 角范围 (度): {yaw_range_degrees[0]:.2f}° 到 {yaw_range_degrees[1]:.2f}°")
        print(f"Pitch 角范围 (度): {pitch_range_degrees[0]:.2f}° 到 {pitch_range_degrees[1]:.2f}°")
    print('Final:')
    print(f"Yaw 角范围 (度): {lidar_yaw_min:.2f}° 到 {lidar_yaw_max:.2f}°")
    print(f"Pitch 角范围 (度): {lidar_pitch_min:.2f}° 到 {lidar_pitch_max:.2f}°")


