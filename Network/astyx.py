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
from scipy.spatial import KDTree
from scipy.stats import ks_2samp
import concurrent.futures

def radar_in_lidarfov(pcd_radar, T_radar2lidar):
    mask = []
    for i in range(len(pcd_radar)):
        geo = pcd_radar[i][0:3]
        geo_lidarcoor = utils1.transform_coor_to_radar(geo, T_radar2lidar)
        # yaw = np.arctan2(geo_lidarcoor[1], geo_lidarcoor[0])
        pitch = np.arctan2(geo_lidarcoor[2], np.linalg.norm(geo_lidarcoor[:2]))
        # yaw_deg = np.degrees(yaw)
        pitch_deg = np.degrees(pitch)
        r = np.linalg.norm(geo_lidarcoor,axis=0)
        if -15 < pitch_deg < 15 and 0 <= r <= 100:
            mask.append(i)
    # radar = pcd_radar[mask,:]
    return mask



def gen_range_image_rcs_translation(local_points, proj_H, proj_W, r_l, dst):
    print("len of local_points:",len(local_points))
    # fov_up = fov_up / 180.0 * np.pi
    # fov_down = fov_down / 180.0 * np.pi
    # fov = abs(fov_down) + abs(fov_up)
    #
    depth = np.linalg.norm(local_points[:, :3], 2, axis=1)
    #depth = local_points[:,0]

    scan_x = local_points[:, 1]
    scan_y = local_points[:, 2]
    scan_z = local_points[:, 0]

    # yaw = -np.arctan2(scan_y, scan_x)
    # pitch = np.arcsin(scan_z / depth)


    proj_x = 0.5*(-scan_x/r_l + 1.0)  # in [0.0, 1.0]
    proj_y = 1 - (scan_y +r_l / 2*r_l)  # in [0.0, 1.0]

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
    range_norm = np.minimum((depth/r_l) * 255, 254)
    # proj_range[proj_y, proj_x] = depth*100  #multiply depth by 100, or the color will not be able tell
    proj_range[proj_y, proj_x] = range_norm

    # print('haha')
    # print(np.mean(range_norm), np.max(range_norm), np.mean(range_norm))

    cv2.imwrite(dst, proj_range)



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

def model_gaussian(center, sigma_x, sigma_y):
    #print('sigma_x,sigma_y',sigma_x,sigma_y)
    # Mean is the center of the radar point in image coordinates
    mean = center

    # Define the 2D Gaussian distribution with the given parameters
    def gaussian(x, y):
        return (1 / (2 * np.pi * sigma_x * sigma_y)) * np.exp(-0.5 * (((x - mean[0]) / sigma_x) ** 2 + ((y - mean[1]) / sigma_y) ** 2))

    return gaussian




def update_radar_map_with_gaussian(radar_2d_map, r_in, r_in_2d, std_var):

    rows, cols = radar_2d_map.shape

    radar_2d_map_pro = np.zeros((rows, cols))




    for i in range(len(r_in_2d)):
        #print(i)
        u, v = int(np.floor(r_in_2d[i, 0])), int(np.floor(r_in_2d[i, 1]))
        #print(u,v)
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


def get_asytx_dir(frame):
    base = 'D:/Astyx dataset/dataset_astyx_hires2019/'
    # base dir depended on your own settings.
    cam_base_dir = base + 'camera_front/'
    radar_base_dir = base + 'radar_6455/'
    calib_base_dir = base + 'calibration/'
    object_base_dir = base +'groundtruth_obj3d/'
    lidar_base_dir = base+'lidar_vlp16/'
    radar_file = radar_base_dir + str(frame).zfill(6) + '.txt'
    lidar_file = lidar_base_dir + str(frame).zfill(6) + '.txt'
    cam_file = cam_base_dir + str(frame).zfill(6) + '.jpg'
    calib_file = calib_base_dir + str(frame).zfill(6) + '.json'
    object_file = object_base_dir + str(frame).zfill(6) + '.json'
    return radar_file,lidar_file,cam_file,calib_file,object_file


def get_transform_matrix(calib_file, sensor_uid_A, sensor_uid_B):
    #'radar_6455',"lidar_vlp16",'camera_front'
    with open(calib_file, 'r') as f:
        calibration_data = json.load(f)

    # 查找传感器A和传感器B的标定数据
    T_A_to_ref = None
    T_B_to_ref = None

    for sensor in calibration_data['sensors']:
        if sensor['sensor_uid'] == sensor_uid_A:
            T_A_to_ref = np.array(sensor['calib_data']['T_to_ref_COS'])
        elif sensor['sensor_uid'] == sensor_uid_B:
            T_B_to_ref = np.array(sensor['calib_data']['T_to_ref_COS'])

    # 检查是否找到了对应的传感器标定数据
    if T_A_to_ref is None or T_B_to_ref is None:
        return None

    # 计算从传感器A到传感器B的变换矩阵
    T_A_to_B = np.linalg.inv(T_B_to_ref) @ T_A_to_ref

    return T_A_to_B


def read_txt_data(file_path):
    #radar:x,y,z,v,r,mag
    #lidar:X Y Z Reflectivity LaserID Timestamp
    txt_data = np.genfromtxt(file_path, delimiter=' ', skip_header=1)
    txt_data = txt_data[~np.isnan(txt_data).any(axis=1)]

    return txt_data

def process_frame(frame):
    if os.path.exists(f'D:/Astyx dataset/dataset_astyx_hires2019/pmap_image/{frame}.jpg'):
        return

    print(f'Processing frame {frame}')
    radar_file,lidar_file,cam_file,calib_file,object_file = get_asytx_dir(frame)
    K = get_camera_intrinsics(calib_file)


    image = cv2.imread(cam_file)
    coor = [0,0,2048,618]

    r_in = np.load(f'D:/Astyx dataset/dataset_astyx_hires2019/r_in/{frame}.npy')
    r_in_2d = np.load(f'D:/Astyx dataset/dataset_astyx_hires2019/r_in_2d/{frame}.npy')
    radar_2d_map = np.zeros((618, 2048))

    standard_variance = [100, 100]
    radar_2d_map_pro = update_radar_map_with_gaussian(radar_2d_map, r_in, r_in_2d, standard_variance)

    np.save(f'D:/Astyx dataset/dataset_astyx_hires2019/radar_2d_map_pro/{frame}.npy',
            radar_2d_map_pro)

    normalized_array = cv2.normalize(radar_2d_map_pro, None, 0, 255, cv2.NORM_MINMAX)
    gray_image_pro = normalized_array.astype(np.uint8)
    cv2.imwrite(f'D:/Astyx dataset/dataset_astyx_hires2019/pmap_image/{frame}.jpg',
                gray_image_pro)


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


def get_camera_intrinsics(calibration_file):
    # 读取标定文件内容
    with open(calibration_file, 'r') as f:
        calibration_data = json.load(f)

    # 遍历传感器列表，找到对应的相机
    for sensor in calibration_data['sensors']:
        if sensor['sensor_uid'] == 'camera_front':
            if 'K' in sensor['calib_data']:
                # 返回相机的内参矩阵
                return np.array(sensor['calib_data']['K'])

    return None

def estimate_radar_velocity_ransac(pcd_static, min_samples=50, residual_threshold= 0.2, max_trials=3000):
    #pcd_static = pcd_static[pcd_static[:,3]<0]
    positions = pcd_static[:, :3]  # 获取x, y, z坐标
    vr = pcd_static[:, 3]  # 获取v_r（径向速度）

    # 归一化位置坐标
    norms = np.linalg.norm(positions, axis=1)
    norms[norms < 1e-6] = 1  # 避免除以零
    normalized_positions = -positions / norms[:, np.newaxis]

    # 使用最小二乘法进行线性回归
    model = LinearRegression(fit_intercept=False)  # 不使用截距项
    model.fit(normalized_positions, vr)

    # 获取估计的雷达速度
    estimated_radar_velocity = model.coef_

    return estimated_radar_velocity
    # positions = pcd_static[:, :3]  # 获取x, y, z
    # vr = pcd_static[:, 3]  # 获取v_r
    # # 归一化位置坐标
    # norms = np.linalg.norm(positions, axis=1)
    # norms[norms < 1e-6] = 1  # 避免除以零
    # normalized_positions = -positions / norms[:, np.newaxis]
    #
    # ransac = RANSACRegressor(min_samples=min_samples, residual_threshold=residual_threshold, max_trials=max_trials)
    #
    # # RANSAC拟合
    # ransac.fit(normalized_positions, vr)
    #
    # # 获取估计的雷达速度
    # estimated_radar_velocity = ransac.estimator_.coef_
    #
    # return estimated_radar_velocity


def read_json_file(file_name):
    with open(file_name, 'r') as file:
        data = json.load(file)
    return data['objects']


def is_point_in_bbox(center, dimensions, orientation_quat, point):
    point = point[:3]
    rotation = R.from_quat(orientation_quat)
    local_point = rotation.inv().apply(np.array(point) - np.array(center))

    # 获取边界框的半边尺寸
    half_dimensions = np.array(dimensions) / 2.0

    # 检查点是否在边界框内
    if (np.abs(local_point) <= half_dimensions).all():
        return True
    else:
        return False



def filter_points_in_bbox(point_cloud, bbox_location, bbox_dimensions, quat1, quat2, quat3, quat4):

    bbox_center = np.array(bbox_location)
    bbox_height, bbox_length, bbox_width = bbox_dimensions
    quat = [quat1, quat2, quat3, quat4]
    rotation = R.from_quat(quat).as_matrix()
    # # Define rotation matrix for yaw (rotation around the z-axis)
    # Rz = np.array([
    #     [np.cos(yaw), -np.sin(yaw), 0],
    #     [np.sin(yaw), np.cos(yaw), 0],
    #     [0, 0, 1]
    # ])

    # Extract the 3D coordinates from the 4D point cloud
    points_xyz = point_cloud[:, :3]

    # Translate and rotate the point cloud to the bbox coordinate system
    translated_points = points_xyz - bbox_center
    local_points = np.dot(translated_points, rotation.T)

    # Check if the local points are within the bounds of the bbox
    mask = (
        (local_points[:, 0] >= -bbox_length / 2) & (local_points[:, 0] <= bbox_length / 2) &
        (local_points[:, 1] >= -bbox_width / 2) & (local_points[:, 1] <= bbox_width / 2) &
        (local_points[:, 2] >= -bbox_height / 2) & (local_points[:, 2] <= bbox_height / 2)
    )

    return point_cloud[mask], mask



def classify_points(radar_pcd, data):
    dynamic_points = []
    static_points = []

    for point in radar_pcd:
        is_dynamic = False
        for obj in data:
            center = obj['center3d']
            dimensions = obj['dimension3d']
            orientation_quat = obj['orientation_quat']

            if is_point_in_bbox(center, dimensions, orientation_quat, point):
                dynamic_points.append(point)
                is_dynamic = True
                break

        if not is_dynamic:
            static_points.append(point)

    return np.array(dynamic_points), np.array(static_points)

def draw_with_velo(r_in_2d, velocity_list, image, color, frame):
    # 定义圆的固定半径
    radius = 5  # 这里将所有点的大小设为固定值
    thickness = -1  # 圆的填充

    # 归一化速度到 [0, 1] 范围
    vel_normalized = np.clip(velocity_list / np.max(velocity_list), 0, 1)

    for i, point in enumerate(r_in_2d):
        center = (int(point[0]), int(point[1]))

        # 通过速度的归一化值调整颜色的深浅，越大颜色越深
        color_intensity = int(vel_normalized[i] * 255)  # 将速度映射到 [0, 255]
        adjusted_color = (int(color[0] * color_intensity / 255),
                          int(color[1] * color_intensity / 255),
                          int(color[2] * color_intensity / 255))

        # 画出圆
        cv2.circle(image, center, radius, adjusted_color, thickness)

    # 保存图片
    cv2.imwrite(f'D:/Astyx dataset/dataset_astyx_hires2019/projection_velocity/{frame}.jpg', image)


if __name__ == '__main__':
    radar_file,lidar_file,cam_file,calib_file,object_file = get_asytx_dir(1)
    #T = get_transform_matrix(calib_file, 'radar_6455','lidar_vlp16')
    radar_pcd = read_txt_data(radar_file)
    lidar_pcd = read_txt_data(lidar_file)
    K = get_camera_intrinsics(calib_file)
    image = cv2.imread(cam_file)
    # print(image.shape)
    T = get_transform_matrix(calib_file, 'lidar_vlp16','camera_front')
    data = read_json_file(object_file)
    # print(data)
    # yaw_deg, pitch_deg = utils1.pixel_to_radar_angles(0,0,K,T)
    # print(yaw_deg, pitch_deg)



    radar_geo = radar_pcd[:,0:3]
    fov_horizontal = np.radians(110)  # 水平 FoV 转为弧度
    fov_vertical = np.radians(10)  # 垂直 FoV 转为弧度
    angle_threshold = np.radians(1.5)
    # 计算每个点的水平和垂直角度
    x, y, z = radar_geo[:, 0], radar_geo[:, 1], radar_geo[:, 2]
    from sklearn.cluster import DBSCAN
    # horizontal_angles = np.arctan2(y, x)
    # vertical_angles = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))
    #
    # # 过滤掉 FoV 外的点
    # # 水平角度范围 [-fov_h / 2, fov_h / 2]，垂直角度范围 [-fov_v / 2, fov_v / 2]
    # valid_indices = np.where(
    #     (horizontal_angles >= -fov_horizontal / 2) & (horizontal_angles <= fov_horizontal / 2) &
    #     (vertical_angles >= -fov_vertical / 2) & (vertical_angles <= fov_vertical / 2)
    # )[0]
    #
    # # 只保留在 FoV 内的点
    # radar_geo_filtered = radar_geo[valid_indices]
    # horizontal_angles_filtered = horizontal_angles[valid_indices]
    # vertical_angles_filtered = vertical_angles[valid_indices]
    #
    # # 使用 KDTree 进行快速邻域查询
    # tree = KDTree(np.vstack((horizontal_angles_filtered, vertical_angles_filtered)).T)
    #
    # # 计算每个点的局部密度
    # local_densities = []
    # for i, angle in enumerate(np.vstack((horizontal_angles_filtered, vertical_angles_filtered)).T):
    #     # 查询在 angle_threshold 角度内的邻域点数量
    #     neighbors = tree.query_ball_point(angle, angle_threshold)
    #     local_densities.append(len(neighbors))
    #
    # # 将局部密度存储为数组
    # local_densities = np.array(local_densities)
    #
    # # 检测密度变化明显的边界位置
    # # 我们可以找到局部密度小于其周围平均值的点，并认为它是一个边界点
    # density_threshold = np.mean(local_densities) - np.std(local_densities)
    # boundary_indices = np.where(local_densities < density_threshold)[0]
    #
    # # 在水平和垂直方向上分别计算边界数量
    # horizontal_boundaries = len(set(horizontal_angles_filtered[boundary_indices]))
    # vertical_boundaries = len(set(vertical_angles_filtered[boundary_indices]))
    #
    # # 估算角分辨率
    # horizontal_resolution = fov_horizontal / horizontal_boundaries if horizontal_boundaries > 1 else fov_horizontal
    # vertical_resolution = fov_vertical / vertical_boundaries if vertical_boundaries > 1 else fov_vertical
    #
    # # 输出估算的角分辨率
    # print("Estimated Horizontal Resolution (radians):", horizontal_resolution)
    # print("Estimated Vertical Resolution (radians):", vertical_resolution)
    # print("Estimated Horizontal Resolution (degrees):", np.degrees(horizontal_resolution))
    # print("Estimated Vertical Resolution (degrees):", np.degrees(vertical_resolution))


    min_points_in_cluster = 5  # 最小的聚类点数，保证足够的局部密度
    horizontal_angles = np.arctan2(y, x)  # 水平角度，arctan2(y, x) 给出 x-y 平面的角度
    vertical_angles = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))  # 竖直角度

    # 过滤掉 FoV 外的点
    valid_horizontal_indices = np.where(
        (horizontal_angles >= -fov_horizontal / 2) & (horizontal_angles <= fov_horizontal / 2)
    )[0]
    valid_vertical_indices = np.where(
        (vertical_angles >= -fov_vertical / 2) & (vertical_angles <= fov_vertical / 2)
    )[0]

    # 只保留在 FoV 内的点
    valid_indices = np.intersect1d(valid_horizontal_indices, valid_vertical_indices)
    radar_geo_filtered = radar_geo[valid_indices]
    horizontal_angles_filtered = horizontal_angles[valid_indices]
    vertical_angles_filtered = vertical_angles[valid_indices]

    # 使用DBSCAN进行聚类
    # 水平角度聚类
    dbscan_horizontal = DBSCAN(eps=0.005, min_samples=min_points_in_cluster)
    labels_horizontal = dbscan_horizontal.fit_predict(horizontal_angles_filtered.reshape(-1, 1))

    # 竖直角度聚类
    dbscan_vertical = DBSCAN(eps=0.005, min_samples=min_points_in_cluster)
    labels_vertical = dbscan_vertical.fit_predict(vertical_angles_filtered.reshape(-1, 1))


    # 计算水平和竖直方向的角分辨率
    def calculate_resolution(labels, angles_filtered):
        # 找到所有有效聚类（排除噪声点，-1代表噪声点）
        valid_clusters = np.unique(labels[labels != -1])
        cluster_sizes = []
        cluster_angles = []

        for cluster in valid_clusters:
            # 获取该聚类的点
            cluster_points = angles_filtered[labels == cluster]

            # 计算该聚类的角度范围
            angle_range = np.max(cluster_points) - np.min(cluster_points)

            # 记录该聚类的角度范围
            cluster_sizes.append(len(cluster_points))
            cluster_angles.append(angle_range)

        # 计算估算的角分辨率
        if len(valid_clusters) > 0:
            avg_angle_resolution = np.mean(cluster_angles) / len(valid_clusters)
        else:
            avg_angle_resolution = fov_vertical  # 如果没有有效聚类，使用整个FoV范围作为角分辨率

        return avg_angle_resolution


    # 水平方向角分辨率
    horizontal_resolution = calculate_resolution(labels_horizontal, horizontal_angles_filtered)
    # 竖直方向角分辨率
    vertical_resolution = calculate_resolution(labels_vertical, vertical_angles_filtered)

    # 输出结果
    print("Estimated Horizontal Resolution (radians):", horizontal_resolution)
    print("Estimated Vertical Resolution (radians):", vertical_resolution)
    print("Estimated Horizontal Resolution (degrees):", np.degrees(horizontal_resolution))
    print("Estimated Vertical Resolution (degrees):", np.degrees(vertical_resolution))
    # lidar_yaw_min, lidar_yaw_max = 1000, -1000
    # lidar_pitch_min, lidar_pitch_max = 1000, -1000
    # for frame in range(0,546):
    #     print('frame',frame)
    #     radar_file,lidar_file,cam_file,calib_file,object_file = get_asytx_dir(frame)
    #     if not (os.path.exists(radar_file) and os.path.exists(lidar_file) and os.path.exists(cam_file) and os.path.exists(calib_file)):
    #         print('file missing at frame ',frame)
    #         continue
    #
    #     lidar_pcd = read_txt_data(lidar_file)
    #     lidar_pcd = lidar_pcd[lidar_pcd[:, 0] > 0][:, 0:3]
    #     max_height, min_height, yaw_range, pitch_range = calculate_height_and_angles(lidar_pcd)
    #
    #     #print(f"最大高度: {max_height:.2f}, 最小高度: {min_height:.2f}")
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


#----------------------------------------------projection---------------------------------------------------------------
    for frame in range(0,546):
        print(f'frame:{frame}')
        radar_file, lidar_file, cam_file, calib_file, object_file = get_asytx_dir(frame)
        coor = [0,0,2018,618]
        T_lidar = get_transform_matrix(calib_file,'lidar_vlp16','camera_front')
        T_radar = get_transform_matrix(calib_file,'radar_6455','camera_front')
        lidar_pcd = read_txt_data(lidar_file)
        lidar_pcd = lidar_pcd[lidar_pcd[:, 0] > 0][:, 0:3]
        lidar_pcd = lidar_pcd[lidar_pcd[:, 0] <= 100]
        radar_pcd = read_txt_data(radar_file)
        K = get_camera_intrinsics(calib_file)
        image = cv2.imread(cam_file)
        # l_in, l_in_2d = utils1.radar_in_image(coor, lidar_pcd, T_lidar,K)
        # print(f'len(lidar_pcd):{len(lidar_pcd)}')
        # print(f'len(l_in):{len(l_in)}')
        # np.save(f'D:/Astyx dataset/dataset_astyx_hires2019/l_in_2d/{frame}.npy',l_in_2d)
        # np.save(f'D:/Astyx dataset/dataset_astyx_hires2019/l_in/{frame}.npy',l_in)
        r_in, r_in_2d = utils1.radar_in_image(coor, radar_pcd, T_radar,K)
        np.save(f'D:/Astyx dataset/dataset_astyx_hires2019/r_in_2d/{frame}.npy',r_in_2d)
        np.save(f'D:/Astyx dataset/dataset_astyx_hires2019/r_in/{frame}.npy',r_in)
        velocity_list = r_in[:,3]
        print(f'len(radar_pcd):{len(radar_pcd)}')
        print(f'len(r_in):{len(r_in)}')
        draw_with_velo(r_in_2d, velocity_list, image, [0,0,255],frame)

# #--------------------------------------------make dataset---------------------------------------------------------------
#     column_names = ['Frame', 'PointNum', 'Velocity']
#     new_df = pd.DataFrame(columns=column_names)
#     for frame in range(0,546):
#         print(f'frame:{frame}')
#         radar_file, lidar_file, cam_file, calib_file, object_file = get_asytx_dir(frame)
#         data = read_json_file(object_file)
#         #radar_pcd = read_txt_data(radar_file)
#         radar_pcd = np.load(f'D:/Astyx dataset/dataset_astyx_hires2019/r_in/{frame}.npy')
#         print(len(radar_pcd))
#         dynamic_pcd, static_pcd = classify_points(radar_pcd,data)
#         print(f'len(dynamic):{len(dynamic_pcd)},len(static):{len(static_pcd)}')
#         ego_velo = estimate_radar_velocity_ransac(static_pcd)
#         print(ego_velo,np.linalg.norm(ego_velo))
#         r_in = np.load(f'D:/Astyx dataset/dataset_astyx_hires2019/r_in/{frame}.npy')
#         num_points = len(r_in)
#         print('pointnum',num_points)
#         new_data = pd.DataFrame({'Frame': [frame], 'PointNum': [num_points], 'Velocity': [np.linalg.norm(ego_velo)]})
#         new_df = pd.concat([new_df, new_data], ignore_index=True)
#     new_df.to_csv('D:/Astyx dataset/dataset_astyx_hires2019/prob_dataset.csv', index=False)


#-------------------------------------------------------edge/corner detection-------------------------------------------
    # for frame in range(0,546):
    #     radar_file, lidar_file, cam_file, calib_file, object_file = get_asytx_dir(frame)
    #     cam_dir = cam_file
    #     des_dir = 'D:/Astyx dataset/dataset_astyx_hires2019/edge_image/'+str(int(frame))+".jpg"
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
    #
    #     cv2.imwrite(des_dir,combined)
    #     print(des_dir)
    #     print(edges.shape)



    #----------------------------------------------distribution map-----------------------------------------------------
    # for frame in range(0,546):
    #     print(frame)
    #     radar_file, lidar_file, cam_file, calib_file, object_file = get_asytx_dir(frame)
    #     image = cv2.imread(cam_file)
    #     r_in_2d = np.load(f'D:/Astyx dataset/dataset_astyx_hires2019/r_in_2d/{frame}.npy')
    #     new_image = utils.visualize_with_image_color(image, r_in_2d, [0, 0, 255])
    #     cv2.imwrite(f'D:/Astyx dataset/dataset_astyx_hires2019/projection_radar/{frame}.jpg',new_image)

    # frames = range(0, 546)
    #
    # # 使用joblib并行处理
    # Parallel(n_jobs=8)(delayed(process_frame)(frame) for frame in frames)




    #------------------------------------------------split dataset-----------------------------------------------------
    # random_seed = 2024
    # np.random.seed(random_seed)
    #
    # # 读取数据集
    # df = pd.read_csv('D:/Astyx dataset/dataset_astyx_hires2019/prob_dataset.csv')
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
    # train_df.to_csv('D:/Astyx dataset/dataset_astyx_hires2019/train_dataset.csv',
    #                 index=False)
    # test_df.to_csv('D:/Astyx dataset/dataset_astyx_hires2019/test_dataset.csv', index=False)
    #
    # # 检查PointNum列的最小值和最大值
    # print(np.min(train_df['PointNum'].values), np.max(train_df['PointNum'].values))




#-------------------------------------------------RCS regression--------------------------------------------------------
    # train_df = pd.read_csv("D:/Astyx dataset/dataset_astyx_hires2019/train_dataset.csv").values
    # test_df = pd.read_csv("D:/Astyx dataset/dataset_astyx_hires2019/test_dataset.csv").values
    # column_names = ['Frame', 'Index', 'x', 'y', 'z', 'u', 'v', 'v_r', 'rcs']
    # new_df_train = pd.DataFrame(columns=column_names)
    # for i in range(len(train_df)):
    #     frame, pointnum, ego_velocity = train_df[i]
    #     frame, pointnum = int(frame), int(pointnum)
    #     print('frame', frame)
    #     radar_file, lidar_file, cam_file, calib_file, object_file = get_asytx_dir(frame)
    #     K = get_camera_intrinsics(calib_file)
    #     if K is None:
    #         print(f'frame {frame} missing intrinsic matrix.')
    #         continue
    #     T_lidar2radar = get_transform_matrix(calib_file, 'lidar_vlp16', 'radar_6455')
    #
    #     r_in = np.load(f'D:/Astyx dataset/dataset_astyx_hires2019/r_in/{frame}.npy')
    #     r_in_2d = np.load(f'D:/Astyx dataset/dataset_astyx_hires2019/r_in_2d/{frame}.npy')
    #     mask = radar_in_lidarfov(r_in,np.linalg.inv(T_lidar2radar))
    #     l_in = np.load(
    #         f'D:/Astyx dataset/dataset_astyx_hires2019/l_in/{frame}.npy')
    #     l_in_radar_coor = utils.trans_point_coor(l_in, T_lidar2radar)
    #     np.save(f'D:/Astyx dataset/dataset_astyx_hires2019/l_in_radar_coor/{frame}.npy',
    #             l_in_radar_coor)
    #     kdtree = KDTree(l_in_radar_coor)
    #     n = 100
    #     sampled_points, sampled_points_2d = sample_points_from_point_cloud(r_in, r_in_2d, n)
    #     #idx = 0
    #     for j in range(0, min(n,len(r_in))):
    #         x, y, z, v_r, rcs = sampled_points[j, 0:5]
    #         u, v = sampled_points_2d[j, :]
    #
    #         r = 1
    #
    #         pitch = np.degrees(np.arctan(z / np.linalg.norm([x, y], 2)))
    #
    #         local_lidar_index = kdtree.query_ball_point([x, y, z], r)
    #         # if len(local_lidar_index) == 0:
    #         #     continue
    #         new_data = pd.DataFrame(
    #             {'Frame': [frame], 'Index': [j], 'x': [x], 'y': [y], 'z': [z], 'u': [u], 'v': [v], 'v_r': [v_r],
    #              'rcs': [rcs]})
    #         new_df_train = pd.concat([new_df_train, new_data], ignore_index=True)
    #
    #
    #         local_lidar_radarcoor = l_in_radar_coor[local_lidar_index]
    #
    #         np.save(f'D:/Astyx dataset/dataset_astyx_hires2019/local_points/{frame}_{j}.npy',
    #                 local_lidar_radarcoor)
    #         dst = f'D:/Astyx dataset/dataset_astyx_hires2019/range_image/{frame}_{j}.jpg'
    #         virtual_point = get_new_point(x, y, z)
    #         fov_down = -15
    #         fov_up = 15
    #         for k in range(len(local_lidar_radarcoor)):
    #             local_lidar_radarcoor[k, 0] -= virtual_point[0]
    #             local_lidar_radarcoor[k, 1] -= virtual_point[1]
    #             local_lidar_radarcoor[k, 2] -= virtual_point[2]
    #             x1, y1, z1 = local_lidar_radarcoor[k]
    #             pitch = np.degrees(np.arctan(z1 / np.linalg.norm([x1, y1], 2)))
    #             if pitch > fov_up:
    #                 fov_up = pitch
    #             if pitch < fov_down:
    #                 fov_down = pitch
    #
    #         proj_H, proj_W, = 32, 128
    #         gen_range_image_rcs(local_lidar_radarcoor, fov_up, fov_down, proj_H, proj_W,
    #                             dst)
    #         # idx += 1
    # new_df_train.to_csv('D:/Astyx dataset/dataset_astyx_hires2019/RCS_dataset_train.csv')



    #-----------------------------------generate dynamic objects dataframe----------------------------------------------
    # column_names = ['Frame', 'Track_ID', 'Class', 'Rotation1', 'Rotation2', 'Rotation3', 'Rotation4', 'Location_x', 'Location_y', 'Location_z',
    #                 'Dimension_x', 'Dimension_y', 'Dimension_z']
    # df = pd.DataFrame(columns=column_names)
    # for frame in range(0, 546):
    #     radar_file, lidar_file, cam_file, calib_file, object_file = get_asytx_dir(frame)
    #     K = get_camera_intrinsics(calib_file)
    #     if K is None:
    #         print(f'frame {frame} missing intrinsic matrix.')
    #         continue
    #     data = read_json_file(object_file)
    #     for j in range(len(data)):
    #         class_ = data[j]['classname']
    #         location_x, location_y, location_z = data[j]['center3d']
    #         dimension_x, dimension_y, dimension_z = data[j]['dimension3d']
    #         quat_1, quat_2, quat_3, quat_4 = data[j]['orientation_quat']
    #         new_data = pd.DataFrame(
    #                     {'Frame': [frame], 'Track_ID': [j], 'Class': [class_], 'Rotation1': [quat_1], 'Rotation2': [quat_2],
    #                      'Rotation3': [quat_3], 'Rotation4': [quat_4],
    #                      'Location_x': [location_x], 'Location_y': [location_y], 'Location_z': [location_z],
    #                      'Dimension_x': [dimension_x], 'Dimension_y': [dimension_y], 'Dimension_z': [dimension_z]})
    #         df = pd.concat([df, new_data], ignore_index=True)
    #
    # df.to_csv('D:/Astyx dataset/dataset_astyx_hires2019/dynamic_objects_total.csv')






    #-------------------------------generate dynamic objects subset-----------------------------------------------------
    # RCS_test = pd.read_csv('D:/Astyx dataset/dataset_astyx_hires2019/RCS_dataset_test.csv')
    # dynamic_object = pd.read_csv('D:/Astyx dataset/dataset_astyx_hires2019/dynamic_objects_total.csv')
    # matched_objects = []
    #
    #
    # def find_matching_objects(RCS_data, dynamic_object):
    #     grouped_RCS_data = RCS_data.groupby('Frame')
    #     grouped_dynamic_object = dynamic_object.groupby('Frame')
    #
    #     total_frames = len(grouped_RCS_data)
    #     checkpoint = total_frames // 10  # 每处理10%帧作为一个进度点
    #
    #     for idx, (frame, frame_RCS_points) in enumerate(grouped_RCS_data):
    #         if frame in grouped_dynamic_object.groups:  # 仅处理相同帧的数据
    #             frame_dynamic_objects = grouped_dynamic_object.get_group(frame)
    #
    #             for _, point in frame_RCS_points.iterrows():
    #                 point_cloud = np.array([[point['x'], point['y'], point['z']]])  # 获取点的坐标
    #                 point_classified = False
    #
    #                 # 遍历同一帧的 dynamic object 的 bounding box
    #                 for _, obj in frame_dynamic_objects.iterrows():
    #                     bbox_location = [obj['Location_x'], obj['Location_y'], obj['Location_z']]
    #                     bbox_dimensions = [obj['Dimension_x'], obj['Dimension_y'], obj['Dimension_z']]
    #                     quat1, quat2, quat3, quat4 = obj['Rotation1'], obj['Rotation2'], obj['Rotation3'], obj[
    #                         'Rotation4']
    #
    #                     # 判断点是否在 bounding box 内
    #                     if len(filter_points_in_bbox(point_cloud, bbox_location, bbox_dimensions, quat1, quat2, quat3,
    #                                                  quat4)[0]) > 0:
    #                         # 匹配成功，保存物体信息
    #                         matched_objects.append(obj)
    #                         point_classified = True
    #                         break  # 一旦点已分类，无需继续判断
    #
    #         # 输出进度：每处理 10% 的帧输出一次进度
    #         if (idx + 1) % checkpoint == 0:
    #             progress = (idx + 1) / total_frames * 100
    #             print(f"Progress: {progress:.1f}%")
    #
    #     print("Matching completed!")
    #
    #
    # print('Matching objects for RCS_test...')
    # find_matching_objects(RCS_test, dynamic_object)
    #
    # matched_objects_df = pd.DataFrame(matched_objects)
    #
    # matched_objects_df = matched_objects_df.drop_duplicates()
    #
    # output_csv_path = 'D:/Astyx dataset/dataset_astyx_hires2019/matched_dynamic_objects_subset.csv'
    # matched_objects_df.to_csv(output_csv_path, index=False)
    #
    # print(f"Subset of matched dynamic objects saved to {output_csv_path}")


    # df = pd.read_csv(
    #     'D:/Astyx dataset/dataset_astyx_hires2019/matched_dynamic_objects_subset.csv').iloc[:, 1:]
    # df['valid'] = 0  # 初始化 valid 列
    # # 遍历数据并计算 valid 列
    # for i in range(len(df)):
    #     Frame, Track_ID, Class, quat1, quat2, quat3, quat4, Location_x, Location_y, Location_z, Dimension_x, Dimension_y, Dimension_z, _ = \
    #     df.iloc[i]
    #     bbox_location = [Location_x, Location_y, Location_z]
    #     bbox_dimensions = [Dimension_x, Dimension_y, Dimension_z]
    #     r_in = np.load(f'D:/Astyx dataset/dataset_astyx_hires2019/r_in/{Frame}.npy')
    #     r_in_2d = np.load(f'D:/Astyx dataset/dataset_astyx_hires2019/r_in_2d/{Frame}.npy')
    #
    #     r_in_local, mask = filter_points_in_bbox(r_in, bbox_location, bbox_dimensions, quat1, quat2, quat3, quat4)
    #
    #     # 计算 valid 值
    #     if len(r_in_local) < 3:
    #         df.at[i, 'valid'] = 0
    #     else:
    #         df.at[i, 'valid'] = 1
    #         # 保存 r_in_local
    #         np.save(
    #             f'D:/Astyx dataset/dataset_astyx_hires2019/rcs_dis_test/object_local_points/{Frame}_{Track_ID}.npy',
    #             r_in_local)
    #         r_in_2d_local = r_in_2d[mask]
    #
    # # 保存更新后的 DataFrame
    # df.to_csv('D:/Astyx dataset/dataset_astyx_hires2019/matched_dynamic_objects_subset_new.csv', index=False)


    #---------------------------------RCS KS test-----------------------------------------------------------------------
    # df = pd.read_csv(
    #     'D:/Astyx dataset/dataset_astyx_hires2019/matched_dynamic_objects_subset_new.csv')
    # arr = df[df['valid'] == 1].values
    # df_rcs_valid = pd.read_csv(
    #     'D:/Astyx dataset/dataset_astyx_hires2019/ablation/RCS_validation_results_ablation_para_250_0.5.csv')
    # df_rcs_points = pd.read_csv('D:/Astyx dataset/dataset_astyx_hires2019/RCS_dataset_test.csv')
    #
    # success_cnt = 0
    # fail_cnt = 0
    # total_cnt = 0
    #
    # for i in range(0,len(arr)):
    #     Frame, Track_ID, Class, quat1, quat2, quat3, quat4, Location_x, Location_y, Location_z, Dimension_x, Dimension_y, Dimension_z, _ = \
    #         arr[i]
    #
    #     # Get indices
    #     indices = df_rcs_valid[df_rcs_valid['Frame'] == Frame]['Index'].values
    #
    #     # Get point cloud data
    #     point_cloud = df_rcs_points[df_rcs_points['Frame'] == Frame][['x', 'y', 'z', 'rcs', 'v_r']].values
    #
    #     # Set bounding box
    #     bbox_location = [Location_x, Location_y, Location_z]
    #     bbox_dimensions = [Dimension_x, Dimension_y, Dimension_z]
    #     #print(Class, bbox_location, bbox_dimensions)
    #     # Filter point cloud
    #     test_p_in, mask = filter_points_in_bbox(point_cloud, bbox_location, bbox_dimensions, quat1, quat2, quat3, quat4)
    #
    #     # Load local points
    #     local_points = np.load(
    #         f'D:/Astyx dataset/dataset_astyx_hires2019/rcs_dis_test/object_local_points/{Frame}_{Track_ID}.npy')
    #     #print(local_points)
    #     # Get true RCS values
    #     rcs_true = local_points[:, -1]
    #
    #     # Get predicted RCS values for test points
    #     rcs_pred_full = df_rcs_valid[(df_rcs_valid['Frame'] == Frame) & (df_rcs_valid['Index'].isin(indices))][
    #         'Predictions'].values
    #
    #     # Filter rcs_pred_full to keep only the predictions corresponding to test_p_in
    #     rcs_pred = rcs_pred_full[mask] if len(mask) == len(rcs_pred_full) else []
    #     #print(rcs_pred)
    #
    #     # Check if predicted values come from the same distribution as true values
    #     if len(rcs_true) > 0 and len(rcs_pred) > 0:
    #         statistic, p_value = ks_2samp(rcs_true, rcs_pred)
    #
    #         # Determine if they come from the same distribution based on p-value
    #         if p_value > 0.05:
    #             success_cnt += 1
    #             print(
    #                 f'Frame {Frame}, Track ID {Track_ID}: RCS predictions and true values come from the same distribution (p={p_value:.4f})')
    #             # print(len(rcs_true), len(rcs_pred))
    #             print('True RCS:', rcs_true)
    #             print('Predicted RCS:', rcs_pred)
    #         else:
    #             fail_cnt += 1
    #             print(
    #                 f'Frame {Frame}, Track ID {Track_ID}: RCS predictions and true values do not come from the same distribution (p={p_value:.4f})')
    #             print('True RCS:', rcs_true)
    #             print('Predicted RCS:', rcs_pred)
    #
    #         total_cnt += 1
    #
    # print('Success rate:', success_cnt / total_cnt)
    # print('Success, Fail, Total:', success_cnt, fail_cnt, total_cnt)


# ------------------------------------------make RCS regression dataset wo vp-------------------------------------------

    # df = pd.read_csv('D:/Astyx dataset/dataset_astyx_hires2019/RCS_dataset_test.csv').values[:,1:]
    # for i in range(len(df)):
    #     # frame, pointnum, ego_velocity = train_df[i]
    #     # frame, pointnum = int(frame), int(pointnum)
    #     # print('frame', frame)
    #     # radar_file, lidar_file, cam_file, calib_file, object_file = get_asytx_dir(frame)
    #     # K = get_camera_intrinsics(calib_file)
    #     # if K is None:
    #     #     print(f'frame {frame} missing intrinsic matrix.')
    #     #     continue
    #     # T_lidar2radar = get_transform_matrix(calib_file, 'lidar_vlp16', 'radar_6455')
    #     #
    #     # r_in = np.load(f'D:/Astyx dataset/dataset_astyx_hires2019/r_in/{frame}.npy')
    #     # r_in_2d = np.load(f'D:/Astyx dataset/dataset_astyx_hires2019/r_in_2d/{frame}.npy')
    #     # mask = radar_in_lidarfov(r_in,np.linalg.inv(T_lidar2radar))
    #     # l_in = np.load(
    #     #     f'D:/Astyx dataset/dataset_astyx_hires2019/l_in/{frame}.npy')
    #     # l_in_radar_coor = utils.trans_point_coor(l_in, T_lidar2radar)
    #     frame, index,x,y,z,u,v,v_r,rcs = df[i]
    #     l_in_radar_coor = np.load(f'D:/Astyx dataset/dataset_astyx_hires2019/l_in_radar_coor/{int(frame)}.npy')
    #     kdtree = KDTree(l_in_radar_coor)
    #     #n = 100
    #     #sampled_points, sampled_points_2d = sample_points_from_point_cloud(r_in, r_in_2d, n)
    #
    #     # for j in range(0, min(n,len(r_in))):
    #     #     x, y, z, v_r, rcs = sampled_points[j, 0:5]
    #     #     u, v = sampled_points_2d[j, :]
    #
    #     r = 1
    #
    #     pitch = np.degrees(np.arctan(z / np.linalg.norm([x, y], 2)))
    #
    #     local_lidar_index = kdtree.query_ball_point([x, y, z], r)
    #
    #
    #     local_lidar_radarcoor = l_in_radar_coor[local_lidar_index]
    #     np.save(f'D:/Astyx dataset/dataset_astyx_hires2019/local_points/{int(frame)}_{int(index)}.npy', local_lidar_radarcoor)
    #     #local_lidar_radarcoor = np.load(f'D:/Astyx dataset/dataset_astyx_hires2019/local_points/{int(frame)}_{int(index)}.npy')
    #     dst = f'D:/Astyx dataset/dataset_astyx_hires2019/ablation/range_image/{int(frame)}_{int(index)}.jpg'
    #     #virtual_point = get_new_point(x, y, z)
    #     fov_down = -15
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

        # df = pd.read_csv('D:/Astyx dataset/dataset_astyx_hires2019/RCS_dataset_test.csv').values[
        #      :, 1:]
        # for i in range(len(df)):
        #     frame, index, x, y, z, u, v, v_r, rcs = df[i]
        #     l_in_radar_coor = np.load(
        #         f'D:/Astyx dataset/dataset_astyx_hires2019/l_in_radar_coor/{int(frame)}.npy')
        #     kdtree = KDTree(l_in_radar_coor)
        #     r = 4
        #     if not os.path.exists(f'D:/Astyx dataset/dataset_astyx_hires2019/ablation/range_image_{r}/'):
        #         os.mkdir(f'D:/Astyx dataset/dataset_astyx_hires2019/ablation/range_image_{r}/')
        #     pitch = np.degrees(np.arctan(z / np.linalg.norm([x, y], 2)))
        #
        #     local_lidar_index = kdtree.query_ball_point([x, y, z], r)
        #
        #     local_lidar_radarcoor = l_in_radar_coor[local_lidar_index]
        #
        #     dst = f'D:/Astyx dataset/dataset_astyx_hires2019/ablation/range_image_{r}/{int(frame)}_{int(index)}.jpg'
        #     virtual_point = get_new_point(x, y, z, r)
        #     fov_down = -15
        #     fov_up = 15
        #     for k in range(len(local_lidar_radarcoor)):
        #         local_lidar_radarcoor[k, 0] -= virtual_point[0]
        #         local_lidar_radarcoor[k, 1] -= virtual_point[1]
        #         local_lidar_radarcoor[k, 2] -= virtual_point[2]
        #         x1, y1, z1 = local_lidar_radarcoor[k]
        #         pitch = np.degrees(np.arctan(z1 / np.linalg.norm([x1, y1], 2)))
        #         if pitch > fov_up:
        #             fov_up = pitch
        #         if pitch < fov_down:
        #             fov_down = pitch
        #
        #     proj_H, proj_W, = 32, 128
        #     gen_range_image_rcs(local_lidar_radarcoor, fov_up, fov_down, proj_H, proj_W,
        #                         dst)


    # for r in [0.5,1,2,3,4]:
    #     def process_data(row):
    #         frame, index, x, y, z, u, v, v_r, rcs = row
    #         l_in_radar_coor = np.load(
    #             f'D:/Astyx dataset/dataset_astyx_hires2019/l_in_radar_coor/{int(frame)}.npy')
    #         kdtree = KDTree(l_in_radar_coor)
    #
    #         if not os.path.exists(f'D:/Astyx dataset/dataset_astyx_hires2019/ablation/newest/range_image_{r}'):
    #             os.mkdir(f'D:/Astyx dataset/dataset_astyx_hires2019/ablation/newest/range_image_{r}')
    #
    #         #pitch = np.degrees(np.arctan(z / np.linalg.norm([x, y], 2)))
    #
    #         local_lidar_index = kdtree.query_ball_point([x, y, z], r)
    #         local_lidar_radarcoor = l_in_radar_coor[local_lidar_index]
    #
    #         dst = f'D:/Astyx dataset/dataset_astyx_hires2019/ablation/newest/range_image_{r}/{int(frame)}_{int(index)}.jpg'
    #         fov_down = -25
    #         fov_up = 15
    #         #virtual_point = get_new_point(x, y, z, r)
    #         original_points = local_lidar_radarcoor.copy()
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
    #         proj_H, proj_W = 32, 128
    #         #gen_range_image_rcs(local_lidar_radarcoor, fov_up, fov_down, proj_H, proj_W, dst)
    #         gen_range_image_rcs_translation_correct(local_lidar_radarcoor,original_points, proj_H, proj_W,  [x,y,z],r,
    #                                         dst)
    #
    #
    #     df = pd.read_csv('D:/Astyx dataset/dataset_astyx_hires2019/RCS_dataset_test.csv').values[:, 1:]
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
    #         executor.map(process_data, df)
    #     df = pd.read_csv('D:/Astyx dataset/dataset_astyx_hires2019/RCS_dataset_train.csv').values[:, 1:]
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
    #         executor.map(process_data, df)
