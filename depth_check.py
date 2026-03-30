import os.path
import random
import cv2
import numpy as np
import math
import pandas as pd
import utils
import open3d as o3d
from scipy.spatial import cKDTree, KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KernelDensity
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR  # Support Vector Regression for regression tasks
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

random.seed(2025)
np.random.seed(2025)


def filter_ground_points_ransac(radar_points, z_threshold=0.5):

    radar_points = np.array(radar_points)

    # 提取x, y作为特征，z作为目标
    X = radar_points[:, :2]  # (N, 2)，即x, y
    y = radar_points[:, 2]  # (N, )，即z

    # 使用RANSAC拟合地面平面
    ransac = RANSACRegressor()
    ransac.fit(X, y)

    # 获取拟合的平面模型
    inlier_mask = ransac.inlier_mask_

    # 使用掩码提取地面点和非地面点
    ground_points = radar_points[inlier_mask]
    non_ground_points = radar_points[~inlier_mask]

    num_ground = len(ground_points)
    num_to_select = int(num_ground * 0.05)
    if num_to_select > 0:
        selected_ground_points = ground_points[np.random.choice(num_ground, num_to_select, replace=False)]
    else:
        selected_ground_points = np.array([])

    combined_points = np.vstack((non_ground_points, selected_ground_points))

    return combined_points, ground_points

def visualize_point_cloud(true_points, estimated_points, bboxes=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 真实点云，用红色显示
    ax.scatter(true_points[:, 0], true_points[:, 1], true_points[:, 2], c='r', label='True Point Cloud', s=1)

    # 估计点云，用蓝色显示
    ax.scatter(estimated_points[:, 0], estimated_points[:, 1], estimated_points[:, 2], c='b', label='Estimated Point Cloud', s=1)

    # 绘制边界框
    if bboxes is not None:
        for bbox in bboxes:
            x_min, x_max, y_min, y_max, z_min, z_max = bbox
            # 计算8个顶点
            corners = np.array([[x_min, y_min, z_min],
                                [x_min, y_min, z_max],
                                [x_min, y_max, z_min],
                                [x_min, y_max, z_max],
                                [x_max, y_min, z_min],
                                [x_max, y_min, z_max],
                                [x_max, y_max, z_min],
                                [x_max, y_max, z_max]])

            # 定义边界框的连接方式 (每一行表示一个边)
            edges = [
                [0, 1], [1, 3], [3, 2], [2, 0],  # 连接底面
                [4, 5], [5, 7], [7, 6], [6, 4],  # 连接顶面
                [0, 4], [1, 5], [2, 6], [3, 7]   # 连接上下面
            ]

            # 绘制每条边
            for edge in edges:
                ax.plot([corners[edge[0], 0], corners[edge[1], 0]],
                        [corners[edge[0], 1], corners[edge[1], 1]],
                        [corners[edge[0], 2], corners[edge[1], 2]], c='g')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.show()

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


def draw(r_in_2d, image, color):

    radius = 10
    thickness = -1

    for point in r_in_2d:
        center = (int(point[0]), int(point[1]))
        cv2.circle(image, center, radius, color, thickness)

    image = image[571:1215,0:1935]
    cv2.imshow('Image with Points', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('C:/Users/PerrySong_nku/Desktop/writing/example_sim_39.jpg', image)

def draw_with_rcs(r_in_2d, rcs_list, image, color):
    # 定义映射的最小和最大半径
    min_radius = 1
    max_radius = 12

    # 将 RCS 映射到半径范围，假设 rcs_list 的范围为 [-50, 50]，可根据数据分布调整
    rcs_normalized = np.clip((rcs_list + 50) / 100, 0, 1)  # 将 RCS 归一化到 [0, 1] 范围
    radii = rcs_normalized * (max_radius - min_radius) + min_radius

    thickness = -1  # 圆的填充

    for i, point in enumerate(r_in_2d):
        center = (int(point[0]), int(point[1]))
        radius = int(radii[i])
        cv2.circle(image, center, radius, color, thickness)

    # 裁剪图像
    image = image[571:1215,0:1935]
    cv2.imshow('Image with RCS-Based Points', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #cv2.imwrite('C:/Users/PerrySong_nku/Desktop/writing/example_rcs.jpg', image)


def draw_with_depth(r_in_2d, depth_list, image, dir=None):
    # 定义固定的半径
    radius = 5

    thickness = -1  # 圆的填充

    # 假设 depth_list 的范围为 [0, max_depth]，根据实际数据分布设置 max_depth
    max_depth = np.max(depth_list)
    min_depth = np.min(depth_list)


    # 将深度归一化到 [0, 1] 范围，深度越远颜色越深
    depth_normalized = np.clip((depth_list - min_depth) / (max_depth - min_depth), 0, 1)

    for i, point in enumerate(r_in_2d):
        center = (int(point[0]), int(point[1]))

        # 颜色映射：距离越远颜色越深
        depth_intensity = int(255 * depth_normalized[i])
        color = (0, 0, depth_intensity)

        cv2.circle(image, center, radius, color, thickness)

    # # 裁剪图像
    image = image[571:1215, 0:1935]
    cv2.imshow('Image with Depth-Based Points', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if dir is not None:
        cv2.imwrite(dir,image)

def is_point_near_bbox(point, bbox, threshold):
    x, y, z = point[:3]
    x_min, x_max, y_min, y_max, z_min, z_max = bbox

    # 判断 x, y, z 是否在 bbox 边界范围内或者半米范围内
    return (x_min - threshold <= x <= x_max + threshold) and \
        (y_min - threshold <= y <= y_max + threshold) and \
        (z_min - threshold <= z <= z_max + threshold)

def filter_points_in_bbox(r_in, r_in_guess, bbox_3d, distance_threshold=0.5):

    filtered_r_in = []
    for point in r_in:
        for bbox in bbox_3d:
            if is_point_near_bbox(point, bbox, distance_threshold):
                filtered_r_in.append(point)
                break  # 一个点只需要满足一个 bbox 就可以

    # 筛选 r_in_guess 中符合条件的点
    filtered_r_in_guess = []
    for point in r_in_guess:
        for bbox in bbox_3d:
            if is_point_near_bbox(point, bbox, distance_threshold):
                filtered_r_in_guess.append(point)
                break  # 一个点只需要满足一个 bbox 就可以

    # 转换为 numpy 数组
    filtered_r_in = np.array(filtered_r_in)
    filtered_r_in_guess = np.array(filtered_r_in_guess)

    return filtered_r_in, filtered_r_in_guess

def filter_points_in_bbox_one(r_in, bbox_3d, distance_threshold=0):

    filtered_r_in = []
    for point in r_in:
        for bbox in bbox_3d:
            if is_point_near_bbox(point, bbox, distance_threshold):
                filtered_r_in.append(point)
                break

    filtered_r_in = np.array(filtered_r_in)
    return filtered_r_in

def filter_points_in_bbox_single_box(r_in, bbox_3d, distance_threshold=0):

    filtered_r_in = []
    for point in r_in:
        if is_point_near_bbox(point, bbox_3d, distance_threshold):
            filtered_r_in.append(point)

    filtered_r_in = np.array(filtered_r_in)
    return filtered_r_in

def filter_lidar_points_corr_bbox(l_in, bbox_3d, distance_threshold=0):
    l_tmp = []
    for point in l_in:
        if is_point_near_bbox(point, bbox_3d, distance_threshold):
            l_tmp.append(point)
    l_tmp = np.array(l_tmp)

    return l_tmp

def get_pitch_yaw(points):
    pitch = np.arctan2(points[:, 2], np.sqrt(points[:, 0]**2 + points[:, 1]**2)) * 180 / np.pi  # 计算 pitch
    yaw = np.arctan2(points[:, 1], points[:, 0]) * 180 / np.pi  # 计算 yaw
    return pitch, yaw


def calculate_angle_and_depth_diff(r_in, l_in, pitch_radar, yaw_radar, pitch_lidar, yaw_lidar, thres_yaw,
                                   thres_pitch, thres_depth, T_radar):

    lidar_angles = np.vstack((pitch_lidar, yaw_lidar)).T  # 激光点的 pitch 和 yaw
    min_pitch, max_pitch = min(lidar_angles[:,0]), max(lidar_angles[:,0])
    min_yaw, max_yaw = min(lidar_angles[:, 1]), max(lidar_angles[:, 1])
    # print('min_pitch, max_pitch:',min_pitch, max_pitch)
    # print('min_yaw, max_yaw:', min_yaw, max_yaw)
    kdtree_lidar = cKDTree(lidar_angles)  # 在角度空间中构建 KDTree

    green_points = []  # 符合条件的点
    red_points = []  # 不符合条件的点

    for i, r_point in enumerate(r_in):
        # 获取雷达点的角度
        r_pitch = pitch_radar[i]
        r_yaw = yaw_radar[i]

        # 查询最近的激光点
        _, idx_lidar = kdtree_lidar.query([r_pitch, r_yaw])  # 查找最接近的激光点的索引

        # 获取最近激光点的 pitch 和 yaw
        pitch_diff = np.abs(r_pitch - pitch_lidar[idx_lidar])
        yaw_diff = np.abs(r_yaw - yaw_lidar[idx_lidar])

        r_point_cam = utils.transform_coor_to_radar(r_point, np.linalg.inv(T_radar))
        l_point_cam = utils.transform_coor_to_radar(l_in[idx_lidar], np.linalg.inv(T_radar))

        # depth_diff = np.abs(np.linalg.norm(r_point) - np.linalg.norm(l_in[idx_lidar]))/np.linalg.norm(l_in[idx_lidar])
        depth_diff = np.abs(r_point_cam[2] - l_point_cam[2]) / l_point_cam[2]

        # 判断是否符合角度差和深度差的条件
        if pitch_diff <= thres_pitch and yaw_diff <= thres_yaw and depth_diff <= thres_depth:
            green_points.append(i)
        elif pitch_diff <= thres_pitch and yaw_diff <= thres_yaw and depth_diff > thres_depth:
            red_points.append(i)
            print(f'radar point depth mismatched: point {r_point}, cam_coor: {r_point_cam}')
            print(f'pitch_diff:{pitch_diff},yaw_diff:{yaw_diff},depth_diff:{depth_diff}')
            print(f'corresponding lidar point: {l_in[idx_lidar]}, cam_coor: {l_point_cam}')
        else:
            print(f'valid lidar point not found: point {r_point}')
            print(f'pitch_diff:{pitch_diff},yaw_diff:{yaw_diff},depth_diff:{depth_diff}')

    return green_points, red_points


def calculate_angle_and_depth_diff_avg_err(r_in, l_in, pitch_radar, yaw_radar, pitch_lidar, yaw_lidar, thres_yaw,
                                   thres_pitch, T_radar):

    lidar_angles = np.vstack((pitch_lidar, yaw_lidar)).T  # 激光点的 pitch 和 yaw
    min_pitch, max_pitch = min(lidar_angles[:,0]), max(lidar_angles[:,0])
    min_yaw, max_yaw = min(lidar_angles[:, 1]), max(lidar_angles[:, 1])

    kdtree_lidar = cKDTree(lidar_angles)  # 在角度空间中构建 KDTree

    found_points = []
    depth_err = []

    for i, r_point in enumerate(r_in):
        r_pitch = pitch_radar[i]
        r_yaw = yaw_radar[i]

        _, idx_lidar = kdtree_lidar.query([r_pitch, r_yaw])  # 查找最接近的激光点的索引

        pitch_diff = np.abs(r_pitch - pitch_lidar[idx_lidar])
        yaw_diff = np.abs(r_yaw - yaw_lidar[idx_lidar])

        r_point_cam = utils.transform_coor_to_radar(r_point,np.linalg.inv(T_radar))
        l_point_cam = utils.transform_coor_to_radar(l_in[idx_lidar],np.linalg.inv(T_radar))

        # depth_diff_rate = np.abs(np.linalg.norm(r_point) - np.linalg.norm(l_in[idx_lidar]))/np.linalg.norm(l_in[idx_lidar])
        depth_diff_rate = np.abs(r_point_cam[2] - l_point_cam[2]) / l_point_cam[2]

        if pitch_diff <= thres_pitch and yaw_diff <= thres_yaw:
            found_points.append(i)
            depth_err.append(depth_diff_rate)


    return found_points, depth_err


def visualize_points(r_in, image, T_radar, K, green_points, red_points):
    green_radar_points = r_in[green_points]
    red_radar_points = r_in[red_points]
    utils.visulize_vod(green_radar_points, image, T_radar, K, [0, 255, 0])  # 绿色
    utils.visulize_vod(red_radar_points, image, T_radar, K, [0, 0, 255], 'E:/writing/depth_estimation.jpg')  # 红色

def process_frame(frame,o,thres_yaw, thres_pitch):
    radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(frame)
    K = utils.get_intrinsic_matrix(calib_file)
    if K is None or (not os.path.exists(txt_base_dir + str(frame).zfill(5) + '.txt')):
        #print(f'frame {frame} missing intrinsic matrix.')
        return None, None, False

    T_radar = utils.get_radar2cam(calib_file)
    T_lidar = utils.get_lidar2cam(lidar_calib_file)
    r_in = np.load(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/r_in/{frame}.npy')[:, :3]
    # l_in = np.load(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/l_in_radar_coor/{frame}.npy')
    l_in = np.fromfile(f'D:/VoD_dataset/view_of_delft_PUBLIC/lidar/training/velodyne/{str(frame).zfill(5)}.bin',
                       dtype=np.float32).reshape(-1, 4)[:, 0:3]
    l_in = l_in[l_in[:, 0] > 2]
    l_in = utils.trans_point_coor(l_in, np.dot(np.linalg.inv(T_radar), T_lidar))
    # l_in = np.load(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/lidar2d_corr_3dpoints/{frame}.npy')
    #
    #
    # print(l_in[0:5, :])
    # print('----------------------------------------------')
    # print(l_in_radar_coor[0:5,:])

    annotations = utils.read_annotation(txt_base_dir + str(frame).zfill(5) + '.txt')
    image = cv2.imread(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/image_2/{str(frame).zfill(5)}.jpg')
    # image1 = image.copy()
    # utils.visulize_vod(l_in, image1, T_radar, K,[255,0,0])
    bbox_3d = []
    for anno in annotations:
        bbox_camera, loc = utils.read_3dbbox(anno)
        distance = np.linalg.norm(loc)
        valid = False
        # if o == '0_10':
        #     if 0 <= distance <= 10:
        #         valid = True
        # elif o == '10_20':
        #     if 10 < distance <= 20:
        #         valid = True
        # elif o == '20_30':
        #     if 20 < distance <= 30:
        #         valid = True
        # elif o == '30_40':
        #     if 30 < distance <= 40:
        #         valid = True
        # else:  # o == 'far':
        #     if 40 < distance:
        #         valid = True
        if o == 'close':
            if 0 <= distance < 10:
                valid = True
        elif o == 'mid':
            if 10 <= distance < 30:
                valid = True
        elif o == 'far':
            if 30 <= distance:
                valid = True
        # distances.append(distance)  # 将距离添加到列表中
        if valid is True:
            bbox_radar = utils.transform_bbox_to_radar(bbox_camera, T_radar)
            bbox_3d.append(bbox_radar)

    r_in_near_bbox = filter_points_in_bbox_one(r_in, bbox_3d, 0)
    l_in_near_bbox = filter_points_in_bbox_one(l_in, bbox_3d, 0)
    num_total1, num_total2 = r_in_near_bbox.shape[0], l_in_near_bbox.shape[0]
    if num_total1 == 0 or num_total2 == 0:
        #print(f'frame {frame} does not have enough points.')
        return None, None, False

    pitch_radar, yaw_radar = get_pitch_yaw(r_in_near_bbox)
    pitch_lidar, yaw_lidar = get_pitch_yaw(l_in_near_bbox)
    # print(len(pitch_radar))

    thres_depth = 0.1
    green_points, red_points = calculate_angle_and_depth_diff(r_in_near_bbox, l_in_near_bbox, pitch_radar, yaw_radar, pitch_lidar,
                                                              yaw_lidar, thres_yaw, thres_pitch, thres_depth, T_radar)
    visualize_points(r_in, image, T_radar, K, green_points, red_points)
    return
    # found_points, depth_err = calculate_angle_and_depth_diff_avg_err(r_in_near_bbox, l_in_near_bbox, pitch_radar,
    #                                                                  yaw_radar,
    #                                                                  pitch_lidar, yaw_lidar, thres_yaw, thres_pitch, T_radar)
    #return len(found_points) / num_total1, depth_err, True



def visualize_radar_points(points):
    # 将valid_r_in转换为open3d的PointCloud对象
    pcd = o3d.geometry.PointCloud()

    # 将valid_r_in的3D坐标添加到点云中
    pcd.points = o3d.utility.Vector3dVector(points)

    # 使用open3d可视化点云
    o3d.visualization.draw_geometries([pcd],
                                      window_name="Radar Point Cloud",
                                      width=800, height=600,
                                      left=50, top=50,
                                      point_show_normal=False)

def calculate_pixel_offset_to_closest_bbox_point(shift_r_in, bbox_radar, T_radar, K):
    """
    计算shift_r_in内的每个点对应的像素点到边界框最近的那个点的像素点之间的偏差
    :param shift_r_in: 需要计算的雷达点坐标 (N, 3) 数组
    :param bbox_radar: 边界框的边界值 [xmin, xmax, ymin, ymax, zmin, zmax]
    :param T_radar: 雷达到相机坐标系的变换矩阵
    :param K: 相机内参矩阵
    :return: 每个点的像素点偏差 (N, 2) 数组，表示每个点在图像上的(u, v)偏差
    """
    xmin, xmax, ymin, ymax, zmin, zmax = bbox_radar

    # 初始化一个数组来存储每个点的像素偏差
    pixel_offsets = []

    # 计算每个点到边界框最近点的偏差
    for point in shift_r_in:
        x, y, z = point

        # 计算每个维度上到边界框最近点的距离
        nearest_x = np.clip(x, xmin, xmax)
        nearest_y = np.clip(y, ymin, ymax)
        nearest_z = np.clip(z, zmin, zmax)

        # 计算最近点的坐标
        nearest_point = np.array([nearest_x, nearest_y, nearest_z])

        # 将雷达坐标系中的最近点转换到相机坐标系
        nearest_point_camera = np.dot(T_radar, np.append(nearest_point, 1))[:3]

        # 使用相机内参矩阵将最近点从3D转换到2D图像坐标
        fx, cx = K[0, 0], K[0, 2]
        fy, cy = K[1, 1], K[1, 2]
        u_nearest = fx * nearest_point_camera[0] / nearest_point_camera[2] + cx
        v_nearest = fy * nearest_point_camera[1] / nearest_point_camera[2] + cy

        # 将当前点转换到相机坐标系
        point_camera = np.dot(T_radar, np.append(point, 1))[:3]

        # 使用相机内参矩阵将当前点从3D转换到2D图像坐标
        u_current = fx * point_camera[0] / point_camera[2] + cx
        v_current = fy * point_camera[1] / point_camera[2] + cy

        # 计算像素点之间的偏差
        delta_u = u_current - u_nearest
        delta_v = v_current - v_nearest

        # 存储偏差
        pixel_offsets.append([delta_u, delta_v])

    return np.array(pixel_offsets)


def get_target_point(frame, index):
    signal = False
    radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(frame)
    K = utils.get_intrinsic_matrix(calib_file)
    if K is None or (not os.path.exists(txt_base_dir + str(frame).zfill(5) + '.txt')):
        # print(f'frame {frame} missing intrinsic matrix.')
        return
    T_radar = utils.get_radar2cam(calib_file)
    T_lidar = utils.get_lidar2cam(lidar_calib_file)
    r_in = np.load(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/r_in/{frame}.npy')[:, :3]
    r_in_2d = np.load(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/r_in_2d/{frame}.npy')
    l_in = np.fromfile(f'D:/VoD_dataset/view_of_delft_PUBLIC/lidar/training/velodyne/{str(frame).zfill(5)}.bin',
                       dtype=np.float32).reshape(-1, 4)[:, 0:3]
    l_in = l_in[l_in[:, 0] > 2]
    l_in = utils.trans_point_coor(l_in, np.dot(np.linalg.inv(T_radar), T_lidar))

    annotations = utils.read_annotation(txt_base_dir + str(frame).zfill(5) + '.txt')

    image = cv2.imread(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/image_2/{str(frame).zfill(5)}.jpg')
    if index < len(annotations):
        signal = True
    else:
        return signal, None, None
    anno = annotations[index]
    cl = anno['Class']
    #print(f'frame {frame}, index {index}, object class {cl}')
    #print(anno)
    umin,vmin,umax,vmax = anno['Bbox']
    shift = 0

    bbox_camera, loc = utils.read_3dbbox(anno)
    #print('target loc: ',loc)
    #print('camera coor location & dimension:',anno['Location'],anno['Dimensions'])
    bbox_radar = utils.transform_bbox_to_radar(bbox_camera, T_radar)
    #print('target bbox in radar coor:',bbox_radar)
    #print('target bbox in camera coor:',bbox_camera)

    image1 = image.copy()


    #get target point by 2d bbox
    # valid_indices = np.where(
    #     (r_in_2d[:, 0] >= umin-shift) & (r_in_2d[:, 0] <= umax+shift) &
    #     (r_in_2d[:, 1] >= vmin-shift) & (r_in_2d[:, 1] <= vmax+shift)
    # )[0]
    # valid_r_in, valid_r_in_2d = r_in[valid_indices], r_in_2d[valid_indices]
    # #print(f'retrieved {len(valid_r_in)} points through 2D bbox.')
    # #print(valid_r_in)
    # r_point_cam = utils.trans_point_coor(valid_r_in, T_radar)
    # #print('depth of points:',r_point_cam[:,2])
    #
    # z_min, z_max = bbox_camera[4], bbox_camera[5]  # 假设bbox_camera的z轴范围是[4, 5]
    # valid_depth_indices = np.where((r_point_cam[:, 2] >= z_min) & (r_point_cam[:, 2] <= z_max))[0]
    # filtered_r_in = valid_r_in[valid_depth_indices]
    # filtered_r_in_2d = valid_r_in_2d[valid_depth_indices]
    # # print(filtered_r_in)
    # # print(filtered_r_in_2d)
    # if filtered_r_in_2d.shape[0] > 1:
    #     center = np.array([(umin + umax) / 2, (vmin + vmax) / 2])
    #     diff = filtered_r_in_2d - center
    #     #print('object 2D center coor:',center)
    #     u_variance = np.var(diff[:, 0])  # u 坐标的方差
    #     v_variance = np.var(diff[:, 1])  # v 坐标的方差
    #     #print(f"u 坐标的方差: {u_variance}")
    #     #print(f"v 坐标的方差: {v_variance}")
    # else:
    #     u_variance, v_variance = None, None

    #print(f'{len(filtered_r_in)} points are kept after filtering by depth.')
    #print('----------------------------------------------------------------------')

    # cv2.rectangle(image, (int(umin), int(vmin)), (int(umax), int(vmax)), (255, 0, 0), 2)
    # utils.visulize_vod(valid_r_in, image, T_radar, K, [0, 0, 255])
    # utils.visulize_vod(filtered_r_in, image, T_radar, K, [0, 255, 0])
    #visualize_radar_points(valid_r_in)


    # get target point by 3d bbox
    xmin, xmax, ymin, ymax, zmin, zmax = bbox_radar
    shift = 0.5
    valid_indices = np.where(
        (r_in[:, 0] >= xmin-shift) & (r_in[:, 0] <= xmax+shift) &
        (r_in[:, 1] >= ymin-shift) & (r_in[:, 1] <= ymax+shift) &
        (r_in[:, 2] >= zmin-shift) & (r_in[:, 2] <= zmax+shift)
    )[0]
    valid_r_in, valid_r_in_2d = r_in[valid_indices], r_in_2d[valid_indices]
    #print(f'retrieved {len(valid_r_in)} points through 3D bbox with shift {shift}.')
    valid_indices_shift_0 = np.where(
        (r_in[:, 0] >= xmin) & (r_in[:, 0] <= xmax) &
        (r_in[:, 1] >= ymin) & (r_in[:, 1] <= ymax) &
        (r_in[:, 2] >= zmin) & (r_in[:, 2] <= zmax)
    )[0]

    # 找出在shift=0.5时满足条件，但是在shift=0时不满足的点
    shift_diff_indices = np.setdiff1d(valid_indices, valid_indices_shift_0)
    delta_u,delta_v = None, None
    if len(shift_diff_indices) > 0:
        #print(f'{len(shift_diff_indices)} points near the bbox.')
        shift_r_in = r_in[shift_diff_indices]
        shift_r_in_2d = r_in_2d[shift_diff_indices]
        pixel_offsets = calculate_pixel_offset_to_closest_bbox_point(shift_r_in,bbox_radar,T_radar,K)
        delta_u,delta_v = list(abs(pixel_offsets[:,0])),list(abs(pixel_offsets[:,1]))

    # r_point_cam = utils.trans_point_coor(valid_r_in, T_radar)
    # print('depth of points:', r_point_cam[:, 2])
    # utils.visulize_vod(valid_r_in, image1, T_radar, K, [0, 0, 255])
    #visualize_radar_points(valid_r_in)



    return signal, delta_u,delta_v


def covariance_3d(frame,o):
    radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(frame)
    K = utils.get_intrinsic_matrix(calib_file)
    if K is None or (not os.path.exists(txt_base_dir + str(frame).zfill(5) + '.txt')):
        #print(f'frame {frame} missing intrinsic matrix.')
        return None, None, False

    T_radar = utils.get_radar2cam(calib_file)
    T_lidar = utils.get_lidar2cam(lidar_calib_file)
    r_in = np.load(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/r_in/{frame}.npy')[:, :3]
    # l_in = np.load(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/l_in_radar_coor/{frame}.npy')
    l_in = np.fromfile(f'D:/VoD_dataset/view_of_delft_PUBLIC/lidar/training/velodyne/{str(frame).zfill(5)}.bin',
                       dtype=np.float32).reshape(-1, 4)[:, 0:3]
    l_in = l_in[l_in[:, 0] > 2]
    l_in = utils.trans_point_coor(l_in, np.dot(np.linalg.inv(T_radar), T_lidar))

    annotations = utils.read_annotation(txt_base_dir + str(frame).zfill(5) + '.txt')
    image = cv2.imread(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/image_2/{str(frame).zfill(5)}.jpg')

    bbox_3d = []
    for anno in annotations:
        bbox_camera, loc = utils.read_3dbbox(anno)
        distance = np.linalg.norm(loc)
        valid = False

        if o == 'close':
            if 0 <= distance < 10:
                valid = True
        elif o == 'mid':
            if 10 <= distance < 30:
                valid = True
        elif o == 'far':
            if 30 <= distance:
                valid = True
        # distances.append(distance)  # 将距离添加到列表中
        if valid is True:
            bbox_radar = utils.transform_bbox_to_radar(bbox_camera, T_radar)
            bbox_3d.append(bbox_radar)

    r_in_near_bbox = filter_points_in_bbox_one(r_in, bbox_3d, 0)
    l_in_near_bbox = filter_points_in_bbox_one(l_in, bbox_3d, 0)

    if r_in_near_bbox is None or len(r_in_near_bbox) <= 3:
        return None, None, False

        # 计算协方差矩阵（shape = [3,3]）
    cov_matrix = np.cov(r_in_near_bbox.T)  # 每一列是 x, y, z

    # 获取主对角线元素（方差），并开方得到标准差
    std_devs = np.sqrt(np.diag(cov_matrix))  # shape: [3,]，即 [sx, sy, sz]

    return std_devs.tolist(), cov_matrix, True





if __name__ == '__main__':
    #-------------------------------process statistics on the whold dataset------------------------------------------
    sequences = {
        "Sequence 1": (0, 543),
        "Sequence 2": (544, 1311),
        "Sequence 3": (1312, 1802),
        "Sequence 4": (1803, 2199),
        "Sequence 5": (2200, 2531),
        #"Sequence 6": (2532, 2797),
        #"Sequence 7": (2798, 3574),
        "Sequence 8": (3575, 4047),
        "Sequence 9": (4049, 4386),  # Frame 04048 is missing   #urban
        "Sequence 10": (4387, 5085),
        #"Sequence 11": (6334, 6570),  # Skipping Frame 05085-06334 as missing
        #"Sequence 12": (6571, 6758),
        "Sequence 13": (6759, 7542),
        #"Sequence 14": (7543, 7899),
        #"Sequence 15": (7900, 8197),
        "Sequence 16": (8198, 8480),
        "Sequence 17": (8481, 8748),    #urban
        "Sequence 18": (8749, 9095),    #urban
        #"Sequence 19": (9096, 9517),
        "Sequence 20": (9518, 9775),    #urban
        "Sequence 21": (9776, 9930),    #urban
    }
    object_type = ['near', 'mid', 'far']
    for o in object_type:
        print('dealing with object type:', o)
        var_3d = []
        # for frame in range(0,9931):
        # frame = 3574
        # print(f'dealing with frame {frame}')
        sampled_frame_file = f"D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/{o}_sampled_frame.npy"
        if os.path.exists(sampled_frame_file):
            sampled_frame_list = np.load(sampled_frame_file)
            print('loading sampled frames from local file.')
            for frame in sampled_frame_list:
                std_devs, cov_matrix, sig = covariance_3d(frame, o)
                if sig is True:
                    var_3d.append(std_devs)
            var_3d = np.array(var_3d)  # shape: [N, 3]
            if len(var_3d) > 0:
                mean_std = np.mean(var_3d, axis=0)  # [mean_sx, mean_sy, mean_sz]
                print(f'Mean std for {o}: sx={mean_std[0]:.4f}, sy={mean_std[1]:.4f}, sz={mean_std[2]:.4f}')
            else:
                print(f'No valid data for object type: {o}')
        else:
            sampled_frame = []
            for sequence_name, (start_frame, end_frame) in sequences.items():
                print(f'Sampling from {sequence_name}, frames {start_frame}-{end_frame}')
                sampled_frames = 0

                while sampled_frames < 10:
                    frame = random.randint(start_frame, end_frame)
                    std_devs, cov_matrix, sig = covariance_3d(frame, o)
                    if sig is True:
                        var_3d.append(std_devs)
                    sampled_frame.append(frame)
                    sampled_frames += 1
            var_3d = np.array(var_3d)  # shape: [N, 3]
            if len(var_3d) > 0:
                mean_std = np.mean(var_3d, axis=0)  # [mean_sx, mean_sy, mean_sz]
                print(f'Mean std for {o}: sx={mean_std[0]:.4f}, sy={mean_std[1]:.4f}, sz={mean_std[2]:.4f}')
            else:
                print(f'No valid data for object type: {o}')
            np.save(f"D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/{o}_sampled_frame.npy",
                    np.array(sampled_frame))



    thres_yaw = 5
    thres_pitch = 5
    # #distances = []
    # object_type = ['near','mid','far']
    # for o in object_type:
    #     print('dealing with object type:',o)
    #     err_list = []
    #     point_reserved = []
    #     #for frame in range(0,9931):
    #     #frame = 3574
    #         #print(f'dealing with frame {frame}')
    #     sampled_frame_file = f"D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/{o}_sampled_frame.npy"
    #     if os.path.exists(sampled_frame_file):
    #         sampled_frame_list = np.load(sampled_frame_file)
    #         print('loading sampled frames from local file.')
    #         for frame in sampled_frame_list:
    #             valid_point_rate, depth_err, sign = process_frame(frame,o,thres_yaw, thres_pitch)
    #             point_reserved.append(valid_point_rate)
    #             err_list.extend(depth_err)
    #     else:
    #         sampled_frame = []
    #         for sequence_name, (start_frame, end_frame) in sequences.items():
    #             print(f'Sampling from {sequence_name}, frames {start_frame}-{end_frame}')
    #             sampled_frames = 0
    #
    #             while sampled_frames < 10:
    #                 frame = random.randint(start_frame, end_frame)
    #                 valid_point_rate, depth_err, sign = process_frame(frame,o,thres_yaw, thres_pitch)
    #                 if sign is True:
    #                     point_reserved.append(valid_point_rate)
    #                     err_list.extend(depth_err)
    #                     sampled_frame.append(frame)
    #                     sampled_frames += 1
    #
    #         np.save(f"D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/{o}_sampled_frame.npy",
    #                 np.array(sampled_frame))

        # print('avg depth error:',np.sum(err_list)/len(err_list))
        # print('avg point reserved:',np.sum(point_reserved)/len(point_reserved))

    #------------------------------visulization on single image--------------------------------------------------------
    # thres_yaw, thres_pitch = 1,1
    #process_frame(0,'mid',thres_yaw, thres_pitch)


    #------------------------------visulization of points reflected by targets------------------------------------------
    # object_type = ['0_10','10_20','20_30','30_40','40_50']
    # for o in object_type:
    #     print(f'object type:{o}')
    #     sampled_frame_file = f"D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/{o}_sampled_frame.npy"
    #     sampled_frame_list = np.load(sampled_frame_file)
    #     u_total,v_total = [],[]
    #     cnt_var = 0
    #     total_frames = len(sampled_frame_list)
    #     for idx,frame in enumerate(sampled_frame_list):
    #         progress = (idx + 1) / total_frames * 100
    #         if progress % 10 == 0 and progress != 0:
    #             print(f'Processing: {int(progress)}% of frames completed for {o}')
    #             #print(u_total,v_total)
    #         #print(f'frame {frame}')
    #         cnt = 0
    #         #print(f'index {cnt}')
    #         sig,u_variance,v_variance = get_target_point(frame,cnt)
    #         cnt += 1
    #         if u_variance is not None:
    #             u_total.extend(u_variance)
    #             v_total.extend(v_variance)
    #             # cnt_var += 1
    #             cnt_var += len(u_variance)
    #         while sig:
    #             #print(f'index {cnt}')
    #             sig,u_variance,v_variance = get_target_point(frame,cnt)
    #             cnt += 1
    #             if u_variance is not None:
    #                 u_total.extend(u_variance)
    #                 v_total.extend(v_variance)
    #                 #cnt_var += 1
    #                 cnt_var += len(u_variance)
    #     u_variance_avg = u_total/cnt_var
    #     v_variance_avg = v_total/cnt_var
    #     print(f"Average u variance for {o}: {u_variance_avg}")
    #     print(f"Average v variance for {o}: {v_variance_avg}")

    # from concurrent.futures import ThreadPoolExecutor
    # def process_object_type(o):
    #     print(f'object type: {o}')
    #     sampled_frame_file = f"D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/{o}_sampled_frame.npy"
    #     sampled_frame_list = np.load(sampled_frame_file)
    #     u_total, v_total = [], []
    #     cnt_var = 0
    #     total_frames = len(sampled_frame_list)
    #
    #     for idx, frame in enumerate(sampled_frame_list):
    #         progress = (idx + 1) / total_frames * 100
    #         if progress % 10 == 0 and progress != 0:
    #             print(f'Processing: {int(progress)}% of frames completed for {o}')
    #         cnt = 0
    #         sig, u_variance, v_variance = get_target_point(frame, cnt)
    #         cnt += 1
    #         if u_variance is not None:
    #             u_total.extend(u_variance)
    #             v_total.extend(v_variance)
    #             cnt_var += len(u_variance)
    #
    #         while sig:
    #             sig, u_variance, v_variance = get_target_point(frame, cnt)
    #             cnt += 1
    #             if u_variance is not None:
    #                 u_total.extend(u_variance)
    #                 v_total.extend(v_variance)
    #                 cnt_var += len(u_variance)
    #
    #     # 计算平均值
    #     if cnt_var > 0:
    #         u_variance_avg = np.mean(u_total)
    #         v_variance_avg = np.mean(v_total)
    #         print(f"Average u variance for {o}: {u_variance_avg}")
    #         print(f"Average v variance for {o}: {v_variance_avg}")
    #     else:
    #         print(f"No valid data found for {o}")
    #
    #
    # # 线程数
    # max_threads = 5
    #
    # # 使用 ThreadPoolExecutor 并行处理
    # with ThreadPoolExecutor(max_workers=max_threads) as executor:
    #     executor.map(process_object_type, object_type)


    # for frame in range(0,9931):
    #     if not os.path.exists(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/r_in/{frame}.npy'):
    #         continue
    #     print(frame)
    #     r_in = np.load(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/r_in/{frame}.npy')
    #     r_in_2d = np.load(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/r_in_2d/{frame}.npy')
    #     depth = np.linalg.norm(r_in[:, :3], axis=1)
    #     valid_indices = depth <= 50
    #     r_in = r_in[valid_indices]
    #     r_in_2d = r_in_2d[valid_indices]
    #     np.save(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/r_in_50/{frame}.npy',r_in)
    #     np.save(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/r_in_2d_50/{frame}.npy', r_in_2d)
    #     np.save(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/depth_50/{frame}.npy', depth)



    #------------------------------------check point number in target zone and distribution------------------------------
    # df = pd.read_csv(
    #     'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/exp_2/estimated_pointnum_vanila_no_freeze_softmax.csv')
    # for i in range(len(df)):
    # #for i in range(0,15):
    #     num_samples = int(df.iloc[i]['Estimated_PointNum'])
    #     #num_samples = int(df3.iloc[i]['PointNum'])
    #     # num_samples *= 2
    #     frame = int(df.iloc[i]['Frame'])
    #     # if os.path.exists(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/generated_3d_pmap_adaptive_covariance/{frame}.npy'):
    #     #     continue
    #     print(f"dealing with frame {frame}")
    #     radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(frame)
    #     image = cv2.imread(cam_file)
    #     image2 = image.copy()
    #     K = utils.get_intrinsic_matrix(calib_file)
    #     T_radar = utils.get_radar2cam(calib_file)
    #     T_lidar = utils.get_lidar2cam(lidar_calib_file)
    #     r_in_2d = np.load(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/r_in_2d_50/{frame}.npy')
    #     annotations = utils.read_annotation(txt_base_dir + str(frame).zfill(5) + '.txt')
    #     r_in = np.load(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/r_in_50/{frame}.npy')
    #     r_guess = np.load(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/generated_3d_pmap_adaptive_covariance/{frame}.npy')
    #
    #     if annotations is None:
    #         continue
    #     bbox_3d = []
    #     bbox_2d = []
    #     bbox_cam = []
    #     num_real_in_2d_bbox = 0
    #     num_guess_in_2d_bbox = 0
    #     for anno in annotations:
    #         bbox_camera, loc = utils.read_3dbbox(anno)
    #         bbox_radar = utils.transform_bbox_to_radar(bbox_camera, T_radar)
    #         bbox_3d.append(bbox_radar)
    #         umin, vmin, umax, vmax = anno['Bbox']
    #         bbox_2d.append([umin, vmin, umax, vmax])
    #         bbox_cam.append(bbox_camera)




