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

class_list = ['ride_uncertain', 'rider', 'moped_scooter', 'bicycle', 'Cyclist', 'vehicle_other', 'Pedestrian', 'truck',
              'DontCare', 'motor', 'bicycle_rack', 'Car', 'human_depiction', 'ride_other']
# 'DontCare' isn't included

def create_point_cloud(points):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    return pc

def compute_class_statistics(data):
    stats = []

    classes = data['Class'].unique()

    for cls in classes:
        class_data = data[data['Class'] == cls]

        min_frames = class_data['Consecutive_Frames'].min()
        max_frames = class_data['Consecutive_Frames'].max()
        avg_frames = class_data['Consecutive_Frames'].mean()

        stats.append({
            'Class': cls,
            'Min_Consecutive_Frames': min_frames,
            'Max_Consecutive_Frames': max_frames,
            'Avg_Consecutive_Frames': avg_frames
        })

    return pd.DataFrame(stats)

def trackid2pcd(track_id,frame):
    label_path = 'D:/VoD_dataset/label_2_with_track_ids/label_2'
    file_path = os.path.join(label_path, f'{frame:05d}.txt')
    radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(frame)
    T_radar = utils.get_radar2cam(calib_file)
    annotations = utils.read_annotation(file_path)
    pcd = None
    for j in range(len(annotations)):
        track_tmp = annotations[j]['Track_ID']
        if track_tmp == track_id:   #found
            bbox, loc = utils.read_3dbbox(annotations[j])
            bbox = utils.transform_bbox_to_radar(bbox, T_radar)
            loc = utils.transform_coor_to_radar(loc, T_radar)
            dimension = annotations[j]['Dimensions']
            yaw = annotations[j]['Rotation']
            pcd = np.fromfile(radar_file, dtype=np.float32).reshape(-1, 7)
            pcd = utils.filter_points_in_bbox(pcd, loc, dimension, yaw)
            pcd = pcd[pcd[:, 0] > 0]
            break
    if pcd is None:
        print(f'track id {track_id} not found in frame {frame}')
    return pcd

def track_id_f2bbox(track_id,frame):
    label_path = 'D:/VoD_dataset/label_2_with_track_ids/label_2'
    file_path = os.path.join(label_path, f'{frame:05d}.txt')
    radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(frame)
    T_radar = utils.get_radar2cam(calib_file)
    annotations = utils.read_annotation(file_path)
    bbox, loc, yaw = None, None, None
    for j in range(len(annotations)):
        track_tmp = annotations[j]['Track_ID']
        if track_tmp == track_id:  # found
            bbox, loc = utils.read_3dbbox(annotations[j])
            bbox = utils.transform_bbox_to_radar(bbox, T_radar)
            loc = utils.transform_coor_to_radar(loc, T_radar)
            yaw = annotations[j]['Rotation']
            dimension = annotations[j]['Dimensions']
            break
    return bbox, loc, yaw

def compute_point_cloud_statistics(point_cloud, bbox, loc):
    """
    计算点云的密度、质心（与loc的偏移）、协方差矩阵和x、y、z方向上的最小间距

    Parameters:
    point_cloud (numpy array): 点云数据 (N x 3 or N x 4, 只使用前三列)
    bbox (array-like): 3D bounding box [xmin, xmax, ymin, ymax, zmin, zmax]
    loc (array-like): Bounding box 的中心坐标 [x, y, z]

    Returns:
    density (float): 点云密度
    centroid_offset (numpy array): 质心（与loc的偏移）
    covariance_matrix (numpy array): 协方差矩阵
    min_distances (numpy array): x, y, z方向上的最小间距
    """
    if point_cloud.shape[1] > 3:
        points = point_cloud[:, :3]
    else:
        points = point_cloud
    # 计算点云密度
    bbox_volume = (bbox[1] - bbox[0]) * (bbox[3] - bbox[2]) * (bbox[5] - bbox[4])
    density = len(points) / bbox_volume

    # 计算质心
    centroid = np.mean(points, axis=0)
    centroid_offset = centroid - np.array(loc)

    # 计算协方差矩阵
    covariance_matrix = np.cov(points, rowvar=False)

    # 计算x, y, z方向上的最小间距
    min_distances = np.zeros(3)
    for i in range(3):
        diffs = np.abs(np.diff(np.sort(points[:, i])))

        min_distances[i] = np.min(diffs) if len(diffs) > 0 else 0.0

    return density, centroid, centroid_offset, covariance_matrix, min_distances


def cal_1pair_rcs(track_id,src_frame,tar_frame,thres):
    #use radar_5frames
    frame0, frame1 = src_frame,tar_frame
    radar_file1, lidar_file1, cam_file1, calib_file1, lidar_calib_file1, txt_base_dir1 = utils.get_vod_dir(frame0)
    radar_file2, lidar_file2, cam_file2, calib_file2, lidar_calib_file2, txt_base_dir2 = utils.get_vod_dir(frame1)
    annotation_file1 = txt_base_dir1 + str(src_frame).zfill(5) + '.txt'
    annotation_file2 = txt_base_dir2 + str(tar_frame).zfill(5) + '.txt'
    annotations1 = utils.read_annotation(annotation_file1)
    annotations2 = utils.read_annotation(annotation_file2)
    index1, index2 = -1, -1
    for j in range(len(annotations1)):
        if track_id == annotations1[j]['Track_ID']:
            index1 = j
            break
    for j in range(len(annotations2)):
        if track_id == annotations2[j]['Track_ID']:
            index2 = j
            break
    if index1 == -1 or index2 == -1:
        print('problem occured!')
        return

    T_radar1 = utils.get_radar2cam(calib_file1)
    T_radar2 = utils.get_radar2cam(calib_file2)
    pcl1 = np.fromfile(radar_file1, dtype=np.float32).reshape(-1, 7)[:, 0:4]
    pcl2 = np.fromfile(radar_file2, dtype=np.float32).reshape(-1, 7)[:, 0:4]
    bbox1, loc1 = utils.read_3dbbox(annotations1[index1])
    bbox2, loc2 = utils.read_3dbbox(annotations2[index2])
    bbox1 = utils.transform_bbox_to_radar(bbox1, T_radar1)
    bbox2 = utils.transform_bbox_to_radar(bbox2, T_radar2)
    loc1 = utils.transform_coor_to_radar(loc1, T_radar1)
    loc2 = utils.transform_coor_to_radar(loc2, T_radar2)
    dimension1 = annotations1[index1]['Dimensions']
    dimension2 = annotations2[index2]['Dimensions']
    yaw1 = annotations1[index1]['Rotation']
    yaw2 = annotations2[index2]['Rotation']
    pcd1 = utils.filter_points_in_bbox(pcl1, loc1, dimension1, yaw1)
    pcd2 = utils.filter_points_in_bbox(pcl2,  loc2, dimension2, yaw2)
    odom_transform, _, _ = utils.compute_transform(frame0, frame1,T_radar1,T_radar2)
    if odom_transform is None:
        return np.array([]),0
    xyz1 = pcd1[:, :3]
    xyz2 = pcd2[:, :3]
    rcs1 = pcd1[:, 3]
    rcs2 = pcd2[:, 3]
    transformed_xyz1 = (odom_transform @ np.hstack((xyz1, np.ones((xyz1.shape[0], 1)))).T).T[:, :3]
    kdtree = KDTree(xyz2)
    distances, indices = kdtree.query(transformed_xyz1)
    #print(distances)
    rcs_differences = []
    for i, distance in enumerate(distances):
        if distance < thres:
            nearest_rcs = rcs2[indices[i]]
            rcs_difference = rcs1[i] - nearest_rcs
            rcs_differences.append(rcs_difference)

    return np.array(rcs_differences), len(np.array(rcs_differences))

# label_path = 'D:/VoD_dataset/label_2_with_track_ids/label_2'
# class_track_info = {cls: defaultdict(list) for cls in class_list}
# for frame in range(9931):
#     file_path = os.path.join(label_path, f'{frame:05d}.txt')
#     if os.path.exists(file_path):
#         annotations = utils.read_annotation(file_path)
#         for annotation in annotations:
#             cls = annotation['Class']
#             track_id = annotation['Track_ID']
#             class_track_info[cls][track_id].append(frame)
#
# with open('D:/VoD_dataset/track_info.csv', 'w', newline='') as csvfile:
#     fieldnames = ['Class', 'Track_ID', 'First_Appearance_Frame', 'Consecutive_Frames']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writeheader()
#
#     for cls, tracks in class_track_info.items():
#         for track_id, frames in tracks.items():
#             first_appearance = frames[0]
#             consecutive_frames = 1
#             for i in range(1, len(frames)):
#                 if frames[i] == frames[i-1] + 1:
#                     consecutive_frames += 1
#                 else:
#                     writer.writerow({
#                         'Class': cls,
#                         'Track_ID': track_id,
#                         'First_Appearance_Frame': first_appearance,
#                         'Consecutive_Frames': consecutive_frames
#                     })
#                     first_appearance = frames[i]
#                     consecutive_frames = 1
#             writer.writerow({
#                 'Class': cls,
#                 'Track_ID': track_id,
#                 'First_Appearance_Frame': first_appearance,
#                 'Consecutive_Frames': consecutive_frames
#             })
#
#
#
#
# csv_file_path = 'D:/VoD_dataset/track_info.csv'
# data = pd.read_csv(csv_file_path)
# statistics = compute_class_statistics(data)
# statistics.to_csv('D:/VoD_dataset/track_statistics.csv', index=False)



# csv_file_path = 'D:/VoD_dataset/track_info.csv'
# data = pd.read_csv(csv_file_path)
# data['valid'] = data['Consecutive_Frames'].apply(lambda x: 1 if x > 5 else 0)
# new_csv_file_path = 'D:/VoD_dataset/track_info_valid.csv'
# data.to_csv(new_csv_file_path, index=False)
# print(data)
# print(trackid2pcd(269,1991))







# track_info_path = 'D:/VoD_dataset/track_info.csv'
# df = pd.read_csv(track_info_path)
# result_df = df.copy()
#
#
# result_df['valid'] = 0
# for class_name in class_list:
#     class_tracks = df[df['Class'] == class_name]
#     for track_id in class_tracks['Track_ID'].unique():
#         track_data = class_tracks[class_tracks['Track_ID'] == track_id]
#         first_appearance = track_data['First_Appearance_Frame'].values[0]
#         #consecutive_frames = track_data['Consecutive_Frames'].values[0]
#         valid = True
#
#         pcd = trackid2pcd(track_id, first_appearance)
#         if pcd is None or len(pcd) == 0:
#             valid = False
#
#         result_df.loc[
#             (result_df['Track_ID'] == track_id) & (result_df['Class'] == class_name), 'valid'] = 1 if valid else 0
#
# result_df[result_df['valid'] == 1].to_csv('D:/VoD_dataset/track_info_valid_pcd.csv', index=False)
#
# # 统计每个类有多少valid=1的行
# valid_counts = result_df[result_df['valid'] == 1]['Class'].value_counts()
# print(valid_counts)





# track_info_path = 'D:/VoD_dataset/track_info_valid_pcd.csv'
# track_info = pd.read_csv(track_info_path)
# # 筛选出 valid 为 1 的行
# valid_tracks = track_info[track_info['valid'] == 1]
# data = []
# for index, row in valid_tracks.iterrows():
#     print(index,'/',len(valid_tracks))
#     track_id = row['Track_ID']
#     first_frame = row['First_Appearance_Frame']
#     consecutive_frames = row['Consecutive_Frames']
#
#     i = first_frame
#     while True:
#         if i + 5 - first_frame >= consecutive_frames:
#             break
#
#         src_pcd = trackid2pcd(track_id, i)
#         tar_pcd = trackid2pcd(track_id, i + 5)
#         if src_pcd is not None and tar_pcd is not None:
#             if len(src_pcd) > 0 and len(tar_pcd) > 0:
#                 data.append([track_id, i, i + 5])
#                 i += 1
#             else:
#                 break
# # 创建 DataFrame
# result_df = pd.DataFrame(data, columns=['track_id', 'src_frame', 'tar_frame'])
# # 保存为 CSV 文件
# result_csv_path = 'D:/VoD_dataset/track_id_frame_pairs.csv'
# result_df.to_csv(result_csv_path, index=False)
# print(f"Saved track_id_frame_pairs.csv to {result_csv_path}")






track_info_valid_pcd_path = 'D:/VoD_dataset/track_info_valid_pcd.csv'
track_id_frame_pairs_path = 'D:/VoD_dataset/track_id_frame_pairs.csv'
# 读取CSV文件
track_info_valid_pcd = pd.read_csv(track_info_valid_pcd_path)
track_id_frame_pairs = pd.read_csv(track_id_frame_pairs_path)
# 过滤掉 `valid` 列不为 1 的行
valid_track_info = track_info_valid_pcd[track_info_valid_pcd['valid'] == 1]
# # 创建一个空的字典来保存每个类的计数
# class_counts = {}
# # 遍历 track_id_frame_pairs
# for _, row in track_id_frame_pairs.iterrows():
#     track_id = row['track_id']
#     # 在 valid_track_info 中找到对应的类
#     class_name = valid_track_info[valid_track_info['Track_ID'] == track_id]['Class'].values[0]
#     # 更新计数
#     if class_name in class_counts:
#         class_counts[class_name] += 1
#     else:
#         class_counts[class_name] = 1
# # 将字典转换为 DataFrame
# class_counts_df = pd.DataFrame(list(class_counts.items()), columns=['Class', 'Count'])
# # 输出结果
# print(class_counts_df)
# # 保存统计结果为 CSV
# class_counts_df.to_csv('D:/VoD_dataset/class_counts.csv', index=False)
# print(f"Saved class_counts.csv to D:/VoD_dataset/class_counts.csv")



#add class for track_id_frame_pairs
# track_id_to_class = dict(zip(track_info_valid_pcd['Track_ID'], track_info_valid_pcd['Class']))
# track_id_frame_pairs['Class'] = track_id_frame_pairs['track_id'].map(track_id_to_class)
# track_id_frame_pairs.to_csv('D:/VoD_dataset/track_id_frame_pairs_class.csv', index=False)



#----------------------------------calculate rcs mse to determine the baseline of rcsnet-------------------------------
# total_rcs = None
# total_len_rcs = 0
# for c in class_list:
#     print('class ',c)
#     len_rcs = 0
#     rcs_array = None
#     data = dataset = track_id_frame_pairs[track_id_frame_pairs['Class'] == c].values
#     for i in range(len(data)):
#         track_id, src_frame, tar_frame = data[i][0:3]
#         rcs_array_tmp, len_rcs_tmp = cal_1pair_rcs(track_id, src_frame, tar_frame, 0.2)
#         if len_rcs_tmp > 0:
#             #print(rcs_array.shape)
#             if rcs_array is None:
#                 rcs_array = rcs_array_tmp
#             else:
#                 rcs_array = np.concatenate((rcs_array,rcs_array_tmp))
#         len_rcs += len_rcs_tmp
#     print(f'matched point found for class {c} is {len_rcs}')
#     if rcs_array is not None:
#         squared_elements = rcs_array ** 2
#         sum_of_squares = np.sum(squared_elements)
#         mse = sum_of_squares / len(rcs_array)
#         print(f'MSE of RCS values of matched points is {mse}')
#         if total_rcs is None:
#             total_rcs = rcs_array
#         else:
#             total_rcs = np.concatenate((total_rcs, rcs_array))
#     total_len_rcs += len_rcs
# squared_elements1 = total_rcs ** 2
# sum_of_squares1 = np.sum(squared_elements1)
# mse1 = sum_of_squares1 / len(total_rcs)
# print(f'mse of all matched points is {mse1}')
#----------------------------------calculate rcs mse to determine the baseline of rcsnet-------------------------------



# ---------------------calculte transformation matrix between frames by odometry data--------------------------
# frame0, frame1 = 0,10
# radar_file1, lidar_file1, cam_file1, calib_file1, lidar_calib_file1, txt_base_dir1 = utils.get_vod_dir(frame0)
# radar_file2, lidar_file2, cam_file2, calib_file2, lidar_calib_file2, txt_base_dir2 = utils.get_vod_dir(frame1)
# T_radar1 = utils.get_radar2cam(calib_file1)
# T_radar2 = utils.get_radar2cam(calib_file2)
# odom_transform, _, _ = utils.compute_transform(frame0, frame1,T_radar1,T_radar2)
# print("Odom Transform from frame0 to frame1:\n", odom_transform)




