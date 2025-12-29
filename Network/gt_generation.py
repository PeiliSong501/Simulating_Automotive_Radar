import pandas as pd
import numpy as np
import os
import cv2
import utils
import random
from sklearn.model_selection import train_test_split
import re
from scipy.spatial import KDTree

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

def radar_in_image(coor, pcd_radar,T_radar,intrinsic):
    point1, point2 = coor
    x1,y1 = point1
    x2,y2 = point2
    geo_radar = pcd_radar[:, 0:3]
    points_3d_r2c = utils.trans_point_coor(geo_radar, T_radar)
    fx, cx = intrinsic[0, 0], intrinsic[0, 2]
    fy, cy = intrinsic[1, 1], intrinsic[1, 2]
    points_2d_radar = utils.project_3d_to_2d(points_3d_r2c, fx, fy, cx, cy)
    x_coords = points_2d_radar[:, 0]
    y_coords = points_2d_radar[:, 1]
    mask_x = (x_coords >= x1) & (x_coords < x2)
    mask_y = (y_coords >= y1) & (y_coords < y2)
    mask = mask_x & mask_y
    r_in = pcd_radar[mask]
    return r_in

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

    step_size = crop_size

    for y_start in range(0, h - crop_size + 1, step_size):
        for x_start in range(0, w - crop_size + 1, step_size):
            crop = image[y_start:y_start + crop_size, x_start:x_start + crop_size]

            u_min, v_min = (x_start, y_start)
            u_max, v_max = (x_start + crop_size, y_start + crop_size)

            # 检查裁剪区域是否在给定的坐标范围内
            if (u_min >= Ru_max and v_min >= Rv_max) or (u_max <= Ru_min and v_max <= Rv_min):
                continue
            else:
            #if u_min >= Ru_min and u_max <= Ru_max and v_min >= Rv_min and v_max <= Rv_max:
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


def gen_range_image(local_points, fov_up, fov_down, yaw1, yaw2, proj_H, proj_W, dst):
    #print("len of local_points:",len(local_points))
    fov_up = fov_up / 180.0 * np.pi
    fov_down = fov_down / 180.0 * np.pi
    fov = abs(fov_down) + abs(fov_up)
    depth = np.linalg.norm(local_points[:, :3], 2, axis=1)
    scan_x = local_points[:, 0]
    scan_y = local_points[:, 1]
    scan_z = local_points[:, 2]
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)
    #proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_x = 0.5 * (np.degrees(yaw) / (abs(yaw1) + abs(yaw2)) + 1.0)
    #proj_x = (np.degrees(yaw) - yaw2) / (yaw1 - yaw2)
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
    #proj_range[proj_y, proj_x] = depth*100  #multiply depth by 100, or the color will not be able tell
    proj_range[proj_y, proj_x] = depth*100
    cv2.imwrite(dst, proj_range)

    return


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


def remove_point_if_exists(local_radar_pcd, radarpoint):
    for i, point in enumerate(local_radar_pcd):
        if np.allclose(point, radarpoint[0:3]):
            return np.delete(local_radar_pcd, i, axis=0), True
    return local_radar_pcd, False

def get_new_point(x, y, z):
    point = np.array([x, y, z])
    vector = point - np.array([0, 0, 0])
    unit_vector = vector / np.linalg.norm(vector)
    new_point = point - unit_vector * 1
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



# utils.get_intrinsic_matrix(txtfile)
if __name__ == '__main__':
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
    #     pcd_lidar = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)[:,0:4]
    #     filtered_points_2d =  radar2image(pcd_lidar,T_lidar,K)
    #     np.save(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/lidar_pixel/{str(frame)}.npy',filtered_points_2d)
    #     if frame % 10 == 0:
    #         print(f'frame {frame} saved.')




    #---------------------------------------------------make dataset----------------------------------------------------
    # column_names = ['Frame', 'Index', 'PointInside', 'LenLidar']
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
    #     pcd_lidar = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
    #     geo_lidar = utils.filter_vlp16(pcd_lidar)[:,0:3]
    #     lidar_uv = lidar2image(geo_lidar,T_lidar,K)
    #     crop_size = 100
    #     #num_crops = 10
    #     uv_radar = np.load(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/velodyne_pixel/{frame}.npy')
    #     Ru_min, Ru_max, Rv_min, Rv_max = np.min(uv_radar[:,0]), np.max(uv_radar[:,0]), np.min(uv_radar[:,1]), np.max(uv_radar[:,1])
    #     crops, coordinates = sliding_window_crop(image, crop_size, Ru_min, Ru_max, Rv_min, Rv_max)
    #     #crops, coordinates = random_crop(image, crop_size, num_crops, Ru_min, Ru_max, Rv_min, Rv_max)
    #
    #     #np.save(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/cropped_image_coor/{frame}.npy',coordinates)
    #     for i in range(len(crops)):
    #         print(f'index {i}')
    #         c = crops[i]
    #         coor = coordinates[i]
    #         u1,v1 = coor[0]
    #         u2,v2 = coor[1]
    #         yaw1, pitch1 = pixel_to_radar_angles(u1, v1, K, T_radar)
    #         yaw2, pitch2 = pixel_to_radar_angles(u2, v2, K, T_radar)
    #
    #         #only condiser pics which have overlap with LiDAR FoV
    #         yaw_l1, pitch_l1 = pixel_to_radar_angles(u1, v1, K, T_lidar)
    #         yaw_l2, pitch_l2 = pixel_to_radar_angles(u2, v2, K, T_lidar)
    #
    #         if (pitch_l1 > 15 and pitch_l2 > 15) or (pitch_l1 < -15 and pitch_l2 < -15):
    #             continue
    #
    #         print('yaw1, yaw2:',yaw1, yaw2)
    #         print('pitch1, pitch2:', pitch1, pitch2)
    #         print('coor_1:',u1,v1,'coor_2:',u2,v2)
    #         mask = (lidar_uv[:, 0] >= u1) & (lidar_uv[:, 0] <= u2) & (lidar_uv[:, 1] >= v1) & (lidar_uv[:, 1] <= v2)
    #         l_points_in_bbox = geo_lidar[mask]
    #         len_lidar = len(l_points_in_bbox)
    #         print('len of lidar points:',len_lidar)
    #         # np.save(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/cropped_lidar/{frame}_{i}.npy',l_points_in_bbox)
    #         # cv2.imwrite(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/cropped_images/{frame}_{i}.jpg',c)
    #         pin = p_in(u1,v1,u2,v2,uv_radar)
    #         if pin == 1:
    #             print('points found in cropped range')
    #             pcd_radar = np.fromfile(radar_file, dtype=np.float32).reshape(-1, 7)
    #             radar_in = radar_in_image(coor, pcd_radar, T_radar, K)
    #             #np.save(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/velodyne_pixel_radar_coor/{frame}_{i}.npy',radar_in)
    #
    #         new_data = pd.DataFrame({'Frame':[frame], 'Index':[i], 'PointInside':[pin], 'LenLidar': [len_lidar]})
    #         df = pd.concat([df, new_data], ignore_index=True)
    # df.to_csv('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/RNet_dataset_within.csv')
    # #df = pd.read_csv('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/RNet_dataset.csv')
    # count_1 = df[df['PointInside'] == 1].shape[0]
    # count_0 = df[df['PointInside'] == 0].shape[0]
    #
    # print(f"Number of 1s in 'PointInside': {count_1}")
    # print(f"Number of 0s in 'PointInside': {count_0}")





#-----------------------------------------------split dataset-----------------------------------------------------------
    # random_seed = 2024
    # # folder_path = r'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/cropped_images/'
    # # file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    # df = pd.read_csv('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/RNet_dataset_within.csv')
    # file_list = []
    # for i in range(len(df)):
    #     frame, index, pointinside, lenlidar = df.values[:,1:][i]
    #     frame, index = int(frame), int(index)
    #     file_list.append(str(frame)+"_"+str(index)+".jpg")
    #
    # train_files, test_files = train_test_split(file_list, test_size=0.3, random_state=random_seed)
    # np.save(r'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/train_files_cls_within.npy', train_files)
    # np.save(r'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/test_files_cls_within.npy', test_files)
    # print(f"训练集文件数量: {len(train_files)}")
    # print(f"测试集文件数量: {len(test_files)}")
    # # csv_file = r'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/RNet_dataset.csv'
    # # df = pd.read_csv(csv_file)
    # train_files = np.load(r'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/train_files_cls_within.npy', allow_pickle=True)
    # test_files = np.load(r'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/test_files_cls_within.npy', allow_pickle=True)
    #
    # train_files_df = pd.DataFrame(train_files, columns=['Filename'])
    # test_files_df = pd.DataFrame(test_files, columns=['Filename'])
    #
    # train_files_df[['Frame', 'Index']] = train_files_df['Filename'].apply(frame_index).apply(pd.Series)
    # test_files_df[['Frame', 'Index']] = test_files_df['Filename'].apply(frame_index).apply(pd.Series)
    #
    # df_merged = pd.merge(df, train_files_df, on=['Frame', 'Index'], how='right', suffixes=('', '_train'))
    # df_merged_test = pd.merge(df, test_files_df, on=['Frame', 'Index'], how='right', suffixes=('', '_test'))
    #
    # train_pointinside_counts = df_merged['PointInside'].value_counts()
    # test_pointinside_counts = df_merged_test['PointInside'].value_counts()
    #
    # print("训练集统计:")
    # print(f"PointInside 为 1 的数量: {train_pointinside_counts.get(1, 0)}")
    # print(f"PointInside 为 0 的数量: {train_pointinside_counts.get(0, 0)}")
    #
    # print("测试集统计:")
    # print(f"PointInside 为 1 的数量: {test_pointinside_counts.get(1, 0)}")
    # print(f"PointInside 为 0 的数量: {test_pointinside_counts.get(0, 0)}")



#balance positive/negetice samples ver
    # random_seed = 2024
    # folder_path = r'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/cropped_images/'
    # file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    # train_files, test_files = train_test_split(file_list, test_size=0.3, random_state=random_seed)
    # np.save(r'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/train_files.npy', train_files)
    # np.save(r'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/test_files.npy', test_files)
    # print(f"训练集文件数量: {len(train_files)}")
    # print(f"测试集文件数量: {len(test_files)}")
    #
    # csv_file = r'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/RNet_dataset.csv'
    # df = pd.read_csv(csv_file)
    # train_files = np.load(r'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/train_files.npy',
    #                       allow_pickle=True)
    # test_files = np.load(r'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/test_files.npy',
    #                      allow_pickle=True)
    #
    # train_files_df = pd.DataFrame(train_files, columns=['Filename'])
    # test_files_df = pd.DataFrame(test_files, columns=['Filename'])
    #
    # train_files_df[['Frame', 'Index']] = train_files_df['Filename'].apply(frame_index).apply(pd.Series)
    # test_files_df[['Frame', 'Index']] = test_files_df['Filename'].apply(frame_index).apply(pd.Series)
    #
    # df_merged = pd.merge(df, train_files_df, on=['Frame', 'Index'], how='right', suffixes=('', '_train'))
    # df_merged_test = pd.merge(df, test_files_df, on=['Frame', 'Index'], how='right', suffixes=('', '_test'))
    #
    # # 获取正负样本
    # positive_samples = df_merged[df_merged['PointInside'] == 1]
    # negative_samples = df_merged[df_merged['PointInside'] == 0]
    #
    # # 进行下采样以匹配正样本数量
    # negative_samples_downsampled = negative_samples.sample(n=len(positive_samples), random_state=random_seed)
    #
    # # 合并采样后的数据
    # balanced_train_df = pd.concat([positive_samples, negative_samples_downsampled])
    #
    # # 打乱数据顺序
    # balanced_train_df = balanced_train_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    #
    # train_pointinside_counts = balanced_train_df['PointInside'].value_counts()
    # test_pointinside_counts = df_merged_test['PointInside'].value_counts()
    #
    # print("训练集统计:")
    # print(f"PointInside 为 1 的数量: {train_pointinside_counts.get(1, 0)}")
    # print(f"PointInside 为 0 的数量: {train_pointinside_counts.get(0, 0)}")
    #
    # print("测试集统计:")
    # print(f"PointInside 为 1 的数量: {test_pointinside_counts.get(1, 0)}")
    # print(f"PointInside 为 0 的数量: {test_pointinside_counts.get(0, 0)}")
    #
    # # 保存平衡后的训练集文件
    # balanced_train_files = balanced_train_df['Filename'].values
    # np.save(r'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/balanced_train_files.npy',
    #         balanced_train_files)
    #
    # # 如果需要保存平衡后的 DataFrame
    # balanced_train_df.to_csv(
    #     r'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/balanced_train_dataset.csv',
    #     index=False)

# -----------------------------------------------split dataset----------------------------------------------------------



#--------------------------------------------fix radar in image---------------------------------------------------------
    # df = pd.read_csv('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/RNet_dataset.csv').values[:,1:]
    # for i in range(len(df)):
    #     if df[i,2] == 0:
    #         continue
    #     else:
    #         frame = df[i,0]
    #         index = df[i,1]
    #         radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(frame)
    #         #image = cv2.imread(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/cropped_images/{frame}_{index}.jpg')
    #         K = utils.get_intrinsic_matrix(calib_file)
    #         T_radar = utils.get_radar2cam(calib_file)
    #         coor = np.load(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/cropped_image_coor/{frame}.npy')[index]
    #         pcd_radar = np.fromfile(radar_file, dtype=np.float32).reshape(-1, 7)
    #         r_in = radar_in_image(coor, pcd_radar, T_radar, K)
    #         np.save(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/velodyne_pixel_radar_coor/{frame}_{index}.npy',r_in)
# --------------------------------------------fix radar in image---------------------------------------------------------




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
#     lidar_path = 'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/cropped_lidar_radarcoor/'
#     file_list = [f for f in os.listdir(lidar_path) if os.path.isfile(os.path.join(lidar_path, f))]
#     #print(file_list)
#     dst = 'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/cropped_lidar_radarcoor_image/'
#     for f in file_list:
#         pcl_file = lidar_path + f
#         lidar_points = np.load(pcl_file)
#         frame, index = frame_index(f)
#         print(f'frame {frame}, index {index}')
#         radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(frame)
#         K = utils.get_intrinsic_matrix(calib_file)
#         if K is None:
#             print(f'frame {frame} missing intrinsic matrix.')
#             continue
#         T_radar = utils.get_radar2cam(calib_file)
#         coor = np.load(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/cropped_image_coor/{frame}.npy')[index]
#         point1, point2 = coor
#         x1, y1 = point1
#         x2, y2 = point2
#         yaw1, pitch1 = pixel_to_radar_angles(x1, y1, K, T_radar)
#         yaw2, pitch2 = pixel_to_radar_angles(x2, y2, K, T_radar)
#         assert yaw1 > yaw2 and pitch1 > pitch2
#         fov_up, fov_down = pitch1, pitch2
#         #fov_up, fov_down = 15, -15
#         dst_name = dst + str(frame)+"_"+str(index)+".jpg"
#         proj_H, proj_W = 100,100
#         gen_range_image(lidar_points, fov_up, fov_down, yaw1, yaw2, proj_H, proj_W, dst_name)


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


# ----------------------------------------------make RCS regression dataset---------------------------------------------
#     df = pd.read_csv('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/RNet_dataset_within.csv').values[:,1:]
#     radar_base = 'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/velodyne_pixel_radar_coor/'
#     column_names = ['Frame', 'Index', 'x', 'y', 'z', 'v_r', 'rcs']
#     new_df = pd.DataFrame(columns=column_names)
#     for i in range(len(df)):
#         frame, index, pointinside, _ = df[i]
#         frame, index = int(frame), int(index)
#         radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(frame)
#         K = utils.get_intrinsic_matrix(calib_file)
#         if K is None:
#             print(f'frame {frame} missing intrinsic matrix.')
#             continue
#         T_radar = utils.get_radar2cam(calib_file)
#         T_lidar = utils.get_lidar2cam(lidar_calib_file)
#         if pointinside == 1:
#             radar_pcd = np.load(radar_base + str(frame)+"_"+str(index)+".npy")
#             #check if radar point is within lidar FoV!
#             geo_radar = radar_pcd[:,0:3]
#             geo_radar = utils.trans_point_coor(geo_radar, T_radar)
#             geo_radar = utils.trans_point_coor(geo_radar, np.linalg.inv(T_lidar))
#             pitch_angles = compute_pitch_angles(geo_radar)
#             mask = generate_pitch_mask(pitch_angles, -15, 15)
#             radar_pcd = radar_pcd[mask,:]
#
#             for j in range(len(radar_pcd)):
#                 x, y, z, RCS, v_r, v_r_compensated, time = radar_pcd[j]
#                 new_data = pd.DataFrame({'Frame':[frame], 'Index':[index], 'x':[x], 'y': [y], 'z': [z]
#                                          , 'v_r':[v_r], 'rcs':RCS})
#                 new_df = pd.concat([new_df, new_data], ignore_index=True)
#     new_df.to_csv('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/RCS_dataset.csv')
#     print(new_df.values)
#
#     df = pd.read_csv('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/RCS_dataset.csv').values[:,1:]
#     indices = np.arange(len(df))
#     train_indices, test_indices = train_test_split(indices, test_size=0.3, random_state=2024)
#     np.save('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/train_files_reg.npy',
#             train_indices)
#     np.save('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/test_files_reg.npy', test_indices)
#
#     print(f"训练集索引数量: {len(train_indices)}")
#     print(f"测试集索引数量: {len(test_indices)}")
# ----------------------------------------------make RCS regression dataset---------------------------------------------




# ------------------------------generate local range image for rcs regression--------------------------------------------
    df = pd.read_csv('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/RCS_dataset.csv').values[:,1:]
    sign_csv = 0
    for i in range(124639,len(df)):
        frame, index, x, y, z, v_r, rcs = df[i]
        frame, index = int(frame), int(index)
        radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(frame)
        K = utils.get_intrinsic_matrix(calib_file)
        if K is None:
            print(f'frame {frame} missing intrinsic matrix.')
            continue
    # for frame in range(5087,9931):
    #     radar_file, lidar_file, cam_file, calib_file, lida r_calib_file, txt_base_dir = utils.get_vod_dir(frame)
    #     if not os.path.exists(radar_file):
    #         print(radar_file,'not exist!')
    #         continue
        pcd_radar = np.fromfile(radar_file, dtype=np.float32).reshape(-1, 7)
        geo_radar = pcd_radar[:, 0:3]
        pcd_lidar = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
        pcd_lidar = utils.filter_vlp16(pcd_lidar)
        geo_lidar = pcd_lidar[:, 0:3]
        T_radar = utils.get_radar2cam(calib_file)
        #T_lidar = utils.get_lidar2cam(lidar_calib_file)
        # geo_lidar = utils.trans_point_coor(geo_lidar, T_lidar)
        # geo_lidar = utils.trans_point_coor(geo_lidar, np.linalg.inv(T_radar))
        # np.save(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/geo_lidar_radar_coor/{str(frame)}.npy',geo_lidar)
        # print(frame)
        geo_lidar = np.load(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/geo_lidar_radar_coor/{str(frame)}.npy')
        #geo_lidar is in radar coordinates
        kdtree1 = KDTree(geo_lidar)
        kdtree2 = KDTree(geo_radar)
        radarpoint = np.array([x,y,z])
        r = 1
        local_lidar_index = kdtree1.query_ball_point(radarpoint[0:3], r)
        local_radar_index = kdtree2.query_ball_point(radarpoint[0:3], r)
        local_lidar = geo_lidar[local_lidar_index]
        local_radar = geo_radar[local_radar_index]
        local_radar, _ = remove_point_if_exists(local_radar, radarpoint)
        local_points = None
        if len(local_lidar_index) + len(local_radar_index) > 1:  # valid
            print('local points found')
            local_points = np.concatenate((local_lidar, local_radar), axis=0)

        np.save(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/rcs_local_points/{str(i)}.npy',local_points)
        print(f'saved D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/rcs_local_points/{str(i)}.npy')


        dst_dir = 'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/SlidingWindow_Dataset/rcs_range_image/'
        name = f'{str(i)}.jpg'
        if local_points is not None:
            virtual_point = get_new_point(x, y, z)
            fov_down = 10000
            fov_up = -10000
            for j in range(len(local_points)):
                local_points[j, 0] -= virtual_point[0]
                local_points[j, 1] -= virtual_point[1]
                local_points[j, 2] -= virtual_point[2]
                x1, y1, z1 = local_points[j]
                pitch = np.degrees(np.arctan(z1 / x1))
                if pitch > fov_up:
                    fov_up = pitch
                if pitch < fov_down:
                    fov_down = pitch

            proj_H, proj_W, = 32, 128
            gen_range_image_rcs(local_points, fov_up, fov_down, proj_H, proj_W,
                            dst_dir+name)
        else:
            print(f'frame {frame}, index {index}, {i}th of rcs dataset, local_points not found')
            black_image = np.zeros((32, 128, 3), dtype=np.uint8)
            cv2.imwrite(dst_dir+name, black_image)



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








