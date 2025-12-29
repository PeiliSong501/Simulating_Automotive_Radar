import cv2
import numpy as np
import pandas as pd
import utils
from scipy.spatial import KDTree
import open3d as o3d


def generate_point_cloud(r_in_2d_guess, depth_list_guess, K, T_radar):
    depth_list_guess = np.array(depth_list_guess)
    K_inv = np.linalg.inv(K)

    uvs_homogeneous = np.hstack([r_in_2d_guess, np.ones((r_in_2d_guess.shape[0], 1))])
    camera_coords = np.dot(K_inv, uvs_homogeneous.T).T * depth_list_guess[:, np.newaxis]
    camera_coords_homogeneous = np.hstack([camera_coords, np.ones((camera_coords.shape[0], 1))])

    T_camera_to_radar = np.linalg.inv(T_radar)
    radar_coords_homogeneous = np.dot(T_camera_to_radar, camera_coords_homogeneous.T).T

    radar_points = radar_coords_homogeneous[:, :3]

    return radar_points

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


def sample_from_density(density_image, num_samples):
    density = cv2.imread(density_image, cv2.IMREAD_GRAYSCALE)

    density = density.astype(np.float32)
    density /= np.sum(density)

    flat_density = density.flatten()

    cdf = np.cumsum(flat_density)
    cdf[-1] = 1.0

    random_values = np.random.rand(num_samples)
    sampled_indices = np.searchsorted(cdf, random_values)

    sampled_coords = np.unravel_index(sampled_indices, density.shape)

    return list(zip(sampled_coords[1], sampled_coords[0]+571))


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

if __name__ == '__main__':
    df = pd.read_csv('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/estimated_pointnum.csv')
    df1 = pd.read_csv('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/prob_dataset.csv')
    # frame = 39
    # image = cv2.imread(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/image_2/{str(frame).zfill(5)}.jpg')
    #
    # r_in_real = np.load(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/r_in_2d_lidarfov/{frame}.npy')
    # # r_in_sim_3d = np.load('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/generated_3d/2357.npy')
    # # radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(frame)
    # # K = utils.get_intrinsic_matrix(calib_file)
    # # fx, cx = K[0, 0], K[0, 2]
    # # fy, cy = K[1, 1], K[1, 2]
    # # print(r_in_sim_3d)
    # #r_in_sim_2d = utils.project_3d_to_2d(r_in_sim_3d, fx, fy, cx, cy)
    # r_in_2d_guess = sample_from_density(
    #     f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/test_image/{frame}.jpg', 187)
    # draw(r_in_2d_guess,image,(0,255,0))

    #-------------------------------------------generate 3D signals from 2D---------------------------------------------
    for i in range(len(df)):
        num_samples = int(df.iloc[i]['Estimated_PointNum'])
        frame = int(df.iloc[i]['Frame'])
        print(frame)
        radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(frame)
        image = cv2.imread(cam_file)
        image2 = image.copy()
        K = utils.get_intrinsic_matrix(calib_file)
        T_radar = utils.get_radar2cam(calib_file)
        T_lidar = utils.get_lidar2cam(lidar_calib_file)
        r_in_2d = np.load(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/r_in_2d/{frame}.npy')

        #r_in_3d = np.load(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/r_in/{frame}.npy')
        color1 = (0, 0, 255)
        #draw(r_in_2d, image, color1)
        r_in = np.load(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/r_in/{frame}.npy')
        r_in_cam = utils.trans_point_coor(r_in[:,:3],T_radar)
        print('len of true radar signals',len(r_in))
        rcs_list = r_in[:,3]
        depth_list = r_in_cam[:,2]

        true_dir = f'C:/Users/PerrySong_nku/Desktop/writing/{frame}_true.jpg'
        r_in_2d_guess = sample_from_density(
            f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/test_image/{frame}.jpg', num_samples)
        print('estimated len:',num_samples)
        lidar_depth = np.load(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/lidar_inter2d_matrix/{frame}.npy')


    # -------------------------------------------generate 3D signals from 2D---------------------------------------------

    #test
    # estimated_pointnum = df.values[:,1:]
    # frame = int(estimated_pointnum[0,0])
    # r_in = np.load(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/r_in/{frame}.npy')[:,:3]
    # r_in_guess = np.load(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/generated_3d/{frame}.npy')
    # print(len(r_in))
    # print(len(r_in_guess))
    # visualize_point_clouds(r_in, r_in_guess)



