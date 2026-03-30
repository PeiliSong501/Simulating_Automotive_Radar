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

"""Utility and experiment script for the Astyx HiRes2019 dataset.
"""

DEFAULT_ASTYX_BASE_DIR = os.environ.get(
    "ASTYX_BASE_DIR", "D:/Astyx dataset/dataset_astyx_hires2019"
)


def dataset_path(*parts):
    """Build a path inside the Astyx dataset root directory."""
    return os.path.join(DEFAULT_ASTYX_BASE_DIR, *parts)


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



def sample_points_from_point_cloud(point_cloud, point_cloud_2d, n):
    np.random.seed(2024)
    points = point_cloud[:, :3]
    num_points = points.shape[0]
    #
    # # randomly select the first point
    # sampled_indices = [np.random.randint(num_points)]
    # distances = np.linalg.norm(points - points[sampled_indices[-1]], axis=1)
    #
    # for _ in range(1, n):
    #     farthest_point_index = np.argmax(distances)
    #     sampled_indices.append(farthest_point_index)
    #     # update distances
    #     new_distances = np.linalg.norm(points - points[farthest_point_index], axis=1)
    #     distances = np.minimum(distances, new_distances)
    #
    # sampled_indices = np.array(sampled_indices)
    # sampled_points = point_cloud[sampled_indices, :]
    # sampled_points_2d = point_cloud_2d[sampled_indices, :]
    if n > num_points:
        n = num_points
    sampled_indices = np.random.choice(num_points, size=n, replace=False)

    # select corresponding points along with their positions in 2D projections
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
            # compute the standard deviation
            #print(i)

            # sigma_x, sigma_y = sigma_x_list[i], sigma_y_list[i]
            sigma_x, sigma_y = std_var[0], std_var[1]
            # standard deviation range
            # range_x = min(int(np.ceil(3 * sigma_x)),500)
            # range_y = min(int(np.ceil(3 * sigma_y)),500)
            left_range = max(-int(np.ceil(3 * sigma_x)), -u)
            right_range = min(int(np.ceil(3 * sigma_x)), cols - u - 1)
            top_range = max(-int(np.ceil(3 * sigma_y)), -v)
            bottom_range = min(int(np.ceil(3 * sigma_y)), rows - v - 1)

            gaussian_fn = model_gaussian((u, v), sigma_x, sigma_y)

            # apply the Gaussian distribution to neighboring pixels
            for di in range(top_range, bottom_range + 1):
                for dj in range(left_range, right_range + 1):
                    ni, nj = v + di, u + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        # compute the Gaussian probability value
                        prob = gaussian_fn(nj, ni)  # note: the input order should be (y, x)
                        radar_2d_map_pro[ni, nj] += prob
                        radar_2d_map_pro[ni, nj] = radar_2d_map_pro[ni, nj] + prob
    total = np.sum(radar_2d_map_pro)
    for i in range(radar_2d_map_pro.shape[0]):
        for j in range(radar_2d_map_pro.shape[1]):
            radar_2d_map_pro[i,j] /= total
    return radar_2d_map_pro


def get_astyx_dir(frame):
    base = DEFAULT_ASTYX_BASE_DIR
    # The dataset root can be changed through ASTYX_BASE_DIR.
    cam_base_dir = dataset_path('camera_front')
    radar_base_dir = dataset_path('radar_6455')
    calib_base_dir = dataset_path('calibration')
    object_base_dir = dataset_path('groundtruth_obj3d')
    lidar_base_dir = dataset_path('lidar_vlp16')
    radar_file = os.path.join(radar_base_dir, str(frame).zfill(6) + '.txt')
    lidar_file = os.path.join(lidar_base_dir, str(frame).zfill(6) + '.txt')
    cam_file = os.path.join(cam_base_dir, str(frame).zfill(6) + '.jpg')
    calib_file = os.path.join(calib_base_dir, str(frame).zfill(6) + '.json')
    object_file = os.path.join(object_base_dir, str(frame).zfill(6) + '.json')
    return radar_file, lidar_file, cam_file, calib_file, object_file


def get_transform_matrix(calib_file, sensor_uid_A, sensor_uid_B):
    #'radar_6455',"lidar_vlp16",'camera_front'
    with open(calib_file, 'r') as f:
        calibration_data = json.load(f)

    # find the calibration data for sensor A and sensor B
    T_A_to_ref = None
    T_B_to_ref = None

    for sensor in calibration_data['sensors']:
        if sensor['sensor_uid'] == sensor_uid_A:
            T_A_to_ref = np.array(sensor['calib_data']['T_to_ref_COS'])
        elif sensor['sensor_uid'] == sensor_uid_B:
            T_B_to_ref = np.array(sensor['calib_data']['T_to_ref_COS'])

    # check whether the calibration data for both sensors was found
    if T_A_to_ref is None or T_B_to_ref is None:
        return None

    # compute the transformation matrix from sensor A to sensor B
    T_A_to_B = np.linalg.inv(T_B_to_ref) @ T_A_to_ref

    return T_A_to_B


def read_txt_data(file_path):
    #radar:x,y,z,v,r,mag
    #lidar:X Y Z Reflectivity LaserID Timestamp
    txt_data = np.genfromtxt(file_path, delimiter=' ', skip_header=1)
    txt_data = txt_data[~np.isnan(txt_data).any(axis=1)]

    return txt_data

def process_frame(frame):
    if os.path.exists(dataset_path('pmap_image', f'{frame}.jpg')):
        return

    print(f'Processing frame {frame}')
    radar_file,lidar_file,cam_file,calib_file,object_file = get_astyx_dir(frame)
    K = get_camera_intrinsics(calib_file)


    image = cv2.imread(cam_file)
    coor = [0,0,2048,618]

    r_in = np.load(dataset_path('r_in', f'{frame}.npy'))
    r_in_2d = np.load(dataset_path('r_in_2d', f'{frame}.npy'))
    radar_2d_map = np.zeros((618, 2048))

    standard_variance = [100, 100]
    radar_2d_map_pro = update_radar_map_with_gaussian(radar_2d_map, r_in, r_in_2d, standard_variance)

    np.save(dataset_path('radar_2d_map_pro', f'{frame}.npy'),
            radar_2d_map_pro)

    normalized_array = cv2.normalize(radar_2d_map_pro, None, 0, 255, cv2.NORM_MINMAX)
    gray_image_pro = normalized_array.astype(np.uint8)
    cv2.imwrite(dataset_path('pmap_image', f'{frame}.jpg'),
                gray_image_pro)


def calculate_height_and_angles(lidar_pcd):
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
    # read calibration
    with open(calibration_file, 'r') as f:
        calibration_data = json.load(f)

    for sensor in calibration_data['sensors']:
        if sensor['sensor_uid'] == 'camera_front':
            if 'K' in sensor['calib_data']:
                # return camera intrinsics
                return np.array(sensor['calib_data']['K'])

    return None

def estimate_radar_velocity_linear(pcd_static, min_samples=50, residual_threshold= 0.2, max_trials=3000):
    #pcd_static = pcd_static[pcd_static[:,3]<0]
    positions = pcd_static[:, :3]  # obtain x, y, z coords
    vr = pcd_static[:, 3]  # obtain v_r

    norms = np.linalg.norm(positions, axis=1)
    norms[norms < 1e-6] = 1  # avoid devided by zero
    normalized_positions = -positions / norms[:, np.newaxis]

    # Least Squares
    model = LinearRegression(fit_intercept=False) 
    model.fit(normalized_positions, vr)

    # obtain estimated radar velocity
    estimated_radar_velocity = model.coef_

    return estimated_radar_velocity

def read_json_file(file_name):
    with open(file_name, 'r') as file:
        data = json.load(file)
    return data['objects']


def is_point_in_bbox(center, dimensions, orientation_quat, point):
    point = point[:3]
    rotation = R.from_quat(orientation_quat)
    local_point = rotation.inv().apply(np.array(point) - np.array(center))

    half_dimensions = np.array(dimensions) / 2.0

    if (np.abs(local_point) <= half_dimensions).all():
        return True
    else:
        return False



def filter_points_in_bbox(point_cloud, bbox_location, bbox_dimensions, quat1, quat2, quat3, quat4):

    bbox_center = np.array(bbox_location)
    bbox_height, bbox_length, bbox_width = bbox_dimensions
    quat = [quat1, quat2, quat3, quat4]
    rotation = R.from_quat(quat).as_matrix()
  
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
    # set the radius of the circle
    radius = 5 
    thickness = -1 

    vel_normalized = np.clip(velocity_list / np.max(velocity_list), 0, 1)

    for i, point in enumerate(r_in_2d):
        center = (int(point[0]), int(point[1]))

        # the greater the value of velocity is, the deeper the color
        color_intensity = int(vel_normalized[i] * 255)  
        adjusted_color = (int(color[0] * color_intensity / 255),
                          int(color[1] * color_intensity / 255),
                          int(color[2] * color_intensity / 255))

        cv2.circle(image, center, radius, adjusted_color, thickness)

    # save image
    cv2.imwrite(dataset_path('projection_velocity', f'{frame}.jpg'), image)


if __name__ == '__main__':

    #----------------------example of using the functions----------------------------
    radar_file,lidar_file,cam_file,calib_file,object_file = get_astyx_dir(1)
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
    fov_horizontal = np.radians(110)  # horizontal fov 
    fov_vertical = np.radians(10)  # vertical fov
    angle_threshold = np.radians(1.5)
    # calculate horizontal and vertical angles of each point
    x, y, z = radar_geo[:, 0], radar_geo[:, 1], radar_geo[:, 2]
    from sklearn.cluster import DBSCAN
    # horizontal_angles = np.arctan2(y, x)
    # vertical_angles = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))
    #
    # # filter out points outside the FoV
    # # horizontal angle range [-fov_h / 2, fov_h / 2], vertical angle range [-fov_v / 2, fov_v / 2]
    # valid_indices = np.where(
    #     (horizontal_angles >= -fov_horizontal / 2) & (horizontal_angles <= fov_horizontal / 2) &
    #     (vertical_angles >= -fov_vertical / 2) & (vertical_angles <= fov_vertical / 2)
    # )[0]
    #
    # # keep only points inside the FoV
    # radar_geo_filtered = radar_geo[valid_indices]
    # horizontal_angles_filtered = horizontal_angles[valid_indices]
    # vertical_angles_filtered = vertical_angles[valid_indices]
    #
    # # use KDTree for fast neighborhood queries
    # tree = KDTree(np.vstack((horizontal_angles_filtered, vertical_angles_filtered)).T)
    #
    # # compute the local density for each point
    # local_densities = []
    # for i, angle in enumerate(np.vstack((horizontal_angles_filtered, vertical_angles_filtered)).T):
    #     # query the number of neighboring points within angle_threshold
    #     neighbors = tree.query_ball_point(angle, angle_threshold)
    #     local_densities.append(len(neighbors))
    #
    # # store the local densities as an array
    # local_densities = np.array(local_densities)
    #
    # # detect boundary positions with significant density changes
    # # points whose local density is lower than the surrounding average can be treated as boundary points
    # density_threshold = np.mean(local_densities) - np.std(local_densities)
    # boundary_indices = np.where(local_densities < density_threshold)[0]
    #
    # horizontal_boundaries = len(set(horizontal_angles_filtered[boundary_indices]))
    # vertical_boundaries = len(set(vertical_angles_filtered[boundary_indices]))
    #
    # # estimate angle resolution
    # horizontal_resolution = fov_horizontal / horizontal_boundaries if horizontal_boundaries > 1 else fov_horizontal
    # vertical_resolution = fov_vertical / vertical_boundaries if vertical_boundaries > 1 else fov_vertical
    #
    # # output estimated angle resolution
    # print("Estimated Horizontal Resolution (radians):", horizontal_resolution)
    # print("Estimated Vertical Resolution (radians):", vertical_resolution)
    # print("Estimated Horizontal Resolution (degrees):", np.degrees(horizontal_resolution))
    # print("Estimated Vertical Resolution (degrees):", np.degrees(vertical_resolution))


    min_points_in_cluster = 5  # minimum number of points per cluster to ensure sufficient local density
    horizontal_angles = np.arctan2(y, x)  # horizontal angle; arctan2(y, x) gives the angle in the x-y plane
    vertical_angles = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))  # vertical angle

    # filter out points outside the FoV
    valid_horizontal_indices = np.where(
        (horizontal_angles >= -fov_horizontal / 2) & (horizontal_angles <= fov_horizontal / 2)
    )[0]
    valid_vertical_indices = np.where(
        (vertical_angles >= -fov_vertical / 2) & (vertical_angles <= fov_vertical / 2)
    )[0]

    # keep only points inside the FoV
    valid_indices = np.intersect1d(valid_horizontal_indices, valid_vertical_indices)
    radar_geo_filtered = radar_geo[valid_indices]
    horizontal_angles_filtered = horizontal_angles[valid_indices]
    vertical_angles_filtered = vertical_angles[valid_indices]

    # use DBSCAN for clustering
    # cluster horizontal angles
    dbscan_horizontal = DBSCAN(eps=0.005, min_samples=min_points_in_cluster)
    labels_horizontal = dbscan_horizontal.fit_predict(horizontal_angles_filtered.reshape(-1, 1))

    # cluster vertical angles
    dbscan_vertical = DBSCAN(eps=0.005, min_samples=min_points_in_cluster)
    labels_vertical = dbscan_vertical.fit_predict(vertical_angles_filtered.reshape(-1, 1))


    # compute the angular resolution in the horizontal and vertical directions
    def calculate_resolution(labels, angles_filtered):
        # find all valid clusters (exclude noise points, where -1 denotes noise)
        valid_clusters = np.unique(labels[labels != -1])
        cluster_sizes = []
        cluster_angles = []

        for cluster in valid_clusters:
            # get the points in this cluster
            cluster_points = angles_filtered[labels == cluster]

            # compute the angular range of this cluster
            angle_range = np.max(cluster_points) - np.min(cluster_points)

            # record the angular range of this cluster
            cluster_sizes.append(len(cluster_points))
            cluster_angles.append(angle_range)

        # compute the estimated angular resolution
        if len(valid_clusters) > 0:
            avg_angle_resolution = np.mean(cluster_angles) / len(valid_clusters)
        else:
            avg_angle_resolution = fov_vertical  # if there are no valid clusters, use the full FoV as the angular resolution

        return avg_angle_resolution


    # horizontal angular resolution
    horizontal_resolution = calculate_resolution(labels_horizontal, horizontal_angles_filtered)
    # vertical angular resolution
    vertical_resolution = calculate_resolution(labels_vertical, vertical_angles_filtered)

    # print results
    print("Estimated Horizontal Resolution (radians):", horizontal_resolution)
    print("Estimated Vertical Resolution (radians):", vertical_resolution)
    print("Estimated Horizontal Resolution (degrees):", np.degrees(horizontal_resolution))
    print("Estimated Vertical Resolution (degrees):", np.degrees(vertical_resolution))



    #------------------------------Stats of FoV--------------------------------------------
    # lidar_yaw_min, lidar_yaw_max = 1000, -1000
    # lidar_pitch_min, lidar_pitch_max = 1000, -1000
    # for frame in range(0,546):
    #     print('frame',frame)
    #     radar_file,lidar_file,cam_file,calib_file,object_file = get_astyx_dir(frame)
    #     if not (os.path.exists(radar_file) and os.path.exists(lidar_file) and os.path.exists(cam_file) and os.path.exists(calib_file)):
    #         print('file missing at frame ',frame)
    #         continue
    #
    #     lidar_pcd = read_txt_data(lidar_file)
    #     lidar_pcd = lidar_pcd[lidar_pcd[:, 0] > 0][:, 0:3]
    #     max_height, min_height, yaw_range, pitch_range = calculate_height_and_angles(lidar_pcd)
    #
    #     #print(f"maximum height: {max_height:.2f}, minimum height: {min_height:.2f}")
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
    #     print(f"Range of Yaw (Degree): {yaw_range_degrees[0]:.2f}° to {yaw_range_degrees[1]:.2f}°")
    #     print(f"Range of Pitch (Degree): {pitch_range_degrees[0]:.2f}° to {pitch_range_degrees[1]:.2f}°")
    # print('Final:')
    # print(f"Range of Yaw (Degree): {lidar_yaw_min:.2f}° to {lidar_yaw_max:.2f}°")
    # print(f"Range of Pitch (Degree): {lidar_pitch_min:.2f}° to {lidar_pitch_max:.2f}°")


#----------------------------------------------projection---------------------------------------------------------------
    for frame in range(0,546):
        print(f'frame:{frame}')
        radar_file, lidar_file, cam_file, calib_file, object_file = get_astyx_dir(frame)
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
        # np.save(dataset_path('l_in_2d', f'{frame}.npy'),l_in_2d)
        # np.save(dataset_path('l_in', f'{frame}.npy'),l_in)
        r_in, r_in_2d = utils1.radar_in_image(coor, radar_pcd, T_radar,K)
        np.save(dataset_path('r_in_2d', f'{frame}.npy'),r_in_2d)
        np.save(dataset_path('r_in', f'{frame}.npy'),r_in)
        velocity_list = r_in[:,3]
        print(f'len(radar_pcd):{len(radar_pcd)}')
        print(f'len(r_in):{len(r_in)}')
        draw_with_velo(r_in_2d, velocity_list, image, [0,0,255],frame)

# #--------------------------------------------make dataset---------------------------------------------------------------
    column_names = ['Frame', 'PointNum', 'Velocity']
    new_df = pd.DataFrame(columns=column_names)
    for frame in range(0,546):
        print(f'frame:{frame}')
        radar_file, lidar_file, cam_file, calib_file, object_file = get_astyx_dir(frame)
        data = read_json_file(object_file)
        #radar_pcd = read_txt_data(radar_file)
        radar_pcd = np.load(dataset_path('r_in', f'{frame}.npy'))
        print(len(radar_pcd))
        dynamic_pcd, static_pcd = classify_points(radar_pcd,data)
        print(f'len(dynamic):{len(dynamic_pcd)},len(static):{len(static_pcd)}')
        ego_velo = estimate_radar_velocity_linear(static_pcd)
        print(ego_velo,np.linalg.norm(ego_velo))
        r_in = np.load(dataset_path('r_in', f'{frame}.npy'))
        num_points = len(r_in)
        print('pointnum',num_points)
        new_data = pd.DataFrame({'Frame': [frame], 'PointNum': [num_points], 'Velocity': [np.linalg.norm(ego_velo)]})
        new_df = pd.concat([new_df, new_data], ignore_index=True)
    new_df.to_csv(dataset_path('prob_dataset.csv'), index=False)



    #----------------------------------------------distribution map-----------------------------------------------------
    for frame in range(0,546):
        print(frame)
        radar_file, lidar_file, cam_file, calib_file, object_file = get_astyx_dir(frame)
        image = cv2.imread(cam_file)
        r_in_2d = np.load(dataset_path('r_in_2d', f'{frame}.npy'))
        new_image = utils.visualize_with_image_color(image, r_in_2d, [0, 0, 255])
        cv2.imwrite(dataset_path('projection_radar', f'{frame}.jpg'),new_image)

    frames = range(0, 546)
    
    # use joblib for parallel processing
    Parallel(n_jobs=8)(delayed(process_frame)(frame) for frame in frames)




    #------------------------------------------------split dataset-----------------------------------------------------
    random_seed = 2024
    np.random.seed(random_seed)
    
    # read the dataset
    df = pd.read_csv(dataset_path('prob_dataset.csv'))
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # custom split ratio
    train_ratio = 0.7  # training set takes 70%
    test_ratio = 0.3  # test set takes 30%
    
    # ensure the ratios sum to 1
    assert train_ratio + test_ratio == 1.0
    
    # compute the split index
    train_index = int(train_ratio * len(df))
    
    # split the dataset
    train_df = df[:train_index]
    test_df = df[train_index:]
    
    # print the size of each split
    print(f"Training set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    
    # save the split datasets (optional)
    train_df.to_csv(dataset_path('train_dataset.csv'),
                    index=False)
    test_df.to_csv(dataset_path('test_dataset.csv'), index=False)
    
    # check the minimum and maximum values of the PointNum column
    print(np.min(train_df['PointNum'].values), np.max(train_df['PointNum'].values))





    #-----------------------------------generate dynamic objects dataframe----------------------------------------------
    # column_names = ['Frame', 'Track_ID', 'Class', 'Rotation1', 'Rotation2', 'Rotation3', 'Rotation4', 'Location_x', 'Location_y', 'Location_z',
    #                 'Dimension_x', 'Dimension_y', 'Dimension_z']
    # df = pd.DataFrame(columns=column_names)
    # for frame in range(0, 546):
    #     radar_file, lidar_file, cam_file, calib_file, object_file = get_astyx_dir(frame)
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
    # df.to_csv(dataset_path('dynamic_objects_total.csv'))






    #-------------------------------generate dynamic objects subset-----------------------------------------------------
    # RCS_test = pd.read_csv(dataset_path('RCS_dataset_test.csv'))
    # dynamic_object = pd.read_csv(dataset_path('dynamic_objects_total.csv'))
    # matched_objects = []
    #
    #
    # def find_matching_objects(RCS_data, dynamic_object):
    #     grouped_RCS_data = RCS_data.groupby('Frame')
    #     grouped_dynamic_object = dynamic_object.groupby('Frame')
    #
    #     total_frames = len(grouped_RCS_data)
    #     checkpoint = total_frames // 10  # report progress every 10% of the processed frames
    #
    #     for idx, (frame, frame_RCS_points) in enumerate(grouped_RCS_data):
    #         if frame in grouped_dynamic_object.groups:  # process only matching frames
    #             frame_dynamic_objects = grouped_dynamic_object.get_group(frame)
    #
    #             for _, point in frame_RCS_points.iterrows():
    #                 point_cloud = np.array([[point['x'], point['y'], point['z']]])  # get the point coordinates
    #                 point_classified = False
    #
    #                 # iterate over the dynamic-object bounding boxes in the same frame
    #                 for _, obj in frame_dynamic_objects.iterrows():
    #                     bbox_location = [obj['Location_x'], obj['Location_y'], obj['Location_z']]
    #                     bbox_dimensions = [obj['Dimension_x'], obj['Dimension_y'], obj['Dimension_z']]
    #                     quat1, quat2, quat3, quat4 = obj['Rotation1'], obj['Rotation2'], obj['Rotation3'], obj[
    #                         'Rotation4']
    #
    #                     # check whether the point is inside the bounding box
    #                     if len(filter_points_in_bbox(point_cloud, bbox_location, bbox_dimensions, quat1, quat2, quat3,
    #                                                  quat4)[0]) > 0:
    #                         # match found; save the object information
    #                         matched_objects.append(obj)
    #                         point_classified = True
    #                         break  # once the point is classified, no further checks are needed
    #
    #         # print progress every 10% of the processed frames
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
    # output_csv_path = dataset_path('matched_dynamic_objects_subset.csv')
    # matched_objects_df.to_csv(output_csv_path, index=False)
    #
    # print(f"Subset of matched dynamic objects saved to {output_csv_path}")


    # df = pd.read_csv(
    #     dataset_path('matched_dynamic_objects_subset.csv')).iloc[:, 1:]
    # df['valid'] = 0  # initialize the valid column
    # # iterate over the data and compute the valid column
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
    #     # compute the valid value
    #     if len(r_in_local) < 3:
    #         df.at[i, 'valid'] = 0
    #     else:
    #         df.at[i, 'valid'] = 1
    #         # save r_in_local
    #         np.save(
    #             dataset_path('rcs_dis_test', 'object_local_points', f'{Frame}_{Track_ID}.npy'),
    #             r_in_local)
    #         r_in_2d_local = r_in_2d[mask]
    #
    # # save the updated DataFrame
    # df.to_csv(dataset_path('matched_dynamic_objects_subset_new.csv'), index=False)




# ------------------------------------------make RSS(RCS) regression dataset-------------------------------------------

    df = pd.read_csv(dataset_path('RCS_dataset_test.csv')).values[:,1:]
    for i in range(len(df)):
        # frame, pointnum, ego_velocity = train_df[i]
        # frame, pointnum = int(frame), int(pointnum)
        # print('frame', frame)
        # radar_file, lidar_file, cam_file, calib_file, object_file = get_astyx_dir(frame)
        # K = get_camera_intrinsics(calib_file)
        # if K is None:
        #     print(f'frame {frame} missing intrinsic matrix.')
        #     continue
        # T_lidar2radar = get_transform_matrix(calib_file, 'lidar_vlp16', 'radar_6455')
        #
        # r_in = np.load(dataset_path('r_in', f'{frame}.npy'))
        # r_in_2d = np.load(dataset_path('r_in_2d', f'{frame}.npy'))
        # mask = radar_in_lidarfov(r_in,np.linalg.inv(T_lidar2radar))
        # l_in = np.load(
        #     dataset_path('l_in', f'{frame}.npy'))
        # l_in_radar_coor = utils.trans_point_coor(l_in, T_lidar2radar)
        frame, index,x,y,z,u,v,v_r,rcs = df[i]
        l_in_radar_coor = np.load(dataset_path('l_in_radar_coor', f'{int(frame)}.npy'))
        kdtree = KDTree(l_in_radar_coor)
        #n = 100
        #sampled_points, sampled_points_2d = sample_points_from_point_cloud(r_in, r_in_2d, n)
    
        # for j in range(0, min(n,len(r_in))):
        #     x, y, z, v_r, rcs = sampled_points[j, 0:5]
        #     u, v = sampled_points_2d[j, :]
    
        r = 1
    
        pitch = np.degrees(np.arctan(z / np.linalg.norm([x, y], 2)))
    
        local_lidar_index = kdtree.query_ball_point([x, y, z], r)
    
    
        local_lidar_radarcoor = l_in_radar_coor[local_lidar_index]
        np.save(dataset_path('local_points', f'{int(frame)}_{int(index)}.npy'), local_lidar_radarcoor)
        #local_lidar_radarcoor = np.load(dataset_path('local_points', f'{int(frame)}_{int(index)}.npy'))
        dst = dataset_path('ablation', 'range_image', f'{int(frame)}_{int(index)}.jpg')
        fov_down = -15
        fov_up = 15
        # for k in range(len(local_lidar_radarcoor)):
        #     local_lidar_radarcoor[k, 0] -= virtual_point[0]
        #     local_lidar_radarcoor[k, 1] -= virtual_point[1]
        #     local_lidar_radarcoor[k, 2] -= virtual_point[2]
        #     x1, y1, z1 = local_lidar_radarcoor[k]
        #     pitch = np.degrees(np.arctan(z1 / np.linalg.norm([x1, y1], 2)))
        #     if pitch > fov_up:
        #         fov_up = pitch
        #     if pitch < fov_down:
        #         fov_down = pitch
    
        proj_H, proj_W, = 32, 128
        gen_range_image_rcs(local_lidar_radarcoor, fov_up, fov_down, proj_H, proj_W,
                            dst)
