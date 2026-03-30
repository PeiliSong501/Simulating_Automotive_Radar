import os
import math
import random
import cv2
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures
import open3d as o3d

from scipy import stats
from scipy.spatial import cKDTree, KDTree, distance_matrix
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import utils
from depth_check import (
    calculate_angle_and_depth_diff_avg_err,
    get_pitch_yaw,
    filter_lidar_points_corr_bbox,
    filter_points_in_bbox_single_box,
)
from gt_generation import pixel_to_radar_angles, radar_in_image



"""
What this script does
---------------------
This script generates 3D simulated radar point clouds for the View-of-Delft (VoD)
dataset based on sampled 2D radar point locations and LiDAR-derived depth.

Main workflow:
1. Read the estimated number of radar points for each frame.
2. Sample 2D radar locations from a predicted probability / density map.
3. Recover depth for each sampled 2D point using a LiDAR depth map.
4. Back-project the sampled image points to 3D camera coordinates.
5. Transform the 3D points from camera coordinates to radar coordinates.
6. Optionally visualize or analyze the generated points against real radar / LiDAR data.

In short, this file is used for the depth-generation stage of the radar simulation
pipeline, turning 2D sampled radar projections into 3D radar point clouds.
"""

random.seed(2025)


# ============================================================
# Reproducible path configuration
# Modify only the paths below when reproducing this script.
# ============================================================
DATA_ROOT = "D:/VoD_dataset/view_of_delft_PUBLIC"
PMAP_ROOT = os.path.join(DATA_ROOT, "radar", "training", "PMapDataset")
LIDAR_ROOT = os.path.join(DATA_ROOT, "lidar", "training", "velodyne")

# Main experiment setting
METHOD_NAME = ""    #PLEASE SET ONE

# Common input files
ESTIMATED_POINTNUM_CSV = os.path.join(
    PMAP_ROOT, "exp_2", f"estimated_pointnum_VoD_{METHOD_NAME}.csv"
)
RAW_POINTNUM_CSV = os.path.join(PMAP_ROOT, "exp_2", "estimated_pointnum_pmap_adap.csv")
PROB_DATASET_CSV = os.path.join(PMAP_ROOT, "prob_dataset.csv")
TEST_DATASET_CSV = os.path.join(PMAP_ROOT, "test_dataset.csv")
TRAIN_DATASET_CSV = os.path.join(PMAP_ROOT, "train_dataset.csv")

# Common directories
R_IN_2D_DIR = os.path.join(PMAP_ROOT, "r_in_2d_50")
R_IN_3D_DIR = os.path.join(PMAP_ROOT, "r_in_50")
LIDAR_DEPTH_DIR = os.path.join(PMAP_ROOT, "lidar_inter2d_matrix")
PMAP_IMAGE_DIR = os.path.join(PMAP_ROOT, "exp_2", f"test_image_VoD_{METHOD_NAME}")
GENERATED_3D_OUTPUT_DIR = os.path.join(
    PMAP_ROOT,  f"generated_3d_{METHOD_NAME}_lidardepth"
)

# Visualization crop region: [u_min, v_min, u_max, v_max]
IMAGE_CROP = [0, 571, 1935, 1215]
CROP_V_MIN = 571
CROP_V_MAX = 1215
CROP_U_MIN = 0
CROP_U_MAX = 1935




def get_frame_paths(frame_id):
    """Return frequently used frame-specific file paths."""
    frame_str = str(frame_id).zfill(5)
    return {
        "r_in_2d": os.path.join(R_IN_2D_DIR, f"{frame_id}.npy"),
        "r_in_3d": os.path.join(R_IN_3D_DIR, f"{frame_id}.npy"),
        "lidar_depth_npz": os.path.join(LIDAR_DEPTH_DIR, f"{frame_id}.npz"),
        "lidar_bin": os.path.join(LIDAR_ROOT, f"{frame_str}.bin"),
        "pmap_image": os.path.join(PMAP_IMAGE_DIR, f"{frame_id}.png"),
        "generated_output": os.path.join(GENERATED_3D_OUTPUT_DIR, f"{frame_id}.npy"),
    }


def filter_ground_points_ransac(radar_points):
    radar_points = np.array(radar_points)

    # Use x and y as input features, and z as the regression target.
    X = radar_points[:, :2]
    y = radar_points[:, 2]

    # Fit a ground plane with RANSAC.
    ransac = RANSACRegressor()
    ransac.fit(X, y)

    # Extract inliers (ground) and outliers (non-ground).
    inlier_mask = ransac.inlier_mask_
    ground_points = radar_points[inlier_mask]
    non_ground_points = radar_points[~inlier_mask]

    num_ground = len(ground_points)
    num_to_select = int(num_ground * 0)

    if num_to_select > 0:
        # Randomly keep a subset of ground points if needed.
        selected_ground_points = ground_points[
            np.random.choice(num_ground, num_to_select, replace=False)
        ]
    else:
        selected_ground_points = np.array([])

    if selected_ground_points.size == 0:
        combined_points = non_ground_points
    else:
        combined_points = np.vstack((non_ground_points, selected_ground_points))

    return combined_points, ground_points


def sample_lidar_points(lidar_points, step=40):
    sampled_indices = np.arange(0, len(lidar_points), step)
    sampled_points = lidar_points[sampled_indices]
    return sampled_points


def visualize_point_cloud(true_points, lidar_points, estimated_points, bboxes=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot real radar points in red.
    ax.scatter(true_points[:, 0], true_points[:, 1], true_points[:, 2], c="r", label="Radar Point Cloud", s=1)

    if lidar_points is not None:
        lidar_points = sample_lidar_points(lidar_points)
        ax.scatter(lidar_points[:, 0], lidar_points[:, 1], lidar_points[:, 2], c="g", label="LiDAR Point Cloud", s=1)

    # Plot simulated radar points in blue.
    ax.scatter(
        estimated_points[:, 0],
        estimated_points[:, 1],
        estimated_points[:, 2],
        c="b",
        label="Simulated Radar Point Cloud",
        s=1,
    )

    # Draw 3D bounding boxes if provided.
    if bboxes is not None:
        for bbox in bboxes:
            x_min, x_max, y_min, y_max, z_min, z_max = bbox
            corners = np.array([
                [x_min, y_min, z_min],
                [x_min, y_min, z_max],
                [x_min, y_max, z_min],
                [x_min, y_max, z_max],
                [x_max, y_min, z_min],
                [x_max, y_min, z_max],
                [x_max, y_max, z_min],
                [x_max, y_max, z_max],
            ])

            edges = [
                [0, 1], [1, 3], [3, 2], [2, 0],
                [4, 5], [5, 7], [7, 6], [6, 4],
                [0, 4], [1, 5], [2, 6], [3, 7],
            ]

            for edge in edges:
                ax.plot(
                    [corners[edge[0], 0], corners[edge[1], 0]],
                    [corners[edge[0], 1], corners[edge[1], 1]],
                    [corners[edge[0], 2], corners[edge[1], 2]],
                    c="g",
                )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()


def r_in_lidar_fov(radar_points, radar_points_2d, lidar_points_radar_coord):
    # Compute yaw and pitch for LiDAR points.
    yaw = np.arctan2(lidar_points_radar_coord[:, 1], lidar_points_radar_coord[:, 0])
    pitch = np.arcsin(
        lidar_points_radar_coord[:, 2] / np.linalg.norm(lidar_points_radar_coord, axis=1)
    )

    # Determine the LiDAR field-of-view range.
    yaw_min, yaw_max = np.min(yaw), np.max(yaw)
    pitch_min, pitch_max = np.min(pitch), np.max(pitch)

    # Check whether radar points fall inside the LiDAR FOV.
    radar_yaw = np.arctan2(radar_points[:, 1], radar_points[:, 0])
    radar_pitch = np.arcsin(
        radar_points[:, 2] / np.linalg.norm(radar_points[:, :3], axis=1)
    )

    valid_mask = (
        (radar_yaw >= yaw_min)
        & (radar_yaw <= yaw_max)
        & (radar_pitch >= pitch_min)
        & (radar_pitch <= pitch_max)
    )

    valid_radar_points = radar_points[valid_mask]
    valid_radar_points_2d = radar_points_2d[valid_mask]

    return valid_radar_points, valid_radar_points_2d


def chamfer_distance(radar_points, simulated_points):
    """Compute the Chamfer Distance between two point clouds."""
    dist_matrix = distance_matrix(radar_points, simulated_points)
    cd1 = np.mean(np.min(dist_matrix, axis=1))
    cd2 = np.mean(np.min(dist_matrix, axis=0))
    return cd1 + cd2


def generate_point_cloud(radar_points_2d_guess, depth_list_guess, intrinsic_matrix, radar_to_camera):
    depth_list_guess = np.array(depth_list_guess)
    intrinsic_inv = np.linalg.inv(intrinsic_matrix)

    uvs_homogeneous = np.hstack(
        [radar_points_2d_guess, np.ones((radar_points_2d_guess.shape[0], 1))]
    )
    camera_coords = np.dot(intrinsic_inv, uvs_homogeneous.T).T * depth_list_guess[:, np.newaxis]
    camera_coords_homogeneous = np.hstack(
        [camera_coords, np.ones((camera_coords.shape[0], 1))]
    )

    camera_to_radar = np.linalg.inv(radar_to_camera)
    radar_coords_homogeneous = np.dot(camera_to_radar, camera_coords_homogeneous.T).T
    radar_points = radar_coords_homogeneous[:, :3]

    return radar_points


def draw(radar_points_2d, image, color, bbox_2d=None):
    radius = 10
    thickness = -1

    for point in radar_points_2d:
        center = (int(point[0]), int(point[1]))
        cv2.circle(image, center, radius, color, thickness)

    if bbox_2d:
        umin, vmin, umax, vmax = bbox_2d
        top_left = (int(umin), int(vmin))
        bottom_right = (int(umax), int(vmax))
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

    image = image[CROP_V_MIN:CROP_V_MAX, CROP_U_MIN:CROP_U_MAX]
    cv2.imshow("Image with Points and BBox", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_with_2dbbox(radar_points_2d, radar_points_3d, image, bbox_2d=None, bbox_radar=None, save_path=None):
    radius = 6
    thickness = -1

    # Count points that fall inside both the 2D and 3D boxes.
    count_in_both = 0

    for i, point in enumerate(radar_points_2d):
        center = (int(point[0]), int(point[1]))
        point_3d = radar_points_3d[i]

        if bbox_2d:
            umin, vmin, umax, vmax = bbox_2d
            in_2d_bbox = (umin <= point[0] <= umax) and (vmin <= point[1] <= vmax)
        else:
            in_2d_bbox = False

        if bbox_radar:
            xmin, xmax, ymin, ymax, zmin, zmax = bbox_radar
            in_3d_bbox = (
                (xmin <= point_3d[0] <= xmax)
                and (ymin <= point_3d[1] <= ymax)
                and (zmin <= point_3d[2] <= zmax)
            )
        else:
            in_3d_bbox = False

        if in_2d_bbox and in_3d_bbox:
            point_color = (0, 255, 0)
            count_in_both += 1
            cv2.circle(image, center, radius, point_color, thickness)

    if bbox_2d:
        umin, vmin, umax, vmax = bbox_2d
        top_left = (int(umin), int(vmin))
        bottom_right = (int(umax), int(vmax))
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

    print(f"Number of points in both 2D bbox and 3D bbox: {count_in_both}")

    image = image[CROP_V_MIN:CROP_V_MAX, CROP_U_MIN:CROP_U_MAX]
    cv2.imshow("Image with Points and BBox", image)
    if save_path is not None:
        cv2.imwrite(save_path, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_with_rcs(radar_points_2d, rcs_list, image, color):
    min_radius = 1
    max_radius = 12

    # Map RCS values to circle radii.
    rcs_normalized = np.clip((rcs_list + 50) / 100, 0, 1)
    radii = rcs_normalized * (max_radius - min_radius) + min_radius

    thickness = -1
    for i, point in enumerate(radar_points_2d):
        center = (int(point[0]), int(point[1]))
        radius = int(radii[i])
        cv2.circle(image, center, radius, color, thickness)

    image = image[CROP_V_MIN:CROP_V_MAX, CROP_U_MIN:CROP_U_MAX]
    cv2.imshow("Image with RCS-Based Points", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_with_depth(radar_points_2d, depth_list, image, lidar_points_2d=None, lidar_depth_list=None, save_path=None):
    radius = 6
    thickness = -1
    max_depth = 50
    min_depth = 0

    depth_normalized = np.clip((depth_list - min_depth) / (max_depth - min_depth), 0, 1)
    if lidar_depth_list is not None:
        lidar_depth_normalized = np.clip(
            (lidar_depth_list - min_depth) / (max_depth - min_depth), 0, 1
        )

    for i, point in enumerate(radar_points_2d):
        center = (int(point[0]), int(point[1]))
        depth_intensity = int(255 * depth_normalized[i])
        color = (0, 0, depth_intensity)
        cv2.circle(image, center, radius, color, thickness)

    if lidar_points_2d is not None:
        radius = 3
        for j, point in enumerate(lidar_points_2d):
            center = (int(point[0]), int(point[1]))
            depth_intensity = int(255 * lidar_depth_normalized[j])
            color = (0, 0, depth_intensity)
            cv2.circle(image, center, radius, color, thickness)

            outer_radius = radius + 2
            outer_thickness = 1
            outer_color = (255, 0, 0)
            cv2.circle(image, center, outer_radius, outer_color, outer_thickness)

    image = image[CROP_V_MIN:CROP_V_MAX, CROP_U_MIN:CROP_U_MAX]
    cv2.imshow("Image with Depth-Based Points", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if save_path is not None:
        cv2.imwrite(save_path, image)


def draw_with_velo(radar_points_2d, velocity_list, image, lidar_points_2d=None, lidar_velocity_list=None, save_path=None):
    radius = 6
    thickness = -1
    velocity_list = np.array(velocity_list)
    lidar_velocity_list = np.array(lidar_velocity_list)

    max_velocity = np.max(velocity_list)
    min_velocity = np.min(velocity_list)

    velocity_normalized = np.clip((velocity_list - min_velocity) / (max_velocity - min_velocity), 0, 1)
    lidar_velocity_normalized = np.clip(
        (lidar_velocity_list - min_velocity) / (max_velocity - min_velocity), 0, 1
    )

    for i, point in enumerate(radar_points_2d):
        center = (int(point[0]), int(point[1]))
        velocity_intensity = int(255 * velocity_normalized[i])
        color = (0, 0, velocity_intensity)
        cv2.circle(image, center, radius, color, thickness)

    if lidar_points_2d is not None:
        for i, point in enumerate(lidar_points_2d):
            center = (int(point[0]), int(point[1]))
            radius = 4
            velocity_intensity = int(255 * lidar_velocity_normalized[i])
            color = (0, 0, velocity_intensity)
            cv2.circle(image, center, radius, color, thickness)

            outer_radius = radius + 2
            outer_thickness = 1
            outer_color = (255, 0, 0)
            cv2.circle(image, center, outer_radius, outer_color, outer_thickness)

    image = image[CROP_V_MIN:CROP_V_MAX, CROP_U_MIN:CROP_U_MAX]
    cv2.imshow("Image with Velocity-Based Points", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if save_path is not None:
        cv2.imwrite(save_path, image)


def compute_mean_and_covariance(points_near_bbox, name):
    mean = np.mean(points_near_bbox, axis=0)
    covariance = np.cov(points_near_bbox, rowvar=False)
    return mean, covariance


def sample_from_density(density_image_path, num_samples):
    density = cv2.imread(density_image_path, cv2.IMREAD_GRAYSCALE)
    density = density.astype(np.float32)
    density /= np.sum(density)

    flat_density = density.flatten()
    cdf = np.cumsum(flat_density)
    cdf[-1] = 1.0

    random_values = np.random.rand(num_samples)
    sampled_indices = np.searchsorted(cdf, random_values)
    sampled_coords = np.unravel_index(sampled_indices, density.shape)

    return list(zip(sampled_coords[1], sampled_coords[0] + CROP_V_MIN))


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


def process_radar_data(lidar_depth, radar_points_2d_guess, intrinsic_matrix, radar_to_camera):
    """Convert sampled 2D radar points into 3D radar coordinates using a dense LiDAR depth map."""
    point_cloud = []

    for j in range(len(radar_points_2d_guess)):
        u, v = radar_points_2d_guess[j]

        # The LiDAR depth map is cropped vertically, so use v - CROP_V_MIN.
        depth = lidar_depth[v - CROP_V_MIN, u]
        if depth == 255:
            continue

        x = (u - intrinsic_matrix[0, 2]) * depth / intrinsic_matrix[0, 0]
        y = (v - intrinsic_matrix[1, 2]) * depth / intrinsic_matrix[1, 1]
        z = depth

        point_cam = np.array([x, y, z])
        point_radar = utils.trans_point_coor(point_cam.reshape(1, 3), np.linalg.inv(radar_to_camera))
        point_cloud.append(point_radar[0])

    return np.array(point_cloud)


def process_radar_data_sparse(lidar_points_2d, lidar_depth, radar_points_2d_guess, intrinsic_matrix, radar_to_camera, thres=15):
    """
    Recover 3D radar points by matching each predicted radar image point to its nearest
    projected LiDAR point in 2D.
    """
    kdtree = cKDTree(lidar_points_2d)
    radar_points = []
    radar_points_2d_guess = np.array(radar_points_2d_guess)

    for radar_point in radar_points_2d_guess:
        dist, idx = kdtree.query(radar_point, k=1)
        if dist > thres:
            continue

        lidar_u, lidar_v = lidar_points_2d[idx]
        depth = lidar_depth[lidar_v - CROP_V_MIN, lidar_u]
        if depth == 0:
            continue

        x = (lidar_u - intrinsic_matrix[0, 2]) * depth / intrinsic_matrix[0, 0]
        y = (lidar_v - intrinsic_matrix[1, 2]) * depth / intrinsic_matrix[1, 1]
        z = depth

        point_cam = np.array([x, y, z])
        point_radar = utils.trans_point_coor(point_cam.reshape(1, 3), np.linalg.inv(radar_to_camera))
        radar_points.append(point_radar[0])

    radar_points, _ = filter_ground_points_ransac(radar_points)
    return np.array(radar_points)


def process_radar_data_kd(lidar_points_2d, lidar_depth, radar_points_2d_guess, intrinsic_matrix, radar_to_camera, thres=15):
    kdtree = KDTree(lidar_points_2d)
    radar_points = []
    radar_points_2d_guess = np.array(radar_points_2d_guess)

    for radar_point in radar_points_2d_guess:
        indices = kdtree.query_ball_point(radar_point, r=thres)
        u, v = np.array(radar_point)

        if indices:
            depth_list = []
            for idx in indices:
                lidar_u, lidar_v = lidar_points_2d[idx]
                depth = lidar_depth[lidar_v - CROP_V_MIN, lidar_u]
                depth_list.append(depth)

            avg_depth = sum(depth_list) / len(depth_list)
            x = (u - intrinsic_matrix[0, 2]) * avg_depth / intrinsic_matrix[0, 0]
            y = (v - intrinsic_matrix[1, 2]) * avg_depth / intrinsic_matrix[1, 1]
            z = avg_depth

            point_cam = np.array([x, y, z])
            point_radar = utils.trans_point_coor(point_cam.reshape(1, 3), np.linalg.inv(radar_to_camera))
            radar_points.append(point_radar[0])

    return np.array(radar_points)


def filter_lidar_depth(depth_data, bbox_cam):
    zmin, zmax = bbox_cam[-2:]
    depth_data = depth_data[(depth_data >= zmin) & (depth_data <= zmax)]
    return np.array(depth_data)


def find_closest_bbox(bbox_2d_list, bbox_cam_list):
    min_weighted_distance = float("inf")
    closest_bbox_index = -1

    for i, bbox in enumerate(bbox_2d_list):
        depth1, depth2 = bbox_cam_list[i][4], bbox_cam_list[i][5]
        weighted_distance = (depth1 + depth2) / 2

        if weighted_distance < min_weighted_distance:
            min_weighted_distance = weighted_distance
            closest_bbox_index = i

    return closest_bbox_index, weighted_distance


def process_radar_data_angle(lidar_points, radar_points_2d_guess, intrinsic_matrix, radar_to_camera, bbox_2d, bbox_3d, bbox_cam, thres_angle=1):
    pitch_lidar, yaw_lidar = get_pitch_yaw(lidar_points)
    lidar_angles = np.vstack((pitch_lidar, yaw_lidar)).T
    kdtree_lidar = cKDTree(lidar_angles)

    radar_points = []
    radar_points_2d_guess = np.array(radar_points_2d_guess)

    for radar_point in radar_points_2d_guess:
        u, v = radar_point
        radar_yaw, radar_pitch = pixel_to_radar_angles(u, v, intrinsic_matrix, radar_to_camera)
        u, v = np.array(radar_point)
        target_flag = False
        bbox_target_list, bbox_cam_target_list, bbox_2d_list, lidar_box_list = [], [], [], []

        for i in range(len(bbox_2d)):
            umin, vmin, umax, vmax = bbox_2d[i]
            u_len = umax - umin
            v_len = vmax - vmin
            u_shift, v_shift = 0.00 * u_len, 0.00 * v_len

            if umin - u_shift <= u <= umax + u_shift and vmin - v_shift <= v <= vmax + v_shift:
                target_flag = True
                bbox_target = bbox_3d[i]
                bbox_cam_target = bbox_cam[i]
                lidar_box = filter_lidar_points_corr_bbox(lidar_points, bbox_target)
                bbox_target_list.append(bbox_target)
                bbox_cam_target_list.append(bbox_cam_target)
                bbox_2d_list.append(bbox_2d[i])
                lidar_box_list.append(lidar_box)

        depth = -1
        if target_flag:
            depth_list_all_targets = []
            for i in range(len(bbox_target_list)):
                bbox_target = bbox_target_list[i]
                bbox_cam_target = bbox_cam_target_list[i]
                lidar_box = lidar_box_list[i]

                if len(lidar_box) == 0:
                    continue

                pitch_lidar_tmp, yaw_lidar_tmp = get_pitch_yaw(lidar_box)
                lidar_angles_tmp = np.vstack((pitch_lidar_tmp, yaw_lidar_tmp)).T
                kdtree = KDTree(lidar_angles_tmp)
                indices_tmp = kdtree.query_ball_point([radar_pitch, radar_yaw], r=5)

                if len(indices_tmp) == 0:
                    continue
                else:
                    lidar_used = lidar_box[indices_tmp]
                    lidar_point_cam = utils.trans_point_coor(lidar_used, radar_to_camera)
                    depth_list = lidar_point_cam[:, 2]
                    depth_list = filter_lidar_depth(depth_list, bbox_cam_target)
                    if len(depth_list) > 0:
                        depth = np.mean(depth_list)
                        depth_list_all_targets.append(depth)

            if len(depth_list_all_targets) > 0:
                depth = depth_list_all_targets[0]
            else:
                continue
        else:
            _, indices = kdtree_lidar.query([radar_pitch, radar_yaw])
            pitch_diff = np.abs(radar_pitch - pitch_lidar[indices])
            yaw_diff = np.abs(radar_yaw - yaw_lidar[indices])
            if pitch_diff <= thres_angle and yaw_diff <= thres_angle:
                lidar_point_cam = utils.transform_coor_to_radar(lidar_points[indices], np.linalg.inv(radar_to_camera))
                if indices:
                    depth = lidar_point_cam[2]

        if depth == -1:
            continue

        x = (u - intrinsic_matrix[0, 2]) * depth / intrinsic_matrix[0, 0]
        y = (v - intrinsic_matrix[1, 2]) * depth / intrinsic_matrix[1, 1]
        z = depth
        point_cam = np.array([x, y, z])

        if np.linalg.norm(point_cam) > 50:
            continue

        point_radar = utils.trans_point_coor(point_cam.reshape(1, 3), np.linalg.inv(radar_to_camera))
        radar_points.append(point_radar[0])

    return np.array(radar_points)


def draw_dis_histogram(real_points, simulated_points):
    x_real, z_real = real_points[:, 0], real_points[:, 2]
    x_sim, z_sim = simulated_points[:, 0], simulated_points[:, 2]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist2d(x_real, z_real, bins=30, cmap="Blues", alpha=0.6)
    plt.colorbar(label="Frequency")
    plt.xlabel("X Coordinate")
    plt.ylabel("Z Coordinate")
    plt.title("Distribution of Real Radar Points")

    plt.subplot(1, 2, 2)
    plt.hist2d(x_sim, z_sim, bins=30, cmap="Reds", alpha=0.6)
    plt.colorbar(label="Frequency")
    plt.xlabel("X Coordinate")
    plt.ylabel("Z Coordinate")
    plt.title("Distribution of Simulated Radar Points")

    plt.tight_layout()
    plt.show()


def is_point_near_bbox(point, bbox, threshold):
    x, y, z = point
    x_min, x_max, y_min, y_max, z_min, z_max = bbox
    return (
        (x_min - threshold <= x <= x_max + threshold)
        and (y_min - threshold <= y <= y_max + threshold)
        and (z_min - threshold <= z <= z_max + threshold)
    )


def filter_points_in_bbox(real_points, simulated_points, bbox_3d, distance_threshold=0.5):
    filtered_real = []
    for point in real_points:
        for bbox in bbox_3d:
            if is_point_near_bbox(point, bbox, distance_threshold):
                filtered_real.append(point)
                break

    filtered_simulated = []
    for point in simulated_points:
        for bbox in bbox_3d:
            if is_point_near_bbox(point, bbox, distance_threshold):
                filtered_simulated.append(point)
                break

    return np.array(filtered_real), np.array(filtered_simulated)


def filter_points_in_bbox_one(points, bbox_3d, distance_threshold=0.5):
    filtered_points = []
    for point in points:
        for bbox in bbox_3d:
            if is_point_near_bbox(point, bbox, distance_threshold):
                filtered_points.append(point)
                break
    return np.array(filtered_points)


def process_frame(i, dataframe):
    num_samples = int(dataframe.iloc[i]["Estimated_PointNum"])
    frame_id = int(dataframe.iloc[i]["Frame"])
    print(f"Processing frame {frame_id}")

    radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(frame_id)
    image = cv2.imread(cam_file)
    image2 = image.copy()
    intrinsic_matrix = utils.get_intrinsic_matrix(calib_file)
    radar_to_camera = utils.get_radar2cam(calib_file)
    lidar_to_camera = utils.get_lidar2cam(lidar_calib_file)

    frame_paths = get_frame_paths(frame_id)
    radar_points_2d = np.load(frame_paths["r_in_2d"])
    radar_points = np.load(frame_paths["r_in_3d"])[:, 0:3]
    radar_points_cam = utils.trans_point_coor(radar_points[:, :3], radar_to_camera)
    print("Number of real radar points:", len(radar_points))

    radar_points_2d_guess = sample_from_density(frame_paths["pmap_image"], num_samples)
    print("Estimated number of points:", num_samples)

    lidar_depth = np.load(os.path.join(LIDAR_DEPTH_DIR, f"{frame_id}.npy"))
    lidar_points = np.fromfile(frame_paths["lidar_bin"], dtype=np.float32).reshape(-1, 4)[:, 0:3]
    lidar_points = lidar_points[lidar_points[:, 0] > 2]
    lidar_points = utils.trans_point_coor(lidar_points, np.dot(np.linalg.inv(radar_to_camera), lidar_to_camera))

    annotations = utils.read_annotation(txt_base_dir + str(frame_id).zfill(5) + ".txt")
    if annotations is None:
        return

    bbox_3d = []
    bbox_2d = []
    bbox_cam = []
    for anno in annotations:
        bbox_camera, loc = utils.read_3dbbox(anno)
        bbox_radar = utils.transform_bbox_to_radar(bbox_camera, radar_to_camera)
        bbox_3d.append(bbox_radar)
        umin, vmin, umax, vmax = anno["Bbox"]
        bbox_2d.append([umin, vmin, umax, vmax])
        bbox_cam.append(bbox_camera)

    pcl_radar_guess = process_radar_data_angle(
        lidar_points,
        radar_points_2d_guess,
        intrinsic_matrix,
        radar_to_camera,
        bbox_2d,
        bbox_3d,
        bbox_cam,
    )
    print(f"Before filtering: {len(radar_points_2d_guess)}, after conversion: {len(pcl_radar_guess)}")

    np.save(os.path.join(PMAP_ROOT, "generated_3d", f"{frame_id}.npy"), pcl_radar_guess)


def main():
    os.makedirs(GENERATED_3D_OUTPUT_DIR, exist_ok=True)

    df_estimated = pd.read_csv(ESTIMATED_POINTNUM_CSV)

    for i in range(len(df_estimated)):
        num_samples = int(df_estimated.iloc[i]["Estimated_PointNum"])
        frame_id = int(df_estimated.iloc[i]["Frame"])

        radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(frame_id)
        image = cv2.imread(cam_file)
        image2 = image.copy()
        intrinsic_matrix = utils.get_intrinsic_matrix(calib_file)
        if intrinsic_matrix is None:
            continue

        radar_to_camera = utils.get_radar2cam(calib_file)
        lidar_to_camera = utils.get_lidar2cam(lidar_calib_file)

        frame_paths = get_frame_paths(frame_id)
        radar_points_2d = np.load(frame_paths["r_in_2d"])

        real_radar = np.load(frame_paths["r_in_3d"])[:, 0:3]
        velocity_list = np.load(frame_paths["r_in_3d"])[:, 4]
        real_radar_cam = utils.trans_point_coor(real_radar[:, :3], radar_to_camera)
        print("Number of real radar points:", len(real_radar))

        depth_list_radar = real_radar_cam[:, 2]
        radar_points_2d_guess = sample_from_density(frame_paths["pmap_image"], num_samples)
        print("Estimated number of sampled 2D points:", num_samples)

        lidar_depth = np.load(frame_paths["lidar_depth_npz"])["depth"]
        lidar_depth = lidar_depth.astype(np.float32) / 255.0 * 50.0

        simulated_radar = process_radar_data(
            lidar_depth,
            radar_points_2d_guess,
            intrinsic_matrix,
            radar_to_camera,
        )
        simulated_radar, radar_points_2d_guess = radar_in_image(
            IMAGE_CROP,
            simulated_radar,
            radar_to_camera,
            intrinsic_matrix,
        )

        simulated_radar_cam = utils.trans_point_coor(simulated_radar[:, :3], radar_to_camera)
        depth_list_sim = simulated_radar_cam[:, 2]
        print("Number of simulated radar points:", len(simulated_radar))

        np.save(frame_paths["generated_output"], simulated_radar)


if __name__ == "__main__":
    main()
