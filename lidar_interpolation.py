
"""LiDAR interpolation utilities and batch generation script.
"""

import math
import os
import random
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils
from matplotlib import cm
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull, Delaunay, KDTree
from shapely import geometry
from shapely.geometry import MultiLineString, MultiPoint, Polygon
from shapely.ops import polygonize, unary_union
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split

DEFAULT_VOD_BASE_DIR = os.environ.get(
    "VOD_BASE_DIR",
    "/workspace/data/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset",
)


def vod_path(*parts: str) -> str:
    """Build a path under the VoD PMapDataset directory."""
    return os.path.join(DEFAULT_VOD_BASE_DIR, *parts)


def alpha_shape(points, alpha):
    """Compute the alpha shape (concave hull) of a set of points."""
    if len(points) < 4:
        return MultiPoint(list(points)).convex_hull

    tri = Delaunay(points)
    triangles = points[tri.simplices]
    a = np.linalg.norm(triangles[:, 0] - triangles[:, 1], axis=1)
    b = np.linalg.norm(triangles[:, 1] - triangles[:, 2], axis=1)
    c = np.linalg.norm(triangles[:, 2] - triangles[:, 0], axis=1)
    s = (a + b + c) / 2.0
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))
    circum_r = a * b * c / (4.0 * area)
    filtered = triangles[circum_r < 1.0 / alpha]

    edge_points = np.concatenate([filtered[:, [0, 1]], filtered[:, [1, 2]], filtered[:, [2, 0]]])
    edge_points = np.unique(edge_points.reshape(-1, 2), axis=0)
    m = MultiLineString(edge_points)

    triangles = list(polygonize(m))
    return unary_union(triangles)


def transform_coor_to_radar(location, radar_transform_matrix):
    """Transform a 3D point with a given homogeneous transform matrix."""
    location_homogeneous = np.array([location[0], location[1], location[2], 1])
    location_radar_homogeneous = np.dot(radar_transform_matrix, location_homogeneous)
    location_radar = location_radar_homogeneous[:3]
    return location_radar


def radar2image(pcd_radar, T_radar, intrinsic):
    """Project radar points into image coordinates."""
    shape_image = [1936, 1216]
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


def radar_in_image(coor, pcd_radar, T_radar, intrinsic):
    """Keep only projected radar points inside a given image crop."""
    x1, y1, x2, y2 = coor
    geo_radar = pcd_radar[:, 0:3]

    # Transform radar points into the camera coordinate system.
    points_3d_r2c = utils.trans_point_coor(geo_radar, T_radar)

    # Camera intrinsics.
    fx, cx = intrinsic[0, 0], intrinsic[0, 2]
    fy, cy = intrinsic[1, 1], intrinsic[1, 2]

    # Project 3D points to the 2D image plane.
    points_2d_radar = utils.project_3d_to_2d(points_3d_r2c, fx, fy, cx, cy)

    # Keep only the points inside the target coordinate range.
    x_coords = points_2d_radar[:, 0]
    y_coords = points_2d_radar[:, 1]
    mask_x = (x_coords >= x1) & (x_coords < x2)
    mask_y = (y_coords >= y1) & (y_coords < y2)
    mask = mask_x & mask_y

    # Filter both the original radar points and their 2D projections.
    r_in = pcd_radar[mask]
    r_in_2d = points_2d_radar[mask]

    # Convert projected 2D points to integer pixel coordinates.
    r_in_2d_int = np.floor(r_in_2d).astype(int)

    # Remove duplicate pixels by keeping the point with the smallest depth.
    unique_points = {}

    for i, (x, y) in enumerate(r_in_2d_int):
        depth = r_in[i, 0]
        key = (x, y)

        if key in unique_points:
            if depth < unique_points[key][0]:
                unique_points[key] = (depth, i)
        else:
            unique_points[key] = (depth, i)

    indices = [v[1] for v in unique_points.values()]
    r_in_final = r_in[indices]
    r_in_2d_final = r_in_2d_int[indices]

    return r_in_final, r_in_2d_final


def radar_in_lidarfov(pcd_radar, T_radar, T_lidar):
    """Filter radar points that fall inside the LiDAR vertical field of view."""
    mask = []
    for i in range(len(pcd_radar)):
        geo = pcd_radar[i][0:3]
        geo_lidarcoor = transform_coor_to_radar(geo, np.dot(np.linalg.inv(T_lidar), T_radar))
        pitch = np.arctan2(geo_lidarcoor[2], np.linalg.norm(geo_lidarcoor[:2]))
        pitch_deg = np.degrees(pitch)
        if -15 < pitch_deg < 15:
            mask.append(i)
    radar = pcd_radar[mask, :]
    return radar


def lidar_in_radarfov(pcd_lidar, T_radar, T_lidar):
    """Filter LiDAR points that fall inside the radar field of view."""
    mask = []
    pcd_lidar = utils.filter_vlp16(pcd_lidar)
    pcd_lidar = pcd_lidar[pcd_lidar[:, 0] > 0]
    for i in range(len(pcd_lidar)):
        geo = pcd_lidar[i][0:3]
        geo_radarcoor = transform_coor_to_radar(geo, np.dot(np.linalg.inv(T_radar), T_lidar))
        yaw = np.arctan2(geo_radarcoor[1], geo_radarcoor[0])
        pitch = np.arctan2(geo_radarcoor[2], np.linalg.norm(geo_radarcoor[:2]))
        yaw_deg = np.degrees(yaw)
        pitch_deg = np.degrees(pitch)
        if -15 < pitch_deg < 15 and -60 < yaw_deg < 60:
            mask.append(i)

    lidar = pcd_lidar[mask, :]
    return lidar


def lidar2image(pcd_lidar, T_lidar, intrinsic):
    """Project LiDAR points into image coordinates."""
    shape_image = [1936, 1216]
    geo_lidar = pcd_lidar[:, 0:3]
    points_3d_r2c = utils.trans_point_coor(geo_lidar, T_lidar)
    fx, cx = intrinsic[0, 0], intrinsic[0, 2]
    fy, cy = intrinsic[1, 1], intrinsic[1, 2]
    points_2d_lidar = utils.project_3d_to_2d(points_3d_r2c, fx, fy, cx, cy)
    return points_2d_lidar


def lidar2uv_depth(pcd_lidar, T_lidar, intrinsic, coor):
    """Build a small local depth patch from projected LiDAR points."""
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
        if mask[i] is True:
            u, v = points_2d_lidar[i]
            depth = geo_lidar[i, 0]
            u_new, v_new = int(np.floor(u - x1)), int(np.floor(v - y1))
            if lidar_uv_depth[u_new, v_new] == -1:
                lidar_uv_depth[u_new, v_new] = depth
                cnt_depth += 1
            else:
                if depth < lidar_uv_depth[u_new, v_new]:
                    lidar_uv_depth[u_new, v_new] = depth
    return lidar_uv_depth, cnt_depth


def pixel_to_radar_angles(u, v, K, T):
    """Convert a pixel location to radar yaw/pitch angles."""
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


def cluster_points_2d(points):
    """Cluster 2D points with DBSCAN."""
    clustering = DBSCAN(eps=0.5, min_samples=5).fit(points)
    labels = clustering.labels_
    return labels


def generate_polygon(points):
    """Generate a convex hull polygon from a set of points."""
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    return hull_points


def compute_min_distances(points):
    """Compute pairwise distances and print the minimum-distance point pair."""
    num_points = len(points)
    min_distance = float("inf")
    min_pair = None

    for i in range(num_points):
        for j in range(i + 1, num_points):
            dist = np.linalg.norm(points[i] - points[j])

            if dist < min_distance:
                min_distance = dist
                min_pair = (points[i], points[j])

                print(f"Updated minimum distance: {min_distance}")
                print(f"Point pair: {min_pair[0]} and {min_pair[1]}")

    print(f"Final minimum distance: {min_distance}")
    print(f"Point pair with minimum distance: {min_pair[0]} and {min_pair[1]}")


def interpolate_depth(image_shape, mask, depth_points, depth_values, search_radius=10):
    """Interpolate sparse depth values inside a binary mask using local neighbors."""
    depth_map = np.full(image_shape, -1, dtype=np.float32)

    # Convert coordinates from [x, y] to [y, x].
    depth_points_transformed = np.array([(y, x) for x, y in depth_points])

    # Place the original valid depth values directly into the depth map.
    for (x, y), depth in zip(depth_points_transformed, depth_values):
        if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
            depth_map[y, x] = depth

    # Build a KDTree over valid depth points.
    kdtree = KDTree(depth_points)

    # Traverse each pixel inside the mask and interpolate missing depth values.
    for y in range(image_shape[0]):
        for x in range(image_shape[1]):
            if mask[y, x] == 1 and depth_map[y, x] == -1:
                neighbors = kdtree.query_ball_point([x, y], search_radius)
                if len(neighbors) > 0:
                    dist_weights = []
                    depths = []
                    for idx in neighbors:
                        depth_x, depth_y = depth_points_transformed[idx]
                        dist = np.sqrt((x - depth_x) ** 2 + (y - depth_y) ** 2)
                        if dist == 0:
                            dist = 1e-5
                        weight = 1.0 / dist
                        dist_weights.append(weight)
                        depths.append(depth_values[idx])

                    total_weight = sum(dist_weights)
                    weighted_depth = sum(w * d for w, d in zip(dist_weights, depths)) / total_weight
                    depth_map[y, x] = weighted_depth
                else:
                    depth_map[y, x] = -1

    return depth_map


def normalize_depth_map(depth_map):
    """Normalize a depth map to 8-bit image range for visualization."""
    depth_map[depth_map == -1] = np.nan
    min_depth = np.nanmin(depth_map)
    max_depth = np.nanmax(depth_map)
    normalized_map = 255 * (depth_map - min_depth) / (max_depth - min_depth)
    normalized_map = np.nan_to_num(normalized_map, nan=0.0)
    return normalized_map.astype(np.uint8)


if __name__ == "__main__":
    for frame in range(7, 9931, 15):
        print(f"frame {frame}")
        radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(frame)
        if not (os.path.exists(radar_file) and os.path.exists(calib_file)):
            print("file not exist!")
            continue
        K = utils.get_intrinsic_matrix(calib_file)
        T_radar = utils.get_radar2cam(calib_file)
        T_lidar = utils.get_lidar2cam(lidar_calib_file)
        l_in_2d = np.load(vod_path("lidar_2d", f"{frame}.npy"))
        l_in = np.load(vod_path("lidar2d_corr_3dpoints", f"{frame}.npy"))
        lmap = -np.ones((1215 - 319, 1935))
        for i in range(len(l_in_2d)):
            u, v = l_in_2d[i]
            lmap[v - 319, u] = l_in[i, 0]
        non_zero_values_count = np.sum(lmap != -1)
        print("Number of values in lmap that are not -1:", non_zero_values_count)

        image_shape = (1215 - 319, 1935)
        coor = [0, 319, 1935, 1215]

        db = DBSCAN(eps=0.5, min_samples=10).fit(l_in)
        labels = db.labels_

        # Create a color map for cluster visualization/debugging.
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        image_shape1 = (1935, 896)

        global_depth_map = np.full(image_shape, -1, dtype=np.float32)
        depth_points_transformed = np.array([(y, x) for x, y in l_in_2d])
        for (y, x), depth in zip(depth_points_transformed, l_in[:, 0]):
            if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
                global_depth_map[y, x] = depth

        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Noise points are shown in black.
                col = [0, 0, 0, 1]

            class_member_mask = labels == k

            xy = l_in_2d[class_member_mask].astype(np.int32)

            for i in range(len(xy)):
                xy[i, 1] -= 319

            if len(xy) < 3:
                continue

            # Build a polygon mask from the cluster convex hull.
            hull = cv2.convexHull(xy.astype(np.float32))
            hull_int = hull.astype(np.int32)

            mask = np.zeros(image_shape, dtype=np.uint8)
            cv2.fillConvexPoly(mask, hull_int, [255, 255, 255])
            binary_mask = np.where(mask == 255, 1, 0)

            depth_points = xy
            depth_values = l_in[class_member_mask][:, 0]
            cluster_depth_map = interpolate_depth(
                image_shape, binary_mask, depth_points, depth_values, search_radius=10
            )

            mask_non_negative = binary_mask == 1
            global_depth_map[mask_non_negative] = np.where(
                global_depth_map[mask_non_negative] == -1,
                cluster_depth_map[mask_non_negative],
                global_depth_map[mask_non_negative],
            )
        print(global_depth_map.shape)
        np.save(vod_path("lidar_inter2d_matrix", f"{frame}.npy"), global_depth_map)
        non_zero_values_count = np.sum(global_depth_map != -1)
        print("Number of values in global_depth_map that are not -1:", non_zero_values_count)

        normalized_array = cv2.normalize(global_depth_map, None, 0, 255, cv2.NORM_MINMAX)
        gray_image = normalized_array.astype(np.uint8)
        cv2.imwrite(vod_path("lidar_inter2d_pic", f"{frame}.jpg"), gray_image)
