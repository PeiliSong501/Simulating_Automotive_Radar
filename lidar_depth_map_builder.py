
"""LiDAR depth map generation and validation utilities.
"""

import math
import os
import random
import re
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.environ.get("RADAR_SIMULATOR2_CODE_DIR", "/workspace/code/RadarSimulator2"))
import utils  # noqa: E402

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


def farthest_point_sampling(points, num_samples):
    """Farthest Point Sampling (FPS) for point clouds."""
    N, _ = points.shape
    if num_samples >= N:
        return points

    sampled_indices = [np.random.randint(N)]
    distances = np.full(N, np.inf)

    for _ in range(num_samples - 1):
        last_sampled_point = points[sampled_indices[-1]]
        dist_to_last = np.linalg.norm(points - last_sampled_point, axis=1)
        distances = np.minimum(distances, dist_to_last)

        next_index = np.argmax(distances)
        sampled_indices.append(next_index)

    return points[sampled_indices]


def generate_depth_map(l_in_2d, points_3d_l2c, image_width, image_height):
    """Generate a sparse depth map and a linearly interpolated depth map."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = l_in_2d[:, 0]
    y = l_in_2d[:, 1]
    z = points_3d_l2c[:, 2]

    u, v = x, y
    v = v - 571

    valid_mask = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height) & (z > 0)
    u = u[valid_mask]
    v = v[valid_mask]
    depth = z[valid_mask]

    u = np.round(u).astype(int)
    v = np.round(v).astype(int)

    depth_map = np.zeros((image_height, image_width))
    depth_map[v, u] = depth

    depth_map_before = depth_map.copy()

    grid_x, grid_y = np.meshgrid(np.arange(image_width), np.arange(image_height))

    valid_points = np.array([u, v]).T
    depth_interpolated = griddata(valid_points, depth, (grid_x, grid_y), method="linear", fill_value=0)

    return depth_map_before, depth_interpolated


def calculate_coverage_and_count(depth_map):
    """Count non-zero pixels in a depth map and compute their percentage."""
    total_pixels = depth_map.size
    non_zero_pixels = np.count_nonzero(depth_map)
    percentage = (non_zero_pixels / total_pixels) * 100
    return non_zero_pixels, percentage


def visualize_depth_map(depth_map, title):
    """Visualize a depth map after min-max normalization."""
    depth_normalized = (255 * (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))).astype(
        np.uint8
    )

    plt.imshow(depth_normalized, cmap="gray")
    plt.colorbar(label="Depth Value")
    plt.title(title)
    plt.axis("off")
    plt.show()


def save_depth_map_as_image(depth_map, file_path):
    """Save a normalized depth map as an image."""
    depth_normalized = (255 * (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))).astype(
        np.uint8
    )
    cv2.imwrite(file_path, depth_normalized)


def check_and_regenerate_depth_maps(start_frame, end_frame, depth_map_dir, regen_dir, image_width, image_height):
    """Validate saved depth maps and regenerate invalid ones when possible."""
    for frame in range(start_frame, end_frame + 1):
        depth_map_file = os.path.join(depth_map_dir, f"{frame}.npy")
        regen_file_path = os.path.join(regen_dir, f"{frame}.npy")

        try:
            if not os.path.exists(depth_map_file):
                print(f"File missing for frame {frame}, regenerating...")
                raise ValueError("Depth map file missing.")

            depth_map = np.load(depth_map_file)
            if depth_map.shape != (image_height, image_width) or np.all(depth_map == 0):
                raise ValueError("Invalid depth map dimensions or all zeros.")

            print(f"Depth map for frame {frame} is valid.")

        except (ValueError, FileNotFoundError) as e:
            print(f"Frame {frame} failed validation: {e}")
            try:
                radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(frame)
                if not (os.path.exists(radar_file) and os.path.exists(calib_file)):
                    print(f"Missing input files for frame {frame}, skipping regeneration.")
                    continue

                l_in_2d = np.load(vod_path("lidar_2d", f"{frame}.npy"))
                l_in = np.load(vod_path("lidar2d_corr_3dpoints", f"{frame}.npy"))
                K = utils.get_intrinsic_matrix(calib_file)
                T_lidar = utils.get_lidar2cam(lidar_calib_file)
                points_3d_l2c = utils.trans_point_coor(l_in, T_lidar)

                depth_map_before, depth_map_after = generate_depth_map(
                    l_in_2d, points_3d_l2c, image_width, image_height
                )

                np.save(regen_file_path, depth_map_after)
                print(f"Regenerated depth map for frame {frame}.")

            except Exception as regen_error:
                print(f"Failed to regenerate depth map for frame {frame}: {regen_error}")


if __name__ == "__main__":
    start_frame = 0
    end_frame = 9930
    depth_map_dir = vod_path("lidar_inter2d_matrix")
    regen_dir = vod_path("lidar_inter2d_matrix")
    image_width, image_height = 1935, 644

    check_and_regenerate_depth_maps(start_frame, end_frame, depth_map_dir, regen_dir, image_width, image_height)

    # frame = 0
    # for frame in range(0, 9931):
    #     radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(frame)
    #     if not (os.path.exists(radar_file) and os.path.exists(calib_file)):
    #         print(f"File not exist for frame {frame}!")
    #         continue
    #     else:
    #         print(f"dealing with frame {frame}")
    #     l_in_2d = np.load(vod_path("lidar_2d", f"{frame}.npy"))
    #     l_in = np.load(vod_path("lidar2d_corr_3dpoints", f"{frame}.npy"))
    #     K = utils.get_intrinsic_matrix(calib_file)
    #     T_lidar = utils.get_lidar2cam(lidar_calib_file)
    #     fx, cx = K[0, 0], K[0, 2]
    #     fy, cy = K[1, 1], K[1, 2]
    #     points_3d_l2c = utils.trans_point_coor(l_in, T_lidar)
    #     image_width, image_height = 1935, 644
    #
    #     depth_map_before, depth_map_after = generate_depth_map(l_in_2d, points_3d_l2c, image_width, image_height)
    #     np.save(vod_path("lidar_inter2d_matrix", f"{frame}.npy"), depth_map_after)
    #     save_path = vod_path("lidar_inter2d_pic", f"{frame}.jpg")
    #     save_depth_map_as_image(depth_map_after, save_path)
