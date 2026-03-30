import os
import csv
import math
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import open3d as o3d
from scipy.spatial import KDTree

import utils


# ============================================
# Path configuration
# Edit only this section when reproducing.
# ============================================
DATASET_ROOT = "D:/VoD_dataset"
LABEL_ROOT = os.path.join(DATASET_ROOT, "label_2_with_track_ids", "label_2")

TRACK_INFO_CSV = os.path.join(DATASET_ROOT, "track_info.csv")
TRACK_STATISTICS_CSV = os.path.join(DATASET_ROOT, "track_statistics.csv")
TRACK_INFO_VALID_CSV = os.path.join(DATASET_ROOT, "track_info_valid.csv")
TRACK_INFO_VALID_PCD_CSV = os.path.join(DATASET_ROOT, "track_info_valid_pcd.csv")
TRACK_ID_FRAME_PAIRS_CSV = os.path.join(DATASET_ROOT, "track_id_frame_pairs.csv")
TRACK_ID_FRAME_PAIRS_CLASS_CSV = os.path.join(DATASET_ROOT, "track_id_frame_pairs_class.csv")
CLASS_COUNTS_CSV = os.path.join(DATASET_ROOT, "class_counts.csv")


# ============================================
# Dataset configuration
# ============================================
NUM_TOTAL_FRAMES = 9931

CLASS_LIST = [
    "ride_uncertain",
    "rider",
    "moped_scooter",
    "bicycle",
    "Cyclist",
    "vehicle_other",
    "Pedestrian",
    "truck",
    "DontCare",
    "motor",
    "bicycle_rack",
    "Car",
    "human_depiction",
    "ride_other",
]
# Note: "DontCare" is listed here for completeness, but is usually excluded
# from meaningful statistics or training-related analysis.


def create_point_cloud(points):
    """
    Convert a numpy array of points into an Open3D point cloud.
    """
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    return pc


def compute_class_statistics(data):
    """
    Compute min / max / average consecutive frame statistics for each class.

    Args:
        data (pd.DataFrame): Must contain columns:
            - Class
            - Consecutive_Frames

    Returns:
        pd.DataFrame: Per-class statistics.
    """
    stats = []
    classes = data["Class"].unique()

    for cls in classes:
        class_data = data[data["Class"] == cls]

        min_frames = class_data["Consecutive_Frames"].min()
        max_frames = class_data["Consecutive_Frames"].max()
        avg_frames = class_data["Consecutive_Frames"].mean()

        stats.append({
            "Class": cls,
            "Min_Consecutive_Frames": min_frames,
            "Max_Consecutive_Frames": max_frames,
            "Avg_Consecutive_Frames": avg_frames,
        })

    return pd.DataFrame(stats)


def trackid2pcd(track_id, frame):
    """
    Extract the radar point cloud belonging to a specific tracked object
    at a given frame.

    Args:
        track_id (int): Target track ID.
        frame (int): Frame index.

    Returns:
        np.ndarray or None:
            Radar points inside the target bounding box.
            Returns None if the track is not found.
    """
    file_path = os.path.join(LABEL_ROOT, f"{frame:05d}.txt")
    radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(frame)
    T_radar = utils.get_radar2cam(calib_file)
    annotations = utils.read_annotation(file_path)

    pcd = None
    for j in range(len(annotations)):
        track_tmp = annotations[j]["Track_ID"]
        if track_tmp == track_id:
            bbox, loc = utils.read_3dbbox(annotations[j])
            bbox = utils.transform_bbox_to_radar(bbox, T_radar)
            loc = utils.transform_coor_to_radar(loc, T_radar)
            dimension = annotations[j]["Dimensions"]
            yaw = annotations[j]["Rotation"]

            pcd = np.fromfile(radar_file, dtype=np.float32).reshape(-1, 7)
            pcd = utils.filter_points_in_bbox(pcd, loc, dimension, yaw)
            pcd = pcd[pcd[:, 0] > 0]
            break

    if pcd is None:
        print(f"Track ID {track_id} not found in frame {frame}")

    return pcd


def track_id_f2bbox(track_id, frame):
    """
    Get the 3D bounding box, object center, and yaw angle
    of a target track in a given frame.

    Args:
        track_id (int): Target track ID.
        frame (int): Frame index.

    Returns:
        tuple:
            bbox, loc, yaw
    """
    file_path = os.path.join(LABEL_ROOT, f"{frame:05d}.txt")
    radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(frame)
    T_radar = utils.get_radar2cam(calib_file)
    annotations = utils.read_annotation(file_path)

    bbox, loc, yaw = None, None, None
    for j in range(len(annotations)):
        track_tmp = annotations[j]["Track_ID"]
        if track_tmp == track_id:
            bbox, loc = utils.read_3dbbox(annotations[j])
            bbox = utils.transform_bbox_to_radar(bbox, T_radar)
            loc = utils.transform_coor_to_radar(loc, T_radar)
            yaw = annotations[j]["Rotation"]
            break

    return bbox, loc, yaw


def compute_point_cloud_statistics(point_cloud, bbox, loc):
    """
    Compute point cloud density, centroid, centroid offset,
    covariance matrix, and minimum spacing along x/y/z.

    Args:
        point_cloud (np.ndarray): Point cloud of shape (N, 3) or (N, >=4).
            Only the first three columns are used.
        bbox (array-like): 3D bounding box range:
            [xmin, xmax, ymin, ymax, zmin, zmax]
        loc (array-like): Bounding box center [x, y, z]

    Returns:
        tuple:
            density (float),
            centroid (np.ndarray),
            centroid_offset (np.ndarray),
            covariance_matrix (np.ndarray),
            min_distances (np.ndarray)
    """
    if point_cloud.shape[1] > 3:
        points = point_cloud[:, :3]
    else:
        points = point_cloud

    bbox_volume = (bbox[1] - bbox[0]) * (bbox[3] - bbox[2]) * (bbox[5] - bbox[4])
    density = len(points) / bbox_volume

    centroid = np.mean(points, axis=0)
    centroid_offset = centroid - np.array(loc)

    covariance_matrix = np.cov(points, rowvar=False)

    min_distances = np.zeros(3)
    for i in range(3):
        diffs = np.abs(np.diff(np.sort(points[:, i])))
        min_distances[i] = np.min(diffs) if len(diffs) > 0 else 0.0

    return density, centroid, centroid_offset, covariance_matrix, min_distances


def cal_1pair_rcs(track_id, src_frame, tar_frame, thres):
    """
    Calculate RCS differences for matched radar points between two frames
    belonging to the same tracked object.

    Matching is performed after transforming the source radar points to the
    target frame using odometry, followed by nearest-neighbor search.

    Args:
        track_id (int): Target track ID.
        src_frame (int): Source frame index.
        tar_frame (int): Target frame index.
        thres (float): Distance threshold for accepting a matched point pair.

    Returns:
        tuple:
            rcs_differences (np.ndarray),
            num_matches (int)
    """
    frame0, frame1 = src_frame, tar_frame

    radar_file1, lidar_file1, cam_file1, calib_file1, lidar_calib_file1, txt_base_dir1 = utils.get_vod_dir(frame0)
    radar_file2, lidar_file2, cam_file2, calib_file2, lidar_calib_file2, txt_base_dir2 = utils.get_vod_dir(frame1)

    annotation_file1 = txt_base_dir1 + str(src_frame).zfill(5) + ".txt"
    annotation_file2 = txt_base_dir2 + str(tar_frame).zfill(5) + ".txt"

    annotations1 = utils.read_annotation(annotation_file1)
    annotations2 = utils.read_annotation(annotation_file2)

    index1, index2 = -1, -1
    for j in range(len(annotations1)):
        if track_id == annotations1[j]["Track_ID"]:
            index1 = j
            break

    for j in range(len(annotations2)):
        if track_id == annotations2[j]["Track_ID"]:
            index2 = j
            break

    if index1 == -1 or index2 == -1:
        print("Target track not found in one of the frames.")
        return np.array([]), 0

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

    dimension1 = annotations1[index1]["Dimensions"]
    dimension2 = annotations2[index2]["Dimensions"]
    yaw1 = annotations1[index1]["Rotation"]
    yaw2 = annotations2[index2]["Rotation"]

    pcd1 = utils.filter_points_in_bbox(pcl1, loc1, dimension1, yaw1)
    pcd2 = utils.filter_points_in_bbox(pcl2, loc2, dimension2, yaw2)

    odom_transform, _, _ = utils.compute_transform(frame0, frame1, T_radar1, T_radar2)
    if odom_transform is None:
        return np.array([]), 0

    xyz1 = pcd1[:, :3]
    xyz2 = pcd2[:, :3]
    rcs1 = pcd1[:, 3]
    rcs2 = pcd2[:, 3]

    transformed_xyz1 = (
        odom_transform @ np.hstack((xyz1, np.ones((xyz1.shape[0], 1)))).T
    ).T[:, :3]

    kdtree = KDTree(xyz2)
    distances, indices = kdtree.query(transformed_xyz1)

    rcs_differences = []
    for i, distance in enumerate(distances):
        if distance < thres:
            nearest_rcs = rcs2[indices[i]]
            rcs_difference = rcs1[i] - nearest_rcs
            rcs_differences.append(rcs_difference)

    rcs_differences = np.array(rcs_differences)
    return rcs_differences, len(rcs_differences)


def build_track_info_csv(output_csv=TRACK_INFO_CSV):
    """
    Build a CSV file that records track occurrence information for each class.

    Output columns:
        - Class
        - Track_ID
        - First_Appearance_Frame
        - Consecutive_Frames
    """
    class_track_info = {cls: defaultdict(list) for cls in CLASS_LIST}

    for frame in range(NUM_TOTAL_FRAMES):
        file_path = os.path.join(LABEL_ROOT, f"{frame:05d}.txt")
        if os.path.exists(file_path):
            annotations = utils.read_annotation(file_path)
            for annotation in annotations:
                cls = annotation["Class"]
                track_id = annotation["Track_ID"]
                class_track_info[cls][track_id].append(frame)

    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = ["Class", "Track_ID", "First_Appearance_Frame", "Consecutive_Frames"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for cls, tracks in class_track_info.items():
            for track_id, frames in tracks.items():
                first_appearance = frames[0]
                consecutive_frames = 1

                for i in range(1, len(frames)):
                    if frames[i] == frames[i - 1] + 1:
                        consecutive_frames += 1
                    else:
                        writer.writerow({
                            "Class": cls,
                            "Track_ID": track_id,
                            "First_Appearance_Frame": first_appearance,
                            "Consecutive_Frames": consecutive_frames,
                        })
                        first_appearance = frames[i]
                        consecutive_frames = 1

                writer.writerow({
                    "Class": cls,
                    "Track_ID": track_id,
                    "First_Appearance_Frame": first_appearance,
                    "Consecutive_Frames": consecutive_frames,
                })

    print(f"Saved track info to: {output_csv}")


def build_track_statistics_csv(input_csv=TRACK_INFO_CSV, output_csv=TRACK_STATISTICS_CSV):
    """
    Compute per-class consecutive frame statistics from track_info.csv.
    """
    data = pd.read_csv(input_csv)
    statistics = compute_class_statistics(data)
    statistics.to_csv(output_csv, index=False)
    print(f"Saved class statistics to: {output_csv}")


def build_valid_track_info_csv(input_csv=TRACK_INFO_CSV, output_csv=TRACK_INFO_VALID_CSV):
    """
    Add a simple validity flag based on consecutive frame length.

    Current rule:
        valid = 1 if Consecutive_Frames > 5 else 0
    """
    data = pd.read_csv(input_csv)
    data["valid"] = data["Consecutive_Frames"].apply(lambda x: 1 if x > 5 else 0)
    data.to_csv(output_csv, index=False)
    print(f"Saved valid track info to: {output_csv}")


def add_class_to_track_pairs(
    track_info_valid_pcd_csv=TRACK_INFO_VALID_PCD_CSV,
    track_id_frame_pairs_csv=TRACK_ID_FRAME_PAIRS_CSV,
    output_csv=TRACK_ID_FRAME_PAIRS_CLASS_CSV,
):
    """
    Add class labels to the track_id_frame_pairs CSV using Track_ID lookup.
    """
    track_info_valid_pcd = pd.read_csv(track_info_valid_pcd_csv)
    track_id_frame_pairs = pd.read_csv(track_id_frame_pairs_csv)

    track_id_to_class = dict(zip(track_info_valid_pcd["Track_ID"], track_info_valid_pcd["Class"]))
    track_id_frame_pairs["Class"] = track_id_frame_pairs["track_id"].map(track_id_to_class)
    track_id_frame_pairs.to_csv(output_csv, index=False)

    print(f"Saved track pairs with class labels to: {output_csv}")


def count_class_occurrences_in_track_pairs(
    track_info_valid_pcd_csv=TRACK_INFO_VALID_PCD_CSV,
    track_id_frame_pairs_csv=TRACK_ID_FRAME_PAIRS_CSV,
    output_csv=CLASS_COUNTS_CSV,
):
    """
    Count how many track pairs belong to each class.
    """
    track_info_valid_pcd = pd.read_csv(track_info_valid_pcd_csv)
    track_id_frame_pairs = pd.read_csv(track_id_frame_pairs_csv)

    valid_track_info = track_info_valid_pcd[track_info_valid_pcd["valid"] == 1]
    class_counts = {}

    for _, row in track_id_frame_pairs.iterrows():
        track_id = row["track_id"]
        matched_rows = valid_track_info[valid_track_info["Track_ID"] == track_id]
        if len(matched_rows) == 0:
            continue

        class_name = matched_rows["Class"].values[0]

        if class_name in class_counts:
            class_counts[class_name] += 1
        else:
            class_counts[class_name] = 1

    class_counts_df = pd.DataFrame(list(class_counts.items()), columns=["Class", "Count"])
    class_counts_df.to_csv(output_csv, index=False)

    print(class_counts_df)
    print(f"Saved class counts to: {output_csv}")


def calculate_rcs_baseline(track_pairs_csv=TRACK_ID_FRAME_PAIRS_CLASS_CSV, distance_threshold=0.2):
    """
    Calculate per-class and overall RCS MSE baseline using matched radar points
    between paired frames.

    Args:
        track_pairs_csv (str): CSV file containing columns:
            - track_id
            - src_frame
            - tar_frame
            - Class
        distance_threshold (float): Nearest-neighbor matching threshold.

    Returns:
        tuple:
            class_mse_results (dict),
            overall_mse (float or None)
    """
    track_id_frame_pairs = pd.read_csv(track_pairs_csv)

    total_rcs = None
    total_len_rcs = 0
    class_mse_results = {}

    for c in CLASS_LIST:
        print(f"Processing class: {c}")

        len_rcs = 0
        rcs_array = None
        data = track_id_frame_pairs[track_id_frame_pairs["Class"] == c].values

        for i in range(len(data)):
            track_id, src_frame, tar_frame = data[i][0:3]
            rcs_array_tmp, len_rcs_tmp = cal_1pair_rcs(track_id, src_frame, tar_frame, distance_threshold)

            if len_rcs_tmp > 0:
                if rcs_array is None:
                    rcs_array = rcs_array_tmp
                else:
                    rcs_array = np.concatenate((rcs_array, rcs_array_tmp))

            len_rcs += len_rcs_tmp

        print(f"Matched points found for class {c}: {len_rcs}")

        if rcs_array is not None and len(rcs_array) > 0:
            mse = np.sum(rcs_array ** 2) / len(rcs_array)
            class_mse_results[c] = mse
            print(f"RCS MSE for class {c}: {mse}")

            if total_rcs is None:
                total_rcs = rcs_array
            else:
                total_rcs = np.concatenate((total_rcs, rcs_array))

        total_len_rcs += len_rcs

    overall_mse = None
    if total_rcs is not None and len(total_rcs) > 0:
        overall_mse = np.sum(total_rcs ** 2) / len(total_rcs)
        print(f"Overall RCS MSE: {overall_mse}")

    return class_mse_results, overall_mse


if __name__ == "__main__":
    # Example: load existing CSV files for quick inspection.
    # Uncomment the parts you want to run.

    track_info_valid_pcd_path = TRACK_INFO_VALID_PCD_CSV
    track_id_frame_pairs_path = TRACK_ID_FRAME_PAIRS_CSV

    if os.path.exists(track_info_valid_pcd_path) and os.path.exists(track_id_frame_pairs_path):
        track_info_valid_pcd = pd.read_csv(track_info_valid_pcd_path)
        track_id_frame_pairs = pd.read_csv(track_id_frame_pairs_path)

        valid_track_info = track_info_valid_pcd[track_info_valid_pcd["valid"] == 1]
        print("Loaded valid track info and track-pair CSV files.")
        print(f"Number of valid tracks: {len(valid_track_info)}")
        print(f"Number of track pairs: {len(track_id_frame_pairs)}")

    # Example pipeline:
    # build_track_info_csv()
    # build_track_statistics_csv()
    # build_valid_track_info_csv()
    # add_class_to_track_pairs()
    # count_class_occurrences_in_track_pairs()
    # calculate_rcs_baseline()