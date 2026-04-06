"""
velocity_estimation.py

This file contains utilities for estimating radar ego-motion and target radial
velocity, and for generating Doppler velocity values for synthesized 3D radar
points. For open-source reproducibility, all editable dataset paths are grouped
at the top of the file so users can adapt the script by changing only a few
configuration variables.
"""

import csv
import gc
import math
import os
import pickle
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
import utils
# from sample_vod_stats import trackid2pcd, track_id_f2bbox
from scipy.spatial import KDTree
from scipy.stats import multivariate_normal
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LinearRegression, RANSACRegressor, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# =============================================================================
# Reproducibility configuration
# Update these paths to match your local environment before running the script.
# =============================================================================
METHOD_NAME = ""    #PLEASE NAME ONE

DATA_ROOT = "/workspace/data/VoD_dataset"
VOD_PUBLIC_ROOT = os.path.join(DATA_ROOT, "view_of_delft_PUBLIC")
RADAR_TRAIN_DIR = os.path.join(VOD_PUBLIC_ROOT, "radar", "training")
PMAP_DIR = os.path.join(RADAR_TRAIN_DIR, "PMapDataset")

GENERATED_3D_DIR = os.path.join(PMAP_DIR,f"generated_3d_{METHOD_NAME}_lidardepth")
GENERATED_4D_DIR = os.path.join(PMAP_DIR,f"generated_4d_{METHOD_NAME}_lidardepth")

LABEL_TRACK_DIR = os.path.join(DATA_ROOT, "label_2_with_track_ids", "label_2")

RADAR_VELO_PATH = os.path.join(RADAR_TRAIN_DIR, "radar_velo_lidar_icp.npy")
DYNAMIC_OBJECTS_CSV = os.path.join(PMAP_DIR, "dynamic_objects_total.csv")
PROB_DATASET_CSV = os.path.join(PMAP_DIR, "prob_dataset.csv")
TRAIN_DATASET_CSV = os.path.join(PMAP_DIR, "train_dataset.csv")
TEST_DATASET_CSV = os.path.join(PMAP_DIR, "test_dataset.csv")
ESTIMATED_POINTNUM_CSV = os.path.join(
    PMAP_DIR,
    "exp_2",
    "estimated_pointnum_vanila_no_freeze_softmax.csv",
)



ego_velocity = np.load(RADAR_VELO_PATH)
class_list = [
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
column_names = [
    "Frame",
    "Track_ID",
    "Class",
    "Rotation",
    "Location_x",
    "Location_y",
    "Location_z",
    "PointNum",
    "Dimension_x",
    "Dimension_y",
    "Dimension_z",
    "v_r_real",
]
df = pd.DataFrame(columns=column_names)

# -----------------------------------------------------------------------------
# Velocity estimation for generated radar point clouds.
# This block assigns Doppler velocity to synthesized 3D radar points.
# -----------------------------------------------------------------------------
df2 = pd.read_csv(TEST_DATASET_CSV)
df_dynamic_obj = pd.read_csv(DYNAMIC_OBJECTS_CSV)
df = pd.read_csv(ESTIMATED_POINTNUM_CSV)
df1 = pd.read_csv(PROB_DATASET_CSV)
df2 = pd.read_csv(TEST_DATASET_CSV)
df3 = pd.read_csv(TRAIN_DATASET_CSV)

sequences_to_filter = [
    (4049, 4386),
    (8481, 8748),
    (8749, 9095),
    (9518, 9775),
    (9776, 9930),
]
filtered_df = df3[
    df3["Frame"].apply(lambda x: any(start <= x <= end for start, end in sequences_to_filter))
]

# Display the filtered results if needed.
# print(len(filtered_df))

save_base = GENERATED_4D_DIR
os.makedirs(save_base, exist_ok=True)

ego_velo_all = np.load(RADAR_VELO_PATH)

for i in range(len(df2)):
    frame = int(df2.iloc[i]["Frame"])
    ego_velo = ego_velo_all[frame, :]
    print(f"dealing with frame {frame} with ego velo {ego_velo}")

    generated_file_path = os.path.join(GENERATED_3D_DIR, f"{frame}.npy")
    if not os.path.exists(generated_file_path):
        continue
    generated_3d = np.load(generated_file_path)

    radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(frame)
    K = utils.get_intrinsic_matrix(calib_file)
    T_radar = utils.get_radar2cam(calib_file)
    annotations = utils.read_annotation(txt_base_dir + str(frame).zfill(5) + ".txt")
    if annotations is None:
        continue

    # Collect the expanded 3D bounding box and reference radial velocity for
    # each annotated object in radar coordinates.
    obj_info = {}
    for anno in annotations:
        bbox_camera, loc = utils.read_3dbbox(anno)
        bbox_radar = utils.transform_bbox_to_radar(bbox_camera, T_radar)
        track_id = anno["Track_ID"]
        v_r_real = df_dynamic_obj[df_dynamic_obj["Track_ID"] == track_id]["v_r_real"].values[0]

        # Expand the bounding box by 0.5 m in each direction.
        bbox_radar_expanded = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        obj_info[track_id] = {
            "bbox_radar": bbox_radar,
            "bbox_radar_expanded": bbox_radar_expanded,
            "v_r_real": v_r_real,
        }

    doppler_velocities = []
    for point in generated_3d:
        x, y, z = point

        radar_position = np.array([0, 0, 0])
        point_vector = np.array([x, y, z]) - radar_position
        point_vector_norm = np.linalg.norm(point_vector)
        point_vector_normalized = point_vector / point_vector_norm

        closest_distance = float("inf")
        closest_bbox = None
        closest_v_r_real = None
        closest_track_id = None

        for track_id, info in obj_info.items():
            bbox_radar = info["bbox_radar"]
            bbox_radar_expanded = info["bbox_radar_expanded"]
            v_r_real = info["v_r_real"]

            if (
                bbox_radar[0] - bbox_radar_expanded[0] <= x <= bbox_radar_expanded[1] + bbox_radar[1]
                and bbox_radar[2] - bbox_radar_expanded[2] <= y <= bbox_radar_expanded[3] + bbox_radar[3]
                and bbox_radar[4] - bbox_radar_expanded[4] <= z <= bbox_radar_expanded[5] + bbox_radar[5]
            ):
                bbox_center = np.array(
                    [
                        (bbox_radar[0] + bbox_radar[1]) / 2,
                        (bbox_radar[2] + bbox_radar[3]) / 2,
                        (bbox_radar[4] + bbox_radar[5]) / 2,
                    ]
                )
                distance = np.linalg.norm(np.array([x, y, z]) - bbox_center)

                if distance < closest_distance:
                    closest_distance = distance
                    closest_bbox = bbox_radar
                    closest_v_r_real = v_r_real
                    closest_track_id = track_id

        # If the point belongs to an object box, use the object's reference
        # radial velocity and remove the radar ego-motion component. Otherwise,
        # use only the ego-motion projection as the Doppler velocity estimate.
        if closest_bbox is not None:
            ego_velocity_component = np.dot(np.array(ego_velo), point_vector_normalized)
            doppler_velocity = closest_v_r_real - ego_velocity_component
        else:
            radar_velocity_vector = np.array(ego_velo)
            doppler_velocity = -np.dot(radar_velocity_vector, point_vector_normalized)

        doppler_velocities.append(doppler_velocity)

    generated_3d = np.column_stack((generated_3d, doppler_velocities))

    save_path = os.path.join(save_base, f"{frame}.npy")
    np.save(save_path, generated_3d)
    print(f"Saved generated 4D data for frame {frame} to {save_path}")
