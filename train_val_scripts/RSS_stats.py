"""
Utility and analysis script for radar cross section (RCS) dataset inspection.

Actually, all the 'RCS' within this code should be RSS(radar signal strength) to ensure rigor.
Not all the commercial radar outputs RCS(radar cross section) values.

Environment variables:
- VOD_BASE_DIR: base directory for the View-of-Delft dataset and derived files
- ASTYX_BASE_DIR: base directory for the Astyx dataset
- MSC_BASE_DIR: base directory for the MSC dataset
- OUTPUT_BASE_DIR: optional base directory for exported plots/results
"""

import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from scipy.spatial import KDTree
from scipy.stats import ks_2samp
from torch.utils.data import DataLoader

import utils
from dataset.dataset import RSS_Dataset_VoD, RSS_Dataset_astyx, RSS_Dataset_MSC
# import gt_generation_p as utils1


DEFAULT_VOD_BASE_DIR = os.environ.get(
    "VOD_BASE_DIR",
    r"D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset",
)
DEFAULT_ASTYX_BASE_DIR = os.environ.get(
    "ASTYX_BASE_DIR",
    r"D:/Astyx dataset/dataset_astyx_hires2019",
)
DEFAULT_MSC_BASE_DIR = os.environ.get(
    "MSC_BASE_DIR",
    r"D:/snail_radar/20231208/data4",
)
DEFAULT_OUTPUT_BASE_DIR = os.environ.get("OUTPUT_BASE_DIR", DEFAULT_VOD_BASE_DIR)
DEFAULT_VOD_RESULTS_BASE_DIR = os.environ.get("VOD_RESULTS_BASE_DIR", r"D:/VoD_dataset")


def vod_path(*parts: str) -> str:
    """Build a path under the configured View-of-Delft base directory."""
    return os.path.join(DEFAULT_VOD_BASE_DIR, *parts)


def astyx_path(*parts: str) -> str:
    """Build a path under the configured Astyx base directory."""
    return os.path.join(DEFAULT_ASTYX_BASE_DIR, *parts)


def msc_path(*parts: str) -> str:
    """Build a path under the configured MSC base directory."""
    return os.path.join(DEFAULT_MSC_BASE_DIR, *parts)


def output_path(*parts: str) -> str:
    """Build a path under the configured output directory."""
    return os.path.join(DEFAULT_OUTPUT_BASE_DIR, *parts)


def vod_results_path(*parts: str) -> str:
    """Build a path under the configured VoD results directory."""
    return os.path.join(DEFAULT_VOD_RESULTS_BASE_DIR, *parts)


def plot_class_distribution(class_count, title, file_name):
    """Plot and save the class-count distribution as a bar chart."""
    # Sort classes by count in descending order.
    sorted_class_count = sorted(class_count.items(), key=lambda item: item[1], reverse=True)

    labels, values = zip(*sorted_class_count)

    # Create the bar chart.
    plt.figure(figsize=(12, 6))  # Use a wider figure to improve readability.
    plt.bar(labels, values, color='skyblue')
    plt.title(title)
    plt.xlabel('Classes')
    plt.ylabel('Counts')

    # Rotate x-axis labels to avoid overlap.
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()  # Adjust layout automatically to reduce overlap.
    plt.savefig(file_name, format='svg')
    plt.show()


# def plot_class_distribution(class_count, title, save_path):
#     # Extract class labels and corresponding counts.
#     classes = list(class_count.keys())
#     counts = list(class_count.values())
#
#     # Create a bar chart.
#     plt.figure(figsize=(10, 6))
#     plt.bar(classes, counts, color='skyblue')
#
#     # Set title and axis labels.
#     plt.title(f'Point Class Distribution - {title}', fontsize=16)
#     plt.xlabel('Class', fontsize=12)
#     plt.ylabel('Number of Points', fontsize=12)
#
#     # Display the count label above each bar.
#     for i, count in enumerate(counts):
#         plt.text(i, count + 0.5, str(count), ha='center', fontsize=10)
#
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#
#     # Save as a vector graphic (SVG).
#     plt.savefig(save_path, format='svg')
#
#     # Show the figure.
#     plt.show()


if __name__ == '__main__':
    # dynamic_object = pd.read_csv(vod_path('dynamic_objects_total.csv')).values[:, 1:]
    # RCS_train = pd.read_csv(vod_path('RCS_dataset_train.csv')).values[:, 3:6]
    # RCS_test = pd.read_csv(vod_path('RCS_dataset_test.csv')).values[:, 3:6]
    # # Create Pandas DataFrames for visualization.
    # train_df = pd.DataFrame(RCS_train, columns=['x', 'y', 'z'])
    # test_df = pd.DataFrame(RCS_test, columns=['x', 'y', 'z'])
    #
    # # Set the visualization style.
    # sns.set(style="whitegrid")
    #
    # # Create multiple subplots.
    # fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    # fig.suptitle('X, Y, Z Distribution of RCS_Train, RCS_Test', fontsize=16)
    #
    # # Plot the distribution of RCS_train.
    # sns.histplot(train_df['x'], kde=True, ax=axs[0, 0], color='blue')
    # axs[0, 0].set_title('RCS_train: X axis')
    #
    # sns.histplot(train_df['y'], kde=True, ax=axs[1, 0], color='green')
    # axs[1, 0].set_title('RCS_train: Y axis')
    #
    # sns.histplot(train_df['z'], kde=True, ax=axs[2, 0], color='red')
    # axs[2, 0].set_title('RCS_train: Z axis')
    #
    # # Plot the distribution of RCS_test.
    # sns.histplot(test_df['x'], kde=True, ax=axs[0, 1], color='blue')
    # axs[0, 1].set_title('RCS_test: X axis')
    #
    # sns.histplot(test_df['y'], kde=True, ax=axs[1, 1], color='green')
    # axs[1, 1].set_title('RCS_test: Y axis')
    #
    # sns.histplot(test_df['z'], kde=True, ax=axs[2, 1], color='red')
    # axs[2, 1].set_title('RCS_test: Z axis')
    #
    # # Adjust subplot layout.
    # plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.savefig(output_path('RCS_train_vs_test_distribution.svg'), format='svg')
    #
    # plt.show()

    # RCS_train = pd.read_csv(vod_path('RCS_dataset_train.csv'))
    # RCS_test = pd.read_csv(vod_path('RCS_dataset_test.csv'))

    # RCS_train = pd.read_csv(astyx_path('RCS_dataset_train.csv'))
    # RCS_test = pd.read_csv(astyx_path('RCS_dataset_test.csv'))

    # RCS_train = pd.read_csv(msc_path('RCS_dataset_train.csv'))
    # RCS_test = pd.read_csv(msc_path('RCS_dataset_test.csv'))

    # from sklearn.metrics import mean_squared_error
    # min_value, max_value = -74.01181, 39.788345
    # # test1 = pd.read_csv(vod_results_path('RCS_validation_results_ablation_para_50_0.5.csv'))
    # test1 = pd.read_csv(vod_results_path('RCS_validation_results_ablation_para_100_2.csv'))
    # labels = test1.values[:, 2]
    # preds = test1.values[:, 3]
    # mse_direct = mean_squared_error(labels, preds)
    #
    # # Compute MSE after min-max normalization.
    # labels_normalized = (labels - min_value) / (max_value - min_value)
    # preds_normalized = (preds - min_value) / (max_value - min_value)
    # mse_normalized = mean_squared_error(labels_normalized, preds_normalized)
    #
    # print("Direct MSE:", mse_direct)
    # print("MSE after normalization:", mse_normalized)

    # -----------------------------------------------dataset augmentation-----------------------------------------------
    # RCS_set = pd.concat((RCS_train, RCS_test), axis=0)
    # rcs_values = RCS_set['rcs']
    # print(RCS_train.shape)
    # total_length = len(RCS_train)
    # min_length = total_length // 12
    #
    # # Define the RCS value bins.
    # min_rcs = rcs_values.min()
    # max_rcs = rcs_values.max()
    #
    # # Automatically generate bins with a width of 10.
    # bin_edges = np.arange(min_rcs, max_rcs + 10, 10)
    # rcs_bins = [(bin_edges[i], bin_edges[i + 1]) for i in range(len(bin_edges) - 1)]
    #
    # # Store the expanded data here.
    # expanded_data = []
    #
    # # Process each bin.
    # for lower_bound, upper_bound in rcs_bins:
    #     group_data = RCS_train[(RCS_train['rcs'] >= lower_bound) & (RCS_train['rcs'] < upper_bound)]
    #     group_length = len(group_data)
    #
    #     if group_length < min_length:
    #         # Expand the group by duplication.
    #         num_copies = min_length // group_length
    #         expanded_data.extend([group_data] * num_copies)
    #     else:
    #         # Randomly sample data from the group.
    #         sampled_data = group_data.sample(n=min_length, replace=False)
    #         expanded_data.append(sampled_data)
    #
    # # Merge expanded data into a single DataFrame.
    # expanded_RCS_train = pd.concat(expanded_data, ignore_index=True)
    #
    # # Shuffle the expanded dataset.
    # expanded_RCS_train = expanded_RCS_train.sample(frac=1).reset_index(drop=True)
    #
    # # Save to CSV.
    # expanded_RCS_train.to_csv(msc_path('expanded_RCS_dataset_train.csv'), index=False)
    #
    # # Print summary information for the expanded dataset.
    # print("Expanded dataset shape:", expanded_RCS_train.shape)
    #
    # expanded_rcs_values = expanded_RCS_train['rcs']
    #
    # # Count the number of samples in each previously defined bin.
    # expanded_group_counts = {}
    #
    # for lower_bound, upper_bound in rcs_bins:
    #     count = expanded_rcs_values[(expanded_rcs_values >= lower_bound) & (expanded_rcs_values < upper_bound)].count()
    #     expanded_group_counts[f"{lower_bound}--{upper_bound}"] = count
    #
    # # Convert to DataFrame for easier inspection.
    # expanded_group_counts_df = pd.DataFrame(list(expanded_group_counts.items()), columns=['RCS Group', 'Count'])
    #
    # # Print the count in each bin.
    # print("Expanded RCS value groups and counts:")
    # print(expanded_group_counts_df)

    # Read dynamic_objects_total.csv.
    # dynamic_object = pd.read_csv(vod_path('dynamic_objects_total.csv'))

    # RCS_set = pd.concat((RCS_train, RCS_test), axis=0)
    # print(np.min(RCS_train.values[:, -1]), np.max(RCS_train.values[:, -1]), np.mean(RCS_train.values[:, -1]), np.std(RCS_train.values[:, -1]))

    # ---------------------------save subset of objects to test RCS distribution---------------------------------------
    # matched_objects = []
    #
    # def find_matching_objects(RCS_data, dynamic_object):
    #     grouped_RCS_data = RCS_data.groupby('Frame')
    #     grouped_dynamic_object = dynamic_object.groupby('Frame')
    #
    #     total_frames = len(grouped_RCS_data)
    #     checkpoint = total_frames // 10  # Report progress every 10% of processed frames.
    #
    #     for idx, (frame, frame_RCS_points) in enumerate(grouped_RCS_data):
    #         if frame in grouped_dynamic_object.groups:  # Only process frames shared by both sources.
    #             frame_dynamic_objects = grouped_dynamic_object.get_group(frame)
    #
    #             for _, point in frame_RCS_points.iterrows():
    #                 point_cloud = np.array([[point['x'], point['y'], point['z']]])  # Point coordinates.
    #                 point_classified = False
    #
    #                 # Traverse the bounding boxes of dynamic objects in the same frame.
    #                 for _, obj in frame_dynamic_objects.iterrows():
    #                     bbox_location = [obj['Location_x'], obj['Location_y'], obj['Location_z']]
    #                     bbox_dimensions = [obj['Dimension_x'], obj['Dimension_y'], obj['Dimension_z']]
    #                     bbox_yaw = obj['Rotation']
    #
    #                     # Check whether the point falls inside the bounding box.
    #                     if len(utils.filter_points_in_bbox(point_cloud, bbox_location, bbox_dimensions, bbox_yaw)[0]) > 0:
    #                         # Save the object once a match is found.
    #                         matched_objects.append(obj)
    #                         point_classified = True
    #                         break  # Stop searching after the first match.
    #
    #         # Print progress every 10% of processed frames.
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
    # output_csv_path = vod_path('matched_dynamic_objects_subset.csv')
    # matched_objects_df.to_csv(output_csv_path, index=False)
    #
    # print(f"Subset of matched dynamic objects saved to {output_csv_path}")

    # ------------------------save classes of targets within RCS_train and RCS_test------------------------------------
    # # Dictionaries used to store class counts.
    # class_count_train = defaultdict(int)
    # class_count_test = defaultdict(int)
    #
    #
    # # Classify points according to frame-level matching.
    # def classify_points(RCS_data, dynamic_object, class_count):
    #     grouped_RCS_data = RCS_data.groupby('Frame')
    #     grouped_dynamic_object = dynamic_object.groupby('Frame')
    #
    #     total_frames = len(grouped_RCS_data)
    #     checkpoint = total_frames // 10  # Report progress every 10% of processed frames.
    #
    #     for idx, (frame, frame_RCS_points) in enumerate(grouped_RCS_data):
    #         if frame in grouped_dynamic_object.groups:  # Only process frames shared by both sources.
    #             frame_dynamic_objects = grouped_dynamic_object.get_group(frame)
    #             for _, point in frame_RCS_points.iterrows():
    #                 point_cloud = np.array([[point['x'], point['y'], point['z']]])  # Point coordinates.
    #                 point_classified = False
    #
    #                 # Traverse the bounding boxes of dynamic objects in the same frame.
    #                 for _, obj in frame_dynamic_objects.iterrows():
    #                     bbox_location = [obj['Location_x'], obj['Location_y'], obj['Location_z']]
    #                     bbox_dimensions = [obj['Dimension_x'], obj['Dimension_y'], obj['Dimension_z']]
    #                     bbox_yaw = obj['Rotation']
    #
    #                     # Check whether the point falls inside the bounding box.
    #                     if len(utils.filter_points_in_bbox(point_cloud, bbox_location, bbox_dimensions, bbox_yaw)[0]) > 0:
    #                         class_count[obj['Class']] += 1
    #                         point_classified = True
    #                         break  # Stop searching after the first match.
    #
    #                 # If the point is not assigned to any dynamic object, it can be counted as Background.
    #                 # if not point_classified:
    #                 #     class_count['Background'] += 1
    #
    #         # Print progress every 10% of processed frames.
    #         if (idx + 1) % checkpoint == 0:
    #             progress = (idx + 1) / total_frames * 100
    #             print(f"Progress: {progress:.1f}%")
    #
    #     print("Classification completed!")
    #
    #
    # # Compute class statistics for RCS_train and RCS_test.
    # print('Dealing with RCS_train')
    # classify_points(RCS_train, dynamic_object, class_count_train)
    #
    # print('Dealing with RCS_test')
    # classify_points(RCS_test, dynamic_object, class_count_test)
    #
    # # Print classification results.
    # print("Point classification statistics for RCS_train:")
    # for cls, count in class_count_train.items():
    #     print(f"{cls}: {count}")
    #
    # print("\nPoint classification statistics for RCS_test:")
    # for cls, count in class_count_test.items():
    #     print(f"{cls}: {count}")
    #
    # plot_class_distribution(class_count_train, "Target Distribution of RCS_train", output_path("RCS_train_distribution.svg"))
    # plot_class_distribution(class_count_test, "Target Distribution of RCS_test", output_path("RCS_test_distribution.svg"))

    # df = pd.read_csv(vod_path('matched_dynamic_objects_subset.csv')).iloc[:, 1:]
    # df['valid'] = 0  # Initialize the valid column.
    #
    # # Traverse the rows and compute the valid flag.
    # for i in range(len(df)):
    #     Frame, Track_ID, Class, Rotation, Location_x, Location_y, Location_z, PointNum, Dimension_x, Dimension_y, Dimension_z, v_r_real, _ = \
    #     df.iloc[i]
    #     l_in_radar_coor = np.load(vod_path('l_in_radar_coor', f'{Frame}.npy'))
    #     kdtree = KDTree(l_in_radar_coor)
    #     bbox_location = [Location_x, Location_y, Location_z]
    #     bbox_dimensions = [Dimension_x, Dimension_y, Dimension_z]
    #     bbox_yaw = Rotation
    #     r_in = np.load(vod_path('r_in', f'{Frame}.npy'))
    #     r_in_2d = np.load(vod_path('r_in_2d', f'{Frame}.npy'))
    #
    #     r_in_local, mask = utils.filter_points_in_bbox(r_in, bbox_location, bbox_dimensions, bbox_yaw)
    #
    #     # Compute the valid flag.
    #     if len(r_in_local) < 3:
    #         df.at[i, 'valid'] = 0
    #     else:
    #         df.at[i, 'valid'] = 1
    #         # Save local radar points.
    #         np.save(vod_path('rcs_dis_test', 'object_local_points', f'{Frame}_{Track_ID}.npy'), r_in_local)
    #         r_in_2d_local = r_in_2d[mask]
    #
    # # Save the updated DataFrame.
    # df.to_csv(vod_path('expanded_new_df_test.csv'), index=False)

    #     for j in range(len(r_in_local)):
    #         x, y, z, v_r, rcs = r_in_local[j, :5]
    #         u, v = r_in_2d_local[j, :]
    #         index = i * 1000 + j
    #         new_data = pd.DataFrame(
    #                         {'Frame': [Frame], 'Index': [index], 'x': [x], 'y': [y], 'z': [z], 'u': [u], 'v': [v], 'v_r': [v_r], 'rcs': [rcs]})
    #         new_df_test = pd.concat([new_df_test, new_data], ignore_index=True)
    #         # Set index based on i so that local points for a specific object are easier to retrieve.
    #         local_lidar_index = kdtree.query_ball_point([x, y, z], r=1)
    #         local_lidar_radarcoor = l_in_radar_coor[local_lidar_index]
    #         dst = vod_path('rcs_dis_test', 'range_image', f'{Frame}_{index}.jpg')
    #         virtual_point = utils1.get_new_point(x, y, z)
    #         fov_down = 7
    #         fov_up = -23
    #         for k in range(len(local_lidar_radarcoor)):
    #             local_lidar_radarcoor[k, 0] -= virtual_point[0]
    #             local_lidar_radarcoor[k, 1] -= virtual_point[1]
    #             local_lidar_radarcoor[k, 2] -= virtual_point[2]
    #             x1, y1, z1 = local_lidar_radarcoor[k]
    #             pitch = np.degrees(np.arctan(z1 / np.linalg.norm([x1, y1], 2)))
    #             if pitch > fov_up:
    #                 fov_up = pitch
    #             if pitch < fov_down:
    #                 fov_down = pitch
    #
    #         proj_H, proj_W = 32, 128
    #         utils1.gen_range_image_rcs(local_lidar_radarcoor, fov_up, fov_down, proj_H, proj_W, dst)

    # ----------------------------test whether generated RCS follows the target distribution-----------------------------
    df = pd.read_csv(vod_path('matched_dynamic_objects_subset_new.csv'))
    arr = df[df['valid'] == 1].values

    # df_rcs_valid = pd.read_csv(vod_path('ablation', 'RCS_validation_results_ablation_para_150_2.csv'))
    df_rcs_valid = pd.read_csv(vod_results_path('RCS_validation_results_ablation_rgb_li_fe.csv'))
    df_rcs_points = pd.read_csv(vod_path('RCS_dataset_test.csv'))

    success_cnt = 0
    fail_cnt = 0
    total_cnt = 0

    for i in range(len(arr)):
        Frame, Track_ID, Class, Rotation, Location_x, Location_y, Location_z, PointNum, Dimension_x, Dimension_y, Dimension_z, v_r_real, _ = arr[i]

        # Get the indices for the current frame.
        indices = df_rcs_valid[df_rcs_valid['Frame'] == Frame]['Index'].values

        # Load the point cloud for the current frame.
        point_cloud = df_rcs_points[df_rcs_points['Frame'] == Frame][['x', 'y', 'z', 'rcs', 'v_r']].values

        # Define the bounding box.
        bbox_location = [Location_x, Location_y, Location_z]
        bbox_dimensions = [Dimension_x, Dimension_y, Dimension_z]
        bbox_yaw = Rotation

        # Filter points inside the bounding box.
        test_p_in, mask = utils.filter_points_in_bbox(point_cloud, bbox_location, bbox_dimensions, bbox_yaw)

        # Load the local radar points for the object.
        local_points = np.load(vod_path('rcs_dis_test', 'object_local_points', f'{Frame}_{Track_ID}.npy'))

        # Retrieve the true RCS values.
        rcs_true = local_points[:, 3]

        # Retrieve the predicted RCS values for test points.
        rcs_pred_full = df_rcs_valid[
            (df_rcs_valid['Frame'] == Frame) & (df_rcs_valid['Index'].isin(indices))
        ]['Predictions'].values

        # Keep only predictions corresponding to points inside the bounding box.
        rcs_pred = rcs_pred_full[mask] if len(mask) == len(rcs_pred_full) else []

        # Test whether predicted values follow the same distribution as the true values.
        if len(rcs_true) > 0 and len(rcs_pred) > 0:
            statistic, p_value = ks_2samp(rcs_true, rcs_pred)

            # Use the p-value threshold to decide whether the distributions match.
            if p_value > 0.05:
                success_cnt += 1
                # print(
                #     f'Frame {Frame}, Track ID {Track_ID}: predicted and true RCS values follow the same distribution (p={p_value:.4f})')
                # print(len(rcs_true), len(rcs_pred))
                # print('True RCS:', rcs_true)
                # print('Predicted RCS:', rcs_pred)
            else:
                fail_cnt += 1
                # print(
                #     f'Frame {Frame}, Track ID {Track_ID}: predicted and true RCS values do not follow the same distribution (p={p_value:.4f})')

            total_cnt += 1

    print('Success rate:', success_cnt / total_cnt)
    print('Success, Fail, Total:', success_cnt, fail_cnt, total_cnt)
