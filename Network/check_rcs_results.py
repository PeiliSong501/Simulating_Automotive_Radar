import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import os
from torch.utils.data import DataLoader
from RCSNet import RCSNet
from dataset import RCS_Dataset_VoD, RCS_Dataset_astyx, RCS_Dataset_Snail
import random
import matplotlib.pyplot as plt
import seaborn as sns
import utils
from collections import defaultdict
import gt_generation_p as utils1
from scipy.spatial import KDTree
from scipy.stats import ks_2samp


def calculate_mse_in_range(df, range_min, range_max):
    filtered_df = df[(df['Range'] >= range_min) & (df['Range'] < range_max)]
    mse = np.mean((filtered_df['Predictions'] - filtered_df['Real RCS']) ** 2)
    return mse


def stats_in_range(df, range_min, range_max):
    filtered_df = df[(df['Range'] >= range_min) & (df['Range'] < range_max)]
    min_rcs = filtered_df['Real RCS'].min()
    max_rcs = filtered_df['Real RCS'].max()
    std_rcs = filtered_df['Real RCS'].std()
    count_in_range = len(filtered_df)

    return min_rcs, max_rcs, std_rcs, count_in_range

pmap_dataset_vod = pd.read_csv('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/prob_dataset.csv').values
pmap_dataset_astyx = pd.read_csv("D:Astyx dataset/dataset_astyx_hires2019/prob_dataset.csv").values
pmap_dataset_sr = pd.read_csv("D:/snail_radar/20231208/data4/prob_dataset.csv").values
# print(pmap_dataset_vod[:,1].mean(),pmap_dataset_astyx[:,1].mean(),pmap_dataset_sr[:,1].mean())

range0, range1, range2, range3 = 0,30,50,100

rcs_vod_estimated = pd.read_csv("D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/ablation/RCS_validation_results_ablation_para_150_2.csv")
rcs_vod_real = pd.read_csv("D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/RCS_dataset_test.csv")
rcs_vod_real['range'] = np.sqrt(rcs_vod_real['x']**2 + rcs_vod_real['y']**2 + rcs_vod_real['z']**2)
merged_df = pd.merge(rcs_vod_estimated, rcs_vod_real[['Frame', 'Index', 'rcs', 'range']], on=['Frame', 'Index'], how='left')
merged_df = merged_df.rename(columns={'rcs': 'Real RCS', 'range': 'Range'})
#print(merged_df.head())
print('VoD dataset:')
print(f'MSE of RCS in range {range0}~{range1}')
print(calculate_mse_in_range(merged_df,range0,range1))
print(f"RCS Stats for Range between {range0} and {range1}:")
min_rcs, max_rcs, std_rcs, count_in_range = stats_in_range(merged_df, range0, range1)
print(f"Count of entries in range: {count_in_range}")
print(f"Min RCS: {min_rcs}")
print(f"Max RCS: {max_rcs}")
print(f"Std RCS: {std_rcs}")

print(f'MSE of RCS in range {range1}~{range2}')
print(calculate_mse_in_range(merged_df,range1,range2))
print(f"RCS Stats for Range between {range1} and {range2}:")
min_rcs, max_rcs, std_rcs, count_in_range = stats_in_range(merged_df, range1, range2)
print(f"Count of entries in range: {count_in_range}")
print(f"Min RCS: {min_rcs}")
print(f"Max RCS: {max_rcs}")
print(f"Std RCS: {std_rcs}")

print(f'MSE of RCS in range {range2}~{range3}')
print(calculate_mse_in_range(merged_df,range2,range3))
print(f"RCS Stats for Range between {range2} and {range3}:")
min_rcs, max_rcs, std_rcs, count_in_range = stats_in_range(merged_df, range2, range3)
print(f"Count of entries in range: {count_in_range}")
print(f"Min RCS: {min_rcs}")
print(f"Max RCS: {max_rcs}")
print(f"Std RCS: {std_rcs}")

print('-----------------------------------------------------------------------------------------------')
rcs_astyx_estimated = pd.read_csv("D:/Astyx dataset/dataset_astyx_hires2019/ablation/RCS_validation_results_ablation_para_200_2.csv")
rcs_astyx_real = pd.read_csv("D:/Astyx dataset/dataset_astyx_hires2019/RCS_dataset_test.csv")
rcs_astyx_real['range'] = np.sqrt(rcs_astyx_real['x']**2 + rcs_astyx_real['y']**2 + rcs_astyx_real['z']**2)
merged_df = pd.merge(rcs_astyx_estimated, rcs_astyx_real[['Frame', 'Index', 'rcs', 'range']], on=['Frame', 'Index'], how='left')
merged_df = merged_df.rename(columns={'rcs': 'Real RCS', 'range': 'Range'})
print('Astyx dataset:')
print(f'MSE of RCS in range {range0}~{range1}')
print(calculate_mse_in_range(merged_df, range0, range1))
print(f"RCS Stats for Range between {range0} and {range1}:")
min_rcs, max_rcs, std_rcs, count_in_range = stats_in_range(merged_df, range0, range1)
print(f"Count of entries in range: {count_in_range}")
print(f"Min RCS: {min_rcs}")
print(f"Max RCS: {max_rcs}")
print(f"Std RCS: {std_rcs}")

print(f'MSE of RCS in range {range1}~{range2}')
print(calculate_mse_in_range(merged_df, range1, range2))
print(f"RCS Stats for Range between {range1} and {range2}:")
min_rcs, max_rcs, std_rcs, count_in_range = stats_in_range(merged_df, range1, range2)
print(f"Count of entries in range: {count_in_range}")
print(f"Min RCS: {min_rcs}")
print(f"Max RCS: {max_rcs}")
print(f"Std RCS: {std_rcs}")

print(f'MSE of RCS in range {range2}~{range3}')
print(calculate_mse_in_range(merged_df, range2, range3))
print(f"RCS Stats for Range between {range2} and {range3}:")
min_rcs, max_rcs, std_rcs, count_in_range = stats_in_range(merged_df, range2, range3)
print(f"Count of entries in range: {count_in_range}")
print(f"Min RCS: {min_rcs}")
print(f"Max RCS: {max_rcs}")
print(f"Std RCS: {std_rcs}")



print('-----------------------------------------------------------------------------------------------')
rcs_snail_estimated = pd.read_csv("D:/snail_radar/20231208/data4/ablation/RCS_validation_results_ablation_para_200_3.csv")
rcs_snail_real = pd.read_csv("D:/snail_radar/20231208/data4/RCS_dataset_test.csv")
rcs_snail_real['range'] = np.sqrt(rcs_snail_real['x']**2 + rcs_snail_real['y']**2 + rcs_snail_real['z']**2)
merged_df = pd.merge(rcs_snail_estimated, rcs_snail_real[['Frame', 'Index', 'rcs', 'range']], on=['Frame', 'Index'], how='left')
merged_df = merged_df.rename(columns={'rcs': 'Real RCS', 'range': 'Range'})

print('Snail-Radar dataset:')
print(f'MSE of RCS in range {range0}~{range1}')
print(calculate_mse_in_range(merged_df, range0, range1))
print(f"RCS Stats for Range between {range0} and {range1}:")
min_rcs, max_rcs, std_rcs, count_in_range = stats_in_range(merged_df, range0, range1)
print(f"Count of entries in range: {count_in_range}")
print(f"Min RCS: {min_rcs}")
print(f"Max RCS: {max_rcs}")
print(f"Std RCS: {std_rcs}")

print(f'MSE of RCS in range {range1}~{range2}')
print(calculate_mse_in_range(merged_df, range1, range2))
print(f"RCS Stats for Range between {range1} and {range2}:")
min_rcs, max_rcs, std_rcs, count_in_range = stats_in_range(merged_df, range1, range2)
print(f"Count of entries in range: {count_in_range}")
print(f"Min RCS: {min_rcs}")
print(f"Max RCS: {max_rcs}")
print(f"Std RCS: {std_rcs}")

print(f'MSE of RCS in range {range2}~{range3}')
print(calculate_mse_in_range(merged_df, range2, range3))
print(f"RCS Stats for Range between {range2} and {range3}:")
min_rcs, max_rcs, std_rcs, count_in_range = stats_in_range(merged_df, range2, range3)
print(f"Count of entries in range: {count_in_range}")
print(f"Min RCS: {min_rcs}")
print(f"Max RCS: {max_rcs}")
print(f"Std RCS: {std_rcs}")