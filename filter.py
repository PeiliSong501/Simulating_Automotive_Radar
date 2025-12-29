import utils
import numpy as np
import pandas as pd
import random
import math
import os
import csv
from collections import defaultdict
import open3d as o3d
from scipy.spatial import KDTree
from sample_vod_stats import trackid2pcd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle
import gc
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import multivariate_normal
from sklearn.linear_model import RANSACRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
import joblib

def pcl_subtraction(scene_pcd, pcd):
    ans = []
    for p in scene_pcd:
        if p not in pcd:
            ans.append(p)
    return np.array(ans)

def load_point_cloud(file_index):
    filename = str(file_index)+".npy"
    data_directory = "D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/velodyne_static/"
    file_path = os.path.join(data_directory, filename)
    return np.load(file_path)


def pad_point_cloud(point_cloud, target_size):
    current_size = point_cloud.shape[0]
    if current_size >= target_size:
        return point_cloud[:target_size]  # 如果点数已够，则截取前target_size个点
    else:
        # 计算需要重复的点数
        num_to_repeat = target_size - current_size
        indices = np.random.choice(current_size, num_to_repeat, replace=True)
        padded_points = np.concatenate((point_cloud, point_cloud[indices]), axis=0)
        return padded_points

def compute_mean_covariance(points, location):
    mean = np.mean(points, axis=0)
    covariance = np.cov(points, rowvar=False)
    difference = mean - location

    return mean, covariance, difference

class_list = ['ride_uncertain', 'rider', 'moped_scooter', 'bicycle', 'Cyclist', 'vehicle_other', 'Pedestrian', 'truck',
              'DontCare', 'motor', 'bicycle_rack', 'Car', 'human_depiction', 'ride_other']
for cl in class_list:
    if not os.path.exists(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/gaussian/{cl}'):
        os.mkdir(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/gaussian/{cl}')

# make dataset for Dynamic Objects:
df = pd.read_csv('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/dynamic_objects_total.csv')
# for cl in class_list:
#     df_tmp = df[df['Class'] == cl]
#     df_tmp = df_tmp[df_tmp['PointNum'] >= 3]
#
#     df_tmp.iloc[:,1:].to_csv(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/gaussian/{cl}/{cl}_gaussian_data.csv')
#
# column_names = ['Frame', 'Track_ID', 'mean_x', 'mean_y', 'mean_z', 'covariance11', 'covariance12', 'covariance13',
#            'covariance21', 'covariance22', 'covariance23', 'covariance31', 'covariance32', 'covariance33']
#
# for cl in class_list:
#     print(f'dealing with class {cl}')
#     newdf = pd.DataFrame(columns=column_names)
#     df_tmp = pd.read_csv(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/gaussian/{cl}/{cl}_gaussian_data.csv').values[:,1:]
#     for j in range(len(df_tmp)):
#         frame, track_id = df_tmp[j,0], df_tmp[j,1]
#         location = df_tmp[j,4:7]
#         name = str(frame)+"_"+str(track_id)+".npy"
#         points = np.load('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/gaussian_points/'+name)
#         mean, covariance, difference = compute_mean_covariance(points, location)
#         new_data = pd.DataFrame({'Frame': [frame], 'Track_ID': [track_id], 'mean_x': [mean[0]], 'mean_y': [mean[1]], 'mean_z':[mean[2]],
#                                  'covariance11': [covariance[0,0]], 'covariance12': [covariance[0,1]], 'covariance13':[covariance[0,2]],
#                                  'covariance21': [covariance[1,0]], 'covariance22': [covariance[1,1]], 'covariance23':[covariance[1,2]],
#                                  'covariance31': [covariance[2,0]], 'covariance32': [covariance[2,1]], 'covariance33':[covariance[2,2]]})
#         newdf = pd.concat([newdf, new_data], ignore_index=True)
#     newdf.to_csv(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/gaussian/{cl}/{cl}_gaussian_cooe.csv')






#---------------------------------------fitting density of dynamic objects---------------------------------------------
# df = pd.read_csv('D:/VoD_dataset/dynamic_obj_total.csv')
# # 选择特征和目标变量
# features = df[['t', 'azi', 'ele', 'l', 'w', 'h', 'dop_vel', 'Tar_Ratation']]
# target = df['density']
# X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
# print(f"X_train shape: {X_train.shape}")
# print(f"X_test shape: {X_test.shape}")
# print(f"y_train shape: {y_train.shape}")
# print(f"y_test shape: {y_test.shape}")
# model = RandomForestRegressor(n_estimators=100, random_state=42)
#
# # 拟合模型
# model.fit(X_train, y_train)
#
# # 预测测试集
# y_pred = model.predict(X_test)
#
# # 计算均方误差
# mse = mean_squared_error(y_test, y_pred)
# print(f"Mean Squared Error: {mse}")
#
# r2 = r2_score(y_test, y_pred)
# print(f"R² Score: {r2}")
#
# print(f"Predicted densities: {y_pred[:5]}")
# print(f"Actual densities: {y_test[:5].values}")
# with open('D:/VoD_dataset/random_forest_model.pkl', 'wb') as file:
#     pickle.dump(model, file)
#
# with open('D:/VoD_dataset/random_forest_model.pkl', 'rb') as file:
#     loaded_rf = pickle.load(file)
#---------------------------------------fitting density of dynamic objects---------------------------------------------




#--------------------------------------fitting Gaussian Distribution of Dynamic Objects---------------------------------
for cl in class_list:
    print('dealing with class ',cl)
    dir_path = f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/gaussian/{cl}'
    if not os.path.exists(dir_path):
        continue
    data_file = os.path.join(dir_path, f'{cl}_gaussian_data.csv')
    cooe_file = os.path.join(dir_path, f'{cl}_gaussian_cooe.csv')

    data_df = pd.read_csv(data_file)
    cooe_df = pd.read_csv(cooe_file)

    if len(data_df) == 0:
        continue

    X = data_df[['Rotation', 'Location_x', 'Location_y', 'Location_z',
                 'Dimension_x', 'Dimension_y', 'Dimension_z', 'v_r_real']].values

    Y = cooe_df[['mean_x', 'mean_y', 'mean_z',
                 'covariance11', 'covariance12', 'covariance13',
                 'covariance21', 'covariance22', 'covariance23',
                 'covariance31', 'covariance32', 'covariance33']].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    svm_model = MultiOutputRegressor(SVR())
    svm_model.fit(X_train, Y_train)

    Y_train_pred = svm_model.predict(X_train)
    Y_test_pred = svm_model.predict(X_test)

    train_mse = mean_squared_error(Y_train, Y_train_pred)
    test_mse = mean_squared_error(Y_test, Y_test_pred)

    print(f'Training MSE for class {cl}: {train_mse}')
    print(f'Test MSE for class {cl}: {test_mse}')

    for i, column in enumerate(cooe_df.columns[3:]):    #ignore unnamed0, frame, track_id
        train_mse_dim = mean_squared_error(Y_train[:, i], Y_train_pred[:, i])
        test_mse_dim = mean_squared_error(Y_test[:, i], Y_test_pred[:, i])
        print(f'  MSE for {column} in class {cl} - Training: {train_mse_dim}, Test: {test_mse_dim}')

    # 模型保存路径
    model_save_path = os.path.join(dir_path, f'{cl}_svm_model.pkl')

    # 保存模型
    joblib.dump(svm_model, model_save_path)

    print(f'Model for class {cl} saved at {model_save_path}')
    #loaded_model = joblib.load(model_path)


#--------------------------------------fitting Gaussian Distribution of Dynamic Objects---------------------------------






