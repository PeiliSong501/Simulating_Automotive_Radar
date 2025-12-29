import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import os
from torch.utils.data import DataLoader

from dataset import RCS_Dataset_VoD, RCS_Dataset_astyx, RCS_Dataset_Snail
import random
import matplotlib.pyplot as plt
import seaborn as sns
import utils
from collections import defaultdict
#import gt_generation_p as utils1
from scipy.spatial import KDTree
from scipy.stats import ks_2samp


def plot_class_distribution(class_count, title, file_name):
    # 对分类结果按数量排序
    sorted_class_count = sorted(class_count.items(), key=lambda item: item[1], reverse=True)

    labels, values = zip(*sorted_class_count)

    # 创建柱状图
    plt.figure(figsize=(12, 6))  # 调整图表大小，宽度设大一些
    plt.bar(labels, values, color='skyblue')
    plt.title(title)
    plt.xlabel('Classes')
    plt.ylabel('Counts')

    # 旋转 X 轴标签以避免重叠
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()  # 自动调整布局以防止重叠
    plt.savefig(file_name, format='svg')
    plt.show()
# def plot_class_distribution(class_count, title, save_path):
#     # 提取类别和对应数量
#     classes = list(class_count.keys())
#     counts = list(class_count.values())
#
#     # 创建柱状图
#     plt.figure(figsize=(10, 6))
#     plt.bar(classes, counts, color='skyblue')
#
#     # 设置标题和标签
#     plt.title(f'Point Class Distribution - {title}', fontsize=16)
#     plt.xlabel('Class', fontsize=12)
#     plt.ylabel('Number of Points', fontsize=12)
#
#     # 显示数量标签
#     for i, count in enumerate(counts):
#         plt.text(i, count + 0.5, str(count), ha='center', fontsize=10)
#
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#
#     # 保存为矢量图 (SVG)
#     plt.savefig(save_path, format='svg')
#
#     # 显示图形
#     plt.show()


if __name__ == '__main__':
    #dynamic_object = pd.read_csv('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/dynamic_objects_total.csv').values[:,1:]
    # RCS_train = pd.read_csv('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/RCS_dataset_train.csv').values[:,3:6]
    # RCS_test = pd.read_csv(
    #     'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/RCS_dataset_test.csv').values[:, 3:6]
    # # 创建 Pandas DataFrame 以便于可视化
    # train_df = pd.DataFrame(RCS_train, columns=['x', 'y', 'z'])
    # test_df = pd.DataFrame(RCS_test, columns=['x', 'y', 'z'])
    #
    # # 设置可视化风格
    # sns.set(style="whitegrid")
    #
    # # 创建多个子图
    # fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    # fig.suptitle('X, Y, Z Distribution of RCS_Train, RCS_Test', fontsize=16)
    #
    # # 绘制 RCS_train 数据的分布
    # sns.histplot(train_df['x'], kde=True, ax=axs[0, 0], color='blue')
    # axs[0, 0].set_title('RCS_train: X axis')
    #
    # sns.histplot(train_df['y'], kde=True, ax=axs[1, 0], color='green')
    # axs[1, 0].set_title('RCS_train: Y axis')
    #
    # sns.histplot(train_df['z'], kde=True, ax=axs[2, 0], color='red')
    # axs[2, 0].set_title('RCS_train: Z axis')
    #
    # # 绘制 RCS_test 数据的分布
    # sns.histplot(test_df['x'], kde=True, ax=axs[0, 1], color='blue')
    # axs[0, 1].set_title('RCS_test: X axis')
    #
    # sns.histplot(test_df['y'], kde=True, ax=axs[1, 1], color='green')
    # axs[1, 1].set_title('RCS_test: Y axis')
    #
    # sns.histplot(test_df['z'], kde=True, ax=axs[2, 1], color='red')
    # axs[2, 1].set_title('RCS_test: Z axis')
    #
    # # 设置子图之间的布局
    # plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.savefig("C:/Users/PerrySong_nku/Desktop/writing/RCS_train_vs_test_distribution.svg", format='svg')
    #
    # plt.show()

    # RCS_train = pd.read_csv('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/RCS_dataset_train.csv')
    # RCS_test = pd.read_csv('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/RCS_dataset_test.csv')

    # RCS_train = pd.read_csv('D:/Astyx dataset/dataset_astyx_hires2019/RCS_dataset_train.csv')
    # RCS_test = pd.read_csv('D:/Astyx dataset/dataset_astyx_hires2019/RCS_dataset_test.csv')

    # RCS_train = pd.read_csv('D:/snail_radar/20231208/data4/RCS_dataset_train.csv')
    # RCS_test = pd.read_csv('D:/snail_radar/20231208/data4/RCS_dataset_test.csv')

    # from sklearn.metrics import mean_squared_error
    # min_value, max_value = -74.01181, 39.788345
    # #test1 = pd.read_csv('D:/VoD_dataset/RCS_validation_results_ablation_para_50_0.5.csv')
    # test1 = pd.read_csv('D:/VoD_dataset/RCS_validation_results_ablation_para_100_2.csv')
    # labels = test1.values[:,2]
    # preds = test1.values[:,3]
    # mse_direct = mean_squared_error(labels, preds)
    #
    # # 2. 最大最小值标准化后计算 MSE
    # labels_normalized = (labels - min_value) / (max_value - min_value)
    # preds_normalized = (preds - min_value) / (max_value - min_value)
    # mse_normalized = mean_squared_error(labels_normalized, preds_normalized)
    #
    # print("直接计算 MSE:", mse_direct)
    # print("标准化后计算 MSE:", mse_normalized)
    #------------------------------------------------dataset augmentation-----------------------------------------------
    # RCS_set = pd.concat((RCS_train, RCS_test), axis=0)
    # rcs_values = RCS_set['rcs']
    # print(RCS_train.shape)
    # total_length = len(RCS_train)
    # min_length = total_length // 12
    #
    # # 定义 RCS 组的范围
    # min_rcs = rcs_values.min()
    # max_rcs = rcs_values.max()
    #
    # # 自动生成 rcs_bins，按每10个数值划分
    # bin_edges = np.arange(min_rcs, max_rcs + 10, 10)
    # rcs_bins = [(bin_edges[i], bin_edges[i + 1]) for i in range(len(bin_edges) - 1)]
    #
    # # 用于存储扩展后的数据
    # expanded_data = []
    #
    # # 遍历每个组
    # for lower_bound, upper_bound in rcs_bins:
    #     group_data = RCS_train[(RCS_train['rcs'] >= lower_bound) & (RCS_train['rcs'] < upper_bound)]
    #     group_length = len(group_data)
    #
    #     if group_length < min_length:
    #         # 复制比例扩展
    #         num_copies = min_length // group_length
    #         expanded_data.extend([group_data] * num_copies)
    #     else:
    #         # 随机抽样
    #         sampled_data = group_data.sample(n=min_length, replace=False)
    #         expanded_data.append(sampled_data)
    #
    # # 将扩展后的数据合并为一个 DataFrame
    # expanded_RCS_train = pd.concat(expanded_data, ignore_index=True)
    #
    # # 随机打乱扩展后的数据
    # expanded_RCS_train = expanded_RCS_train.sample(frac=1).reset_index(drop=True)
    #
    # # 保存为 CSV 文件
    # expanded_RCS_train.to_csv(
    #     'D:/snail_radar/20231208/data4/expanded_RCS_dataset_train.csv', index=False)
    #
    # # 打印扩展后的数据集信息
    # print("Expanded dataset shape:", expanded_RCS_train.shape)
    #
    # expanded_rcs_values = expanded_RCS_train['rcs']
    #
    # # 使用之前定义的组范围
    # expanded_group_counts = {}
    #
    # for lower_bound, upper_bound in rcs_bins:
    #     count = expanded_rcs_values[(expanded_rcs_values >= lower_bound) & (expanded_rcs_values < upper_bound)].count()
    #     expanded_group_counts[f"{lower_bound}--{upper_bound}"] = count
    #
    # # 转换为 DataFrame 方便查看
    # expanded_group_counts_df = pd.DataFrame(list(expanded_group_counts.items()), columns=['RCS Group', 'Count'])
    #
    # # 打印每组的出现次数
    # print("Expanded RCS Value Groups and Counts:")
    # print(expanded_group_counts_df)



    # 读取dynamic_objects_total.csv
    # dynamic_object = pd.read_csv(
    #     'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/dynamic_objects_total.csv')



    #RCS_set = pd.concat((RCS_train,RCS_test),axis=0)
    #print(np.min(RCS_train.values[:,-1]),np.max(RCS_train.values[:,-1]),np.mean(RCS_train.values[:,-1]),np.std(RCS_train.values[:,-1]))

    #------------------------------save subset of objects to test RCS distribution--------------------------------------
    # matched_objects = []
    # def find_matching_objects(RCS_data, dynamic_object):
    #     grouped_RCS_data = RCS_data.groupby('Frame')
    #     grouped_dynamic_object = dynamic_object.groupby('Frame')
    #
    #     total_frames = len(grouped_RCS_data)
    #     checkpoint = total_frames // 10  # 每处理10%帧作为一个进度点
    #
    #     for idx, (frame, frame_RCS_points) in enumerate(grouped_RCS_data):
    #         if frame in grouped_dynamic_object.groups:  # 仅处理相同帧的数据
    #             frame_dynamic_objects = grouped_dynamic_object.get_group(frame)
    #
    #             for _, point in frame_RCS_points.iterrows():
    #                 point_cloud = np.array([[point['x'], point['y'], point['z']]])  # 获取点的坐标
    #                 point_classified = False
    #
    #                 # 遍历同一帧的 dynamic object 的 bounding box
    #                 for _, obj in frame_dynamic_objects.iterrows():
    #                     bbox_location = [obj['Location_x'], obj['Location_y'], obj['Location_z']]
    #                     bbox_dimensions = [obj['Dimension_x'], obj['Dimension_y'], obj['Dimension_z']]
    #                     bbox_yaw = obj['Rotation']
    #
    #                     # 判断点是否在 bounding box 内
    #                     if len(utils.filter_points_in_bbox(point_cloud, bbox_location, bbox_dimensions, bbox_yaw)[0]) > 0:
    #                         # 匹配成功，保存物体信息
    #                         matched_objects.append(obj)
    #                         point_classified = True
    #                         break  # 一旦点已分类，无需继续判断
    #
    #         # 输出进度：每处理 10% 的帧输出一次进度
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
    # output_csv_path = 'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/matched_dynamic_objects_subset.csv'
    # matched_objects_df.to_csv(output_csv_path, index=False)
    #
    # print(f"Subset of matched dynamic objects saved to {output_csv_path}")




    # ------------------------------save classes of targets within RCS_train, RCS_test----------------------------------
    # # 定义用于存储分类结果的字典
    # class_count_train = defaultdict(int)
    # class_count_test = defaultdict(int)
    #
    #
    # # 定义函数来处理分类逻辑，基于帧匹配
    # def classify_points(RCS_data, dynamic_object, class_count):
    #     grouped_RCS_data = RCS_data.groupby('Frame')
    #     grouped_dynamic_object = dynamic_object.groupby('Frame')
    #
    #     total_frames = len(grouped_RCS_data)
    #     checkpoint = total_frames // 10  # 每处理10%帧作为一个进度点
    #
    #     for idx, (frame, frame_RCS_points) in enumerate(grouped_RCS_data):
    #         if frame in grouped_dynamic_object.groups:  # 仅处理相同帧的数据
    #             frame_dynamic_objects = grouped_dynamic_object.get_group(frame)
    #             for _, point in frame_RCS_points.iterrows():
    #                 point_cloud = np.array([[point['x'], point['y'], point['z']]])  # 获取点的坐标
    #                 point_classified = False
    #
    #                 # 遍历同一帧的 dynamic object 的 bounding box
    #                 for _, obj in frame_dynamic_objects.iterrows():
    #                     bbox_location = [obj['Location_x'], obj['Location_y'], obj['Location_z']]
    #                     bbox_dimensions = [obj['Dimension_x'], obj['Dimension_y'], obj['Dimension_z']]
    #                     bbox_yaw = obj['Rotation']
    #
    #                     # 判断点是否在 bounding box 内
    #                     if len(utils.filter_points_in_bbox(point_cloud, bbox_location, bbox_dimensions, bbox_yaw)[0]) > 0:
    #                         class_count[obj['Class']] += 1
    #                         point_classified = True
    #                         break  # 一旦点已分类，无需继续判断
    #
    #                 # 如果该点没有分类到任何 dynamic object，将其归为 Background
    #                 # if not point_classified:
    #                 #     class_count['Background'] += 1
    #
    #         # 进度输出：每处理 10% 的帧输出一次进度
    #         if (idx + 1) % checkpoint == 0:
    #             progress = (idx + 1) / total_frames * 100
    #             print(f"Progress: {progress:.1f}%")
    #
    #     print("Classification completed!")
    #
    #
    # # 对RCS_train和RCS_test进行分类统计
    # print('Dealing with RCS_train')
    # classify_points(RCS_train, dynamic_object, class_count_train)
    #
    # print('Dealing with RCS_test')
    # classify_points(RCS_test, dynamic_object, class_count_test)
    #
    # # 输出分类结果
    # print("RCS_train的点分类统计：")
    # for cls, count in class_count_train.items():
    #     print(f"{cls}: {count}")
    #
    # print("\nRCS_test的点分类统计：")
    # for cls, count in class_count_test.items():
    #     print(f"{cls}: {count}")
    #
    # plot_class_distribution(class_count_train, "Target Distribution of RCS_train", "C:/Users/PerrySong_nku/Desktop/writing/RCS_train_distribution.svg")
    # plot_class_distribution(class_count_test, "Target Distribution of RCS_test", "C:/Users/PerrySong_nku/Desktop/writing/RCS_test_distribution.svg")

    # df = pd.read_csv(
    #     'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/matched_dynamic_objects_subset.csv').iloc[:, 1:]
    # df['valid'] = 0  # 初始化 valid 列
    #
    # # 遍历数据并计算 valid 列
    # for i in range(len(df)):
    #     Frame, Track_ID, Class, Rotation, Location_x, Location_y, Location_z, PointNum, Dimension_x, Dimension_y, Dimension_z, v_r_real, _ = \
    #     df.iloc[i]
    #     l_in_radar_coor = np.load(
    #         f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/l_in_radar_coor/{Frame}.npy')
    #     kdtree = KDTree(l_in_radar_coor)
    #     bbox_location = [Location_x, Location_y, Location_z]
    #     bbox_dimensions = [Dimension_x, Dimension_y, Dimension_z]
    #     bbox_yaw = Rotation
    #     r_in = np.load(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/r_in/{Frame}.npy')
    #     r_in_2d = np.load(f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/r_in_2d/{Frame}.npy')
    #
    #     r_in_local, mask = utils.filter_points_in_bbox(r_in, bbox_location, bbox_dimensions, bbox_yaw)
    #
    #     # 计算 valid 值
    #     if len(r_in_local) < 3:
    #         df.at[i, 'valid'] = 0
    #     else:
    #         df.at[i, 'valid'] = 1
    #         # 保存 r_in_local
    #         np.save(
    #             f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/rcs_dis_test/object_local_points/{Frame}_{Track_ID}.npy',
    #             r_in_local)
    #         r_in_2d_local = r_in_2d[mask]
    #
    # # 保存更新后的 DataFrame
    # df.to_csv('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/expanded_new_df_test.csv', index=False)


        # for j in range(len(r_in_local)):
        #     x,y,z,v_r,rcs = r_in_local[j,:5]
        #     u,v = r_in_2d_local[j,:]
        #     index = i*1000+j
        #     new_data = pd.DataFrame(
        #                     {'Frame': [Frame], 'Index': [index], 'x':[x], 'y':[y], 'z':[z], 'u':[u], 'v':[v], 'v_r':[v_r], 'rcs':[rcs]})
        #     new_df_test = pd.concat([new_df_test, new_data], ignore_index=True)
        #     # set index to be i, so the local points of a specific object can be retrieved easily
        #     local_lidar_index = kdtree.query_ball_point([x, y, z], r=1)
        #     local_lidar_radarcoor = l_in_radar_coor[local_lidar_index]
        #     dst = f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/rcs_dis_test/range_image/{Frame}_{index}.jpg'
        #     virtual_point = utils1.get_new_point(x, y, z)
        #     fov_down = 7
        #     fov_up = -23
        #     for k in range(len(local_lidar_radarcoor)):
        #         local_lidar_radarcoor[k, 0] -= virtual_point[0]
        #         local_lidar_radarcoor[k, 1] -= virtual_point[1]
        #         local_lidar_radarcoor[k, 2] -= virtual_point[2]
        #         x1, y1, z1 = local_lidar_radarcoor[k]
        #         pitch = np.degrees(np.arctan(z1 / np.linalg.norm([x1,y1],2)))
        #         if pitch > fov_up:
        #             fov_up = pitch
        #         if pitch < fov_down:
        #             fov_down = pitch
        #
        #     proj_H, proj_W, = 32, 128
        #     utils1.gen_range_image_rcs(local_lidar_radarcoor, fov_up, fov_down, proj_H, proj_W,
        #                     dst)


    #-------------------------------test if generated RCS belongs to a distribution----------------------------------
    df = pd.read_csv(
        'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/matched_dynamic_objects_subset_new.csv')
    arr = df[df['valid'] == 1].values

    # df_rcs_valid = pd.read_csv(
    #     'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/ablation/RCS_validation_results_ablation_para_150_2.csv')
    df_rcs_valid = pd.read_csv("D:/VoD_dataset/RCS_validation_results_ablation_rgb_li_fe.csv")
    df_rcs_points = pd.read_csv('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/RCS_dataset_test.csv')

    success_cnt = 0
    fail_cnt = 0
    total_cnt = 0

    for i in range(len(arr)):
        Frame, Track_ID, Class, Rotation, Location_x, Location_y, Location_z, PointNum, Dimension_x, Dimension_y, Dimension_z, v_r_real, _ = \
        arr[i]

        # Get indices
        indices = df_rcs_valid[df_rcs_valid['Frame'] == Frame]['Index'].values

        # Get point cloud data
        point_cloud = df_rcs_points[df_rcs_points['Frame'] == Frame][['x', 'y', 'z', 'rcs', 'v_r']].values

        # Set bounding box
        bbox_location = [Location_x, Location_y, Location_z]
        bbox_dimensions = [Dimension_x, Dimension_y, Dimension_z]
        bbox_yaw = Rotation

        # Filter point cloud
        test_p_in, mask = utils.filter_points_in_bbox(point_cloud, bbox_location, bbox_dimensions, bbox_yaw)

        # Load local points
        local_points = np.load(
            f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/rcs_dis_test/object_local_points/{Frame}_{Track_ID}.npy')

        # Get true RCS values
        rcs_true = local_points[:, 3]

        # Get predicted RCS values for test points
        rcs_pred_full = df_rcs_valid[(df_rcs_valid['Frame'] == Frame) & (df_rcs_valid['Index'].isin(indices))][
            'Predictions'].values

        # Filter rcs_pred_full to keep only the predictions corresponding to test_p_in
        rcs_pred = rcs_pred_full[mask] if len(mask) == len(rcs_pred_full) else []

        # Check if predicted values come from the same distribution as true values
        if len(rcs_true) > 0 and len(rcs_pred) > 0:
            statistic, p_value = ks_2samp(rcs_true, rcs_pred)

            # Determine if they come from the same distribution based on p-value
            if p_value > 0.05:
                success_cnt += 1
                # print(
                #     f'Frame {Frame}, Track ID {Track_ID}: RCS predictions and true values come from the same distribution (p={p_value:.4f})')
                # print(len(rcs_true), len(rcs_pred))
                # print('True RCS:', rcs_true)
                # print('Predicted RCS:', rcs_pred)
            else:
                fail_cnt += 1
                # print(
                #     f'Frame {Frame}, Track ID {Track_ID}: RCS predictions and true values do not come from the same distribution (p={p_value:.4f})')

            total_cnt += 1

    print('Success rate:', success_cnt / total_cnt)
    print('Success, Fail, Total:', success_cnt, fail_cnt, total_cnt)