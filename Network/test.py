import cv2
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import os
from torch.utils.data import DataLoader
#from RNet import RNet
# from dataset import Dis_Dataset_VoD, Dis_Dataset_astyx, Dis_Dataset_Snail, Dis_Dataset_Snail_eagle
from dataset_RPGen import Dataset_VoD, Dataset_astyx, Dataset_Snail, Dataset_Snail_eagle, Dataset_VoD_Depth, Dataset_MSC_Depth, Dataset_hercules_Depth
import random
from torch import optim
import argparse
import time
import torch.nn.functional as F
from Cls_Reg_Net import *
from DisNet import DisNet, KLDivergenceLoss
import re
from PIL import Image



def seed_everything(seed=2024):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def frame_index(filename):
    match = re.search(r'(\d+)_(\d+)\.jpg', filename)
    if not match:
        raise ValueError(f"Filename {filename} does not match the expected pattern 'frame_index.jpg'")
    frame = int(match.group(1))
    index = int(match.group(2))
    return frame, index
def files2df(files,df):
    train_data = []
    for filename in files:
        frame, index = frame_index(filename)
        row_mask = (df[:, 0] == frame) & (df[:, 1] == index)
        matching_rows = df[row_mask]
        if matching_rows.size > 0:
            train_data.append(matching_rows)
    if train_data:
        train_df = np.vstack(train_data)
    else:
        train_df = np.array([])
    return train_df


def save_images(output, frame, save_dir):
    # 创建保存目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)

    # 假设 output 的 shape 为 [8, 1, 40, 40]
    batch_size, channels, height, width = output.shape
    x_min = output.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
    x_max = output.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]

    output = (output - x_min) / (x_max - x_min + 1e-8)
    output = output.clamp(0, 1)
    # 遍历每个图像
    for i in range(batch_size):
        img = output[i, 0].cpu().detach().numpy()  # 取出第 i 张图像，并转换为 NumPy 数组
        img = (img * 255).astype(np.uint8)  # 将图像缩放到 [0, 255] 范围并转换为 uint8 类型

        filename = f'{frame[i]}.jpg'
        filepath = os.path.join(save_dir, filename)
        cv2.imwrite(filepath, img)
        #print(f'Saved {filepath}')


def pixel_mse_loss(output_gray, criterion, target):
    x_min = output_gray.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
    x_max = output_gray.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    output_gray = (output_gray - x_min) / (x_max - x_min + 1e-8)
    output_gray = output_gray.clamp(0, 1)

    loss0 = criterion(output_gray, target / 255)
    return loss0

def test_dis(args, dataset, log, weights = None):
    base_dir = ""
    frame_df = None
    if dataset == 'VoD':
        base_dir = '/workspace/data/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/'
    elif dataset == 'astyx':
        base_dir = 'D:/Astyx dataset/dataset_astyx_hires2019/'
    elif dataset == 'snail_radar':
        base_dir = 'D:/snail_radar/20231208/data4/'
        columns_to_read_as_str = ['timestamp', 'cam_left_time', 'cam_right_time', "radar_arg_time", "radar_eagle_time",
                                  'lidar_time', 'odom_time']
        frame_df = pd.read_csv('D:/snail_radar/20231208/data4/frames_timestamps.csv', usecols=columns_to_read_as_str,
                         dtype=str).values
    elif dataset == 'snail_radar_eagle':
        base_dir = 'D:/snail_radar/20231208/data4/eagle/'
        columns_to_read_as_str = ['timestamp', 'cam_left_time', 'cam_right_time', "radar_arg_time", "radar_eagle_time",
                                  'lidar_time', 'odom_time']
        frame_df = pd.read_csv('D:/snail_radar/20231208/data4/frames_timestamps.csv', usecols=columns_to_read_as_str,
                         dtype=str).values

    device = torch.device(args.device)
    batch_size = args.batch_size

    df = pd.read_csv(base_dir+'prob_dataset.csv').values
    train_data = pd.read_csv(base_dir+'train_dataset.csv').values
    valid_data = pd.read_csv(base_dir+'test_dataset.csv').values
    train_dataloader,test_dataloader = None, None
    original_size = None
    if dataset == 'VoD':
        train_dataloader = DataLoader(dataset=Dataset_VoD_Depth(train_data),
                                      batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers = 4)
        original_size = [644,1935]
    elif dataset == 'astyx':
        train_dataloader = DataLoader(dataset=Dis_Dataset_astyx(train_data),
                                      batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers = 4)
        original_size = [618,2048]
    elif dataset == 'snail_radar':
        train_dataloader = DataLoader(dataset=Dis_Dataset_Snail(train_data,frame_df),
                                      batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers = 4)
        original_size = [316, 640]
    elif dataset == 'snail_radar_eagle':
        train_dataloader = DataLoader(dataset=Dis_Dataset_Snail_eagle(train_data,frame_df),
                                      batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers = 4)
        original_size = [316, 640]

    model = torch.load(weights).to(device)


    criterion = torch.nn.MSELoss(reduction='mean')
    kl_loss = KLDivergenceLoss()
    # optimizer = optim.Adam(model.parameters(), lr=1e-4)

    min_val_loss = 10000000



    model.eval()
    # valid_dataloader = DataLoader(dataset=Dis_Dataset_VoD(valid_data),
    #                               batch_size=args.batch_size, shuffle=True, drop_last=False)
    if dataset == 'VoD':
        valid_dataloader = DataLoader(dataset=Dataset_VoD_Depth(valid_data),
                                        batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers = 4)
    elif dataset == 'astyx':
        valid_dataloader = DataLoader(dataset=Dis_Dataset_astyx(valid_data),
                                        batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers = 4)
    elif dataset == 'snail_radar':
        valid_dataloader = DataLoader(dataset=Dis_Dataset_Snail(valid_data, frame_df),
                                        batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers = 4)
    elif dataset == 'snail_radar_eagle':
        valid_dataloader = DataLoader(dataset=Dis_Dataset_Snail_eagle(valid_data, frame_df),
                                        batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers = 4)
    total_val_loss = 0
    total_val_loss1, total_val_loss2 = 0, 0
    total_val_loss0 = 0
    total_val_num = 0
    total_loss2_percentage = 0
    cnt = 0
    column_names = ['Frame', 'Estimated_PointNum']
    new_df = pd.DataFrame(columns=column_names)
    with torch.no_grad():
        for idx, (batch_dict, target, pointnum) in enumerate(valid_dataloader):
            target = target.to(device)
            target = target.unsqueeze(1).float()
            pointnum = pointnum.to(device)
            frame = batch_dict['frame'].reshape(len(target),-1).cpu().detach().numpy()

            output_gray, output_num = model(batch_dict)
            output_gray, output_num = output_gray.float(), output_num.float()

            # output_gray = model(batch_dict)
            # output_gray = output_gray.float()


            output_num = output_num.reshape(-1, 1)
            pointnum = pointnum.reshape(-1, 1)

            num_est = output_num.cpu().detach().numpy()*1000

            for i in range(len(num_est)):
                #print(frame[i,0])
                new_data = pd.DataFrame({'Frame': [frame[i,0]], 'Estimated_PointNum': [num_est[i,0]]})
                # new_df = pd.concat([new_df, new_data], ignore_index=True)
                if not new_data.empty and not new_data.isna().all(axis=None):
                    new_df = pd.concat([new_df, new_data], ignore_index=True)

            loss1 = kl_loss(output_gray, target)


            #loss0 = criterion(output_gray, target/255)

            #print(output_num, pointnum)
            #loss2 = criterion(output_num, pointnum/1000)
            #loss2_true = criterion(output_num*1000, pointnum)
            error = abs(output_num * 1000 - pointnum) / pointnum
            loss2_percentage = criterion(error, torch.zeros_like(error))
            total_loss2_percentage += loss2_percentage.item()


            frame = batch_dict['frame']
            if dataset == 'VoD':
                save_images(output_gray, frame,f'/workspace/data/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/exp_2/test_image_{log}/')
            if dataset == 'astyx':
                save_images(output_gray,frame,'D:/Astyx dataset/dataset_astyx_hires2019/ablation/test_image_wo_v/')
            elif dataset == ('snail_radar'):
                save_images(output_gray, frame, 'D:/snail_radar/20231208/data4/ablation/test_image_wo_F/')
            elif dataset == ('snail_radar_eagle'):
                save_images(output_gray, frame, 'D:/snail_radar/20231208/data4/eagle/ablation/test_image_wo_v/')


            loss = loss1 + loss2_percentage
            total_val_loss += loss.item()
            total_val_loss1 += loss1.item()
            cnt += 1

        total_val_loss /= cnt
        # if total_val_loss < min_val_loss:
        #     min_val_loss = total_val_loss
        #     new_df.to_csv(base_dir+f'exp_2/estimated_pointnum_{log}.csv')
            
        print('validation set:')
        print("min total loss = ", min_val_loss)
        print(f'avg KL Divergence loss: ', total_val_loss1 / cnt)
        print(f'avg number percentage loss: ', total_loss2_percentage / cnt)






if __name__ == '__main__':
    seed_everything()
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default="cuda:3",
                        help='use gpu')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (per gpu).')
    args = parser.parse_args()
    weights = None
    dataset = 'VoD'
    log_name = 'radar_depth_w_smooth_loss_2'
    print(f'testing on {log_name}')
    # if dataset == 'VoD':
    #     weights = f'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/ablation/weights/ablation_wo_v_model_best.pth'
    # elif dataset == 'astyx':
    #     weights = f'D:/Astyx dataset/dataset_astyx_hires2019/ablation/weights/ablation_wo_F_model_best.pth'
    # elif dataset == 'snail_radar':
    #     weights = f'D:/snail_radar/20231208/data4/ablation/weights/ablation_wo_F_model_best.pth'
    # elif dataset == 'snail_radar_eagle':
    #     weights = f'D:/snail_radar/20231208/data4/eagle/ablation/weights/ablation_wo_v_model_best.pth

    path_VoD = f'/workspace/data/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/exp_RPGen/weights/model_best_{dataset}_{log_name}/model_best_{dataset}_{log_name}_e28.pth'
    if dataset == 'VoD':
        # print(path_VoD)
        weights = path_VoD
        #weights = f'/workspace/data/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/exp_2/weights/model_best_transformer.pth'
    elif dataset == 'astyx':
        weights = f'D:/Astyx dataset/dataset_astyx_hires2019/exp_2/weights/ablation_wo_F_model_best.pth'
    elif dataset == 'snail_radar':
        weights = f'D:/snail_radar/20231208/data4/exp_2/weights/ablation_wo_F_model_best.pth'
    elif dataset == 'snail_radar_eagle':
        weights = f'D:/snail_radar/20231208/data4/eagle/exp_2/weights/ablation_wo_v_model_best.pth'
    if os.path.exists(weights):
        test_dis(args,dataset,log_name, weights)
    else:
        print('weight not found.')
