import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import os
from torch.utils.data import DataLoader
from RCSNet import RCSNet
from dataset import RCS_Dataset_VoD, RCS_Dataset_astyx, RCS_Dataset_Snail, RCS_Dataset_Snail_eagle, RCS_Dataset_MSC
import random
from torch import optim
import argparse
import time
import logging

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,  # 设置日志级别为 INFO
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),  # 使用 mode='w' 覆盖已有内容
            logging.StreamHandler()  # 同时在控制台显示日志
        ]
    )


def seed_everything(seed=2024):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train_rcs(args, epochs, dataset, ablation, rc, weights = None):
    if not os.path.exists("./rcs_log/"):
        os.mkdir('./rcs_log/')
    log_file = f"./rcs_log/training_log_{dataset}_{ablation}.txt"
    setup_logging(log_file)
    logging.info(f"Training started for dataset: {dataset}")
    logging.info(f"Batch size: {args.batch_size}, Epochs: {epochs}")

    base_dir = ""
    frame_df = None
    min_value, max_value = 0,0
    if dataset == 'VoD':
        min_value, max_value =-74.01181,39.788345
        base_dir = '/workspace/data/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/'
    elif dataset == 'astyx':
        min_value, max_value = 45.0, 95.0
        base_dir = 'D:/Astyx dataset/dataset_astyx_hires2019/'
    elif dataset == 'snail_radar':
        min_value, max_value = 108, 178
        base_dir = 'D:/snail_radar/20231208/data4/'
        columns_to_read_as_str = ['timestamp', 'cam_left_time', 'cam_right_time', "radar_arg_time", "radar_eagle_time",
                                  'lidar_time', 'odom_time']
        frame_df = pd.read_csv('D:/snail_radar/20231208/data4/frames_timestamps.csv', usecols=columns_to_read_as_str,
                               dtype=str).values
    elif dataset == 'snail_radar_eagle':
        min_value, max_value = 0.0, 38.13
        base_dir = 'D:/snail_radar/20231208/data4/eagle/'
        columns_to_read_as_str = ['timestamp', 'cam_left_time', 'cam_right_time', "radar_arg_time", "radar_eagle_time",
                                  'lidar_time', 'odom_time']
        frame_df = pd.read_csv('D:/snail_radar/20231208/data4/frames_timestamps.csv', usecols=columns_to_read_as_str,
                               dtype=str).values
    elif dataset == 'MSC':
        min_value, max_value = 1.5049268, 54.761814
        base_dir = '/workspace/data/MSC-Rad4R/URBAN_D0/pmap_dataset/'

    batch_size = args.batch_size
    train_dataloader, valid_dataloader = None, None
    if dataset == 'VoD':
        train_data = pd.read_csv(
            '/workspace/data/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/RCS_dataset_train.csv').values[:,1:]
        valid_data = pd.read_csv(
            '/workspace/data/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/RCS_dataset_test.csv').values[:,1:]
        train_dataloader = DataLoader(dataset=RCS_Dataset_VoD(train_data,rc),
                                      batch_size=args.batch_size, shuffle=True, drop_last=False,num_workers=4)
        valid_dataloader = DataLoader(dataset=RCS_Dataset_VoD(valid_data,rc),
                                      batch_size=args.batch_size, shuffle=True, drop_last=False,num_workers=4)
    elif dataset == 'astyx':
        train_data = pd.read_csv(
            '/workspace/data/Astyx dataset/dataset_astyx_hires2019/RCS_dataset_train.csv').values[:, 1:]
        valid_data = pd.read_csv(
            'D:/Astyx dataset/dataset_astyx_hires2019/RCS_dataset_test.csv').values[:, 1:]
        train_dataloader = DataLoader(dataset=RCS_Dataset_astyx(train_data),
                                  batch_size=args.batch_size, shuffle=True, drop_last=False,num_workers=4)
        valid_dataloader = DataLoader(dataset=RCS_Dataset_astyx(valid_data),
                                      batch_size=args.batch_size, shuffle=True, drop_last=False,num_workers=4)
    elif dataset == 'snail_radar':
        train_data = pd.read_csv(
            'D:/snail_radar/20231208/data4/RCS_dataset_train.csv').values[:, 1:]
        valid_data = pd.read_csv(
            'D:/snail_radar/20231208/data4/RCS_dataset_test.csv').values[:, 1:]
        train_dataloader = DataLoader(dataset=RCS_Dataset_Snail(train_data, frame_df),
                                  batch_size=args.batch_size, shuffle=True, drop_last=False,num_workers=4)
        valid_dataloader = DataLoader(dataset=RCS_Dataset_Snail(valid_data, frame_df),
                                      batch_size=args.batch_size, shuffle=True, drop_last=False,num_workers=4)
    elif dataset == 'snail_radar_eagle':
        train_data = pd.read_csv(
            'D:/snail_radar/20231208/data4/eagle/RCS_dataset_train.csv').values[:, 1:]
        valid_data = pd.read_csv(
            'D:/snail_radar/20231208/data4/eagle/RCS_dataset_test.csv').values[:, 1:]
        train_dataloader = DataLoader(dataset=RCS_Dataset_Snail_eagle(train_data, frame_df),
                                  batch_size=args.batch_size, shuffle=True, drop_last=False,num_workers=4)
        valid_dataloader = DataLoader(dataset=RCS_Dataset_Snail_eagle(valid_data, frame_df),
                                      batch_size=args.batch_size, shuffle=True, drop_last=False,num_workers=4)
    elif dataset == 'MSC':
        train_data = pd.read_csv(
            '/workspace/data/MSC-Rad4R/URBAN_D0/pmap_dataset/RCS_dataset_train.csv').values[:,1:]
        valid_data = pd.read_csv(
            '/workspace/data/MSC-Rad4R/URBAN_D0/pmap_dataset/RCS_dataset_test.csv').values[:,1:]
        train_dataloader = DataLoader(dataset=RCS_Dataset_MSC(train_data,rc),
                                      batch_size=args.batch_size, shuffle=True, drop_last=False,num_workers=4)
        valid_dataloader = DataLoader(dataset=RCS_Dataset_MSC(valid_data,rc),
                                      batch_size=args.batch_size, shuffle=True, drop_last=False,num_workers=4)
    model = None
    device = args.device
    if weights is not None:
        model = torch.load(weights).to(device)
        print('continue training')
    else:
        if dataset == 'VoD':
            model = RCSNet(batch_size,5).to(device)
        elif dataset == 'astyx':
            model = RCSNet(batch_size, 4).to(device)
        elif dataset == 'snail_radar' or dataset == 'snail_radar_eagle':
            model = RCSNet(batch_size,5).to(device)
        elif dataset == 'MSC':
            model = RCSNet(batch_size,5).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    min_val_loss = 1000000

    for e in range(epochs):
        logging.info(f"Epoch {e + 1}/{epochs} started.")
        l_td = len(train_dataloader)
        checkpoint = l_td * 0.10
        next_checkpoint = checkpoint
        last_time = time.time()  # set timer
        print('epoch ',e)
        total_loss = 0
        cnt = 0
        model.train()
        for idx, (batch_dict,rcs_values) in enumerate(train_dataloader):
            #{'local_pic': local_pic, 'edge_image': edge_image, 'range_image': range_image, 'frame': frame, 'radarpoint':radarpoint}
            #current_batch_size = batch_dict['local_pic'].size(0)
            current_batch_size = batch_dict['radarpoint'].size(0)
            label = rcs_values
            label = torch.unsqueeze(label.to(device).float(), 1)
            pred = model(batch_dict,current_batch_size)
            #loss = criterion(pred,label)
            pred, label = pred.reshape(-1, 1), label.reshape(-1, 1)
            pred = (pred - min_value)/(max_value - min_value)
            label = (label - min_value) / (max_value - min_value)
            # error = abs(pred - label) / label
            loss = criterion(pred,label)
            #loss = criterion(error, torch.zeros_like(error))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            cnt += 1
            #process bar
            if idx + 1 >= next_checkpoint:
                current_time = time.time()
                elapsed_time = current_time - last_time
                last_time = current_time  # 更新最后记录时间
                while next_checkpoint <= idx + 1:
                    next_checkpoint += checkpoint
                percent = (idx + 1) / l_td * 100
                print(f"Processed {percent:.1f}% of batches. Time elapsed: {elapsed_time:.2f} seconds.")

        logging.info(f'avg training loss of epoch {e}: {total_loss/cnt}')
        #logging.info(f'avg KL Divergence loss (resized image) of epoch {e}: {total_loss1_small/cnt}')


        #print(f'avg training loss of epoch {e}: ',total_loss/cnt)

        #----------------------------------validation---------------------------------------------
        model.eval()

        total_val_loss = 0
        total_val_loss1 = 0
        cnt = 0
        predictions = []
        labels = []
        frames = []
        indices = []

        with torch.no_grad():
            for idx, (batch_dict,rcs_values) in enumerate(valid_dataloader):
                # current_batch_size = batch_dict['local_pic'].size(0)
                current_batch_size = batch_dict['radarpoint'].size(0)
                label = rcs_values
                label = torch.unsqueeze(label.to(device).float(), 1)
                pred = model(batch_dict,current_batch_size)
                loss1 = criterion(pred, label)


                predictions.append(pred.cpu().numpy())
                labels.append(label.cpu().numpy())
                frames.extend(batch_dict['frame'].cpu().numpy())
                indices.extend(batch_dict['index'].cpu().numpy())

                pred, label = pred.reshape(-1, 1), label.reshape(-1, 1)
                pred = (pred - min_value) / (max_value - min_value)
                label = (label - min_value) / (max_value - min_value)
                # error = abs(pred - label) / label
                loss = criterion(pred, label)

                #loss = criterion(pred,label)
                total_val_loss += loss.item()
                total_val_loss1 += loss1.item()
                cnt += 1
            total_val_loss /= cnt
            total_val_loss1 /=cnt
            if total_val_loss < min_val_loss:
                min_val_loss = total_val_loss
                if not os.path.exists(base_dir + 'ablation/weights/'):
                    os.mkdir(base_dir + 'ablation/weights/')
                else:
                    modelsave = base_dir + 'ablation/weights/' + f'RCS_{dataset}_model_best_{ablation}.pth'
                    torch.save(model,modelsave)
                    # print("min loss = ", min_val_loss)
                    # print('total_loss_1:',total_val_loss1)
                    # print("Model Saved!")
                    logging.info(f"min loss = {min_val_loss}" )
                    logging.info(f'total_loss_1:{total_val_loss1}')
                    logging.info("Model Saved!")

                predictions = np.concatenate(predictions)
                labels = np.concatenate(labels)
                df_results = pd.DataFrame({
                    'Frame': frames,
                    'Index': indices,
                    'True Labels': labels.flatten(),
                    'Predictions': predictions.flatten()
                })
                df_results.to_csv(base_dir + f'ablation/RCS_validation_results_ablation_{dataset}_{ablation}.csv', index=False)
                #print("Validation results saved.")
                logging.info("Validation results saved.")






if __name__ == '__main__':
    # df = pd.read_csv('/workspace/data/MSC-Rad4R/URBAN_D0/pmap_dataset/pmap_dataset.csv')
    # average_pointnum = df['PointNum'].mean()
    # print(f"The average value of 'PointNum' is: {average_pointnum}")


    seed_everything()
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default="cuda:3",
                        help='use gpu')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (per gpu).')
    args = parser.parse_args()
    epochs = 40
    weights = None
    dataset = 'VoD'
    rc = 100
    ablation = f'para_{rc}_1_only_cam'
    print(f'running on dataset {dataset}, ablation study {ablation}')
    if dataset == 'VoD':
        weights = f'/workspace/data/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/ablation/weights/RCS_{dataset}_model_best_{ablation}.pth'
    elif dataset == 'astyx':
        weights = f'D:/Astyx dataset/dataset_astyx_hires2019/ablation/weights/RCS_{dataset}_model_best_{ablation}.pth'
    elif dataset == 'snail_radar':
        weights = f'D:/snail_radar/20231208/data4/ablation/weights/RCS_{dataset}_model_best_{ablation}.pth'
    elif dataset == 'snail_radar_eagle':
        weights = f'D:/snail_radar/20231208/data4/eagle/ablation/weights/RCS_{dataset}_model_best_{ablation}.pth'
    elif dataset == 'MSC':
        weights = f'/workspace/data/MSC-Rad4R/URBAN_D0/pmap_dataset/ablation/weights/RCS_{dataset}_model_best_{ablation}.pth'

    if os.path.exists(weights):
        train_rcs(args, epochs, dataset, ablation, rc, weights)
    else:
        train_rcs(args, epochs, dataset, ablation, rc)