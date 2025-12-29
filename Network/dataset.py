import os.path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


def crop_image(image, u, v, crop_size=100, gray = False):

    height, width, channels = 0,0,0
    if gray:
        height, width = image.shape
    else:
        height, width, channels = image.shape
    # 计算裁剪区域的左上角坐标
    x_start = u - crop_size // 2
    y_start = v - crop_size // 2

    # 检查裁剪区域是否超出图片范围
    if x_start < 0:
        x_start = 0
    elif x_start + crop_size > width:
        x_start = width - crop_size

    if y_start < 0:
        y_start = 0
    elif y_start + crop_size > height:
        y_start = height - crop_size

    # 计算裁剪区域的右下角坐标
    x_end = x_start + crop_size
    y_end = y_start + crop_size

    # 裁剪图片
    cropped_image = image[y_start:y_end, x_start:x_end]

    return cropped_image


class Dis_Dataset_VoD(Dataset):
    def __init__(self, df, base_dir = '/workspace/data/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/'):
        self.df = df      #prob_dataset
        self.base_dir = base_dir
        self.lidar_depth = base_dir + 'lidar_inter2d_matrix/'
        self.lidar_3dpoints_dir = '/workspace/data/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/l_down_512/'
        self.local_pics_dir = '/workspace/data/VoD_dataset/view_of_delft_PUBLIC/radar/training/image_2/'
        self.prob_map_dir = base_dir + 'pmap_image_adaptive_covariance/'
        self.edge_image = base_dir + 'edge_image/'
        #self.target_dir = base_dir + 'radar_2d_map_pro/'


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        frame, pointnum, ego_velocity = self.df[idx,:]
        frame = int(frame)
        
        pointnum = torch.tensor(pointnum).float()

        # lidar_3d = np.load(self.lidar_3dpoints_dir+f'{frame}.npy')
        # lidar_3d = torch.FloatTensor(lidar_3d)

        # local_lidar_depth = np.load(self.lidar_depth + f'{frame}.npy')
        # local_lidar_depth = torch.tensor(local_lidar_depth)
        # try:
        #     local_lidar_depth = np.load(self.lidar_depth + f'{frame}.npy')
        #     local_lidar_depth = local_lidar_depth[:644, :1935]  # 裁剪数据
        # except ValueError:
        #     print(f"Error in reshaping file {frame}.npy")


        local_pic = np.array(cv2.imread(self.local_pics_dir + str(frame).zfill(5) + '.jpg')[571:1215,0:1935])
        local_pic = torch.tensor(local_pic)

        target_image = cv2.imread(self.prob_map_dir+f'{frame}.jpg', cv2.IMREAD_GRAYSCALE)
        resized_target = cv2.resize(target_image, (53, 20), interpolation=cv2.INTER_LINEAR)
        target_image = torch.tensor(target_image)
        resized_target = torch.tensor(resized_target)

        ego_velocity = torch.tensor(ego_velocity)


        # sample = {'local_pic': local_pic, 'edge_image': edge_image, 'local_lidar_depth': local_lidar_depth, 'frame': frame, 'ego_velo':ego_velocity}
        #sample = {'local_pic': local_pic, 'frame': frame, 'local_lidar_depth': local_lidar_depth, 'ego_velo': ego_velocity}
        sample = {'local_pic': local_pic, 'frame': frame, 'ego_velo': ego_velocity}
        #sample = {'local_pic': local_pic, 'frame': frame, 'ego_velo': ego_velocity, 'l_3d':lidar_3d, 'local_lidar_depth': local_lidar_depth}
        return sample, target_image, pointnum


class Dis_Dataset_astyx(Dataset):
    def __init__(self, df, base_dir = 'D:/Astyx dataset/dataset_astyx_hires2019/'):
        self.df = df      #prob_dataset
        self.base_dir = base_dir
        #self.lidar_depth = base_dir + 'lidar_inter2d_matrix/'
        self.local_pics_dir = base_dir+'camera_front/'

        self.prob_map_dir = base_dir + 'pmap_image/'
        self.edge_image = base_dir + 'edge_image/'
        #self.target_dir = base_dir + 'radar_2d_map_pro/'


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        frame, pointnum, ego_velocity = self.df[idx,:]
        frame = int(frame)
        pointnum = torch.tensor(pointnum).float()

        # local_lidar_depth = np.load(self.lidar_depth + f'{frame}.npy')
        # local_lidar_depth = torch.tensor(local_lidar_depth)

        local_pic = np.array(cv2.imread(self.local_pics_dir + str(frame).zfill(6) + '.jpg'))
        #local_pic = np.array(cv2.imread(self.aug_image_dir + str(int(frame)) + '.jpg'))
        local_pic = torch.tensor(local_pic)

        edge_image = cv2.imread(self.edge_image + f'{frame}.jpg', cv2.IMREAD_GRAYSCALE)
        edge_image = torch.tensor(edge_image)

        target_image = cv2.imread(self.prob_map_dir+f'{frame}.jpg', cv2.IMREAD_GRAYSCALE)
        target_image = torch.tensor(target_image)

        ego_velocity = torch.tensor(ego_velocity)


        sample = {'local_pic': local_pic, 'edge_image': edge_image,  'frame': frame, 'ego_velo':ego_velocity}
        return sample, target_image, pointnum


class Dis_Dataset_Snail(Dataset):
    def __init__(self, df, frame_df, base_dir = '/workspace/data/SR/data4/'):
        self.df = df      #prob_dataset
        self.frame_df =frame_df
        self.base_dir = base_dir
        #self.lidar_depth = base_dir + 'lidar_inter2d_matrix/'
        self.local_pics_dir = base_dir+'zed2i/left/'

        self.prob_map_dir = base_dir + 'pmap_image/'
        self.edge_image = base_dir + 'edge_image/'
        #self.target_dir = base_dir + 'radar_2d_map_pro/'


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        frame, pointnum, ego_velocity = self.df[idx,:]
        frame = int(frame)
        #print(self.frame_df[frame],self.frame_df[frame].shape)
        timestamp, cam_left_time, cam_right_time, radar_arg_time, radar_eagle_time, lidar_time, odom_time = self.frame_df[frame]

        #print(cam_left_time)
        pointnum = torch.tensor(pointnum).float().cuda()

        # local_lidar_depth = np.load(self.lidar_depth + f'{frame}.npy')
        # local_lidar_depth = torch.tensor(local_lidar_depth)

        local_pic = np.array(cv2.imread(self.local_pics_dir + str(cam_left_time) + '.jpg'))
        #local_pic = np.array(cv2.imread(self.aug_image_dir + str(int(frame)) + '.jpg'))
        local_pic = torch.tensor(local_pic)

        # edge_image = cv2.imread(self.edge_image + f'{frame}.jpg', cv2.IMREAD_GRAYSCALE)
        # edge_image = torch.tensor(edge_image)

        target_image = cv2.imread(self.prob_map_dir+f'{frame}.jpg', cv2.IMREAD_GRAYSCALE)
        target_image = torch.tensor(target_image)

        ego_velocity = torch.tensor(ego_velocity)


        #sample = {'local_pic': local_pic, 'edge_image': edge_image,  'frame': frame, 'ego_velo':ego_velocity}
        sample = {'local_pic': local_pic, 'frame': frame, 'ego_velo': ego_velocity}
        return sample, target_image, pointnum


class Dis_Dataset_Snail_eagle(Dataset):
    def __init__(self, df, frame_df, base_dir = '/workspace/data/SR/data4/'):
        self.df = df      #prob_dataset
        self.frame_df =frame_df
        self.base_dir = base_dir
        #self.lidar_depth = base_dir + 'lidar_inter2d_matrix/'
        self.local_pics_dir = base_dir+'zed2i/left/'

        self.prob_map_dir = base_dir + 'eagle/pmap_image/'
        self.edge_image = base_dir + 'edge_image/'
        #self.target_dir = base_dir + 'radar_2d_map_pro/'


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        frame, pointnum, ego_velocity = self.df[idx,:]
        frame = int(frame)
        #print(self.frame_df[frame],self.frame_df[frame].shape)
        timestamp, cam_left_time, cam_right_time, radar_arg_time, radar_eagle_time, lidar_time, odom_time = self.frame_df[frame]

        #print(cam_left_time)
        pointnum = torch.tensor(pointnum).float().cuda()

        # local_lidar_depth = np.load(self.lidar_depth + f'{frame}.npy')
        # local_lidar_depth = torch.tensor(local_lidar_depth)

        local_pic = np.array(cv2.imread(self.local_pics_dir + str(cam_left_time) + '.jpg'))
        #local_pic = np.array(cv2.imread(self.aug_image_dir + str(int(frame)) + '.jpg'))
        local_pic = torch.tensor(local_pic)

        edge_image = cv2.imread(self.edge_image + f'{frame}.jpg', cv2.IMREAD_GRAYSCALE)
        edge_image = torch.tensor(edge_image)

        target_image = cv2.imread(self.prob_map_dir+f'{frame}.jpg', cv2.IMREAD_GRAYSCALE)
        target_image = torch.tensor(target_image)

        ego_velocity = torch.tensor(ego_velocity)


        sample = {'local_pic': local_pic, 'edge_image': edge_image,  'frame': frame, 'ego_velo':ego_velocity}
        return sample, target_image, pointnum



class Dis_Dataset_MSC(Dataset):
    def __init__(self, df, base_dir = '/workspace/data/MSC-Rad4R/URBAN_D0/pmap_dataset/'):
        self.df = df      #prob_dataset
        self.base_dir = base_dir
        self.local_pics_dir = '/workspace/data/MSC-Rad4R/URBAN_D0/1_IMAGE/LEFT/'
        self.prob_map_dir = base_dir + 'pmap_image_adaptive_covariance/'
        #self.edge_image = base_dir + 'edge_image/'


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        frame, pointnum, ego_velocity = self.df[idx,:]
        frame = int(frame)
        
        pointnum = torch.tensor(pointnum).float()
        local_pic = np.array(cv2.imread(self.local_pics_dir + str(frame).zfill(6) + '.png')[0:540,77:720])
        local_pic = torch.tensor(local_pic)

        target_image = cv2.imread(self.prob_map_dir+f'{frame}.jpg', cv2.IMREAD_GRAYSCALE)
        # resized_target = cv2.resize(target_image, (53, 20), interpolation=cv2.INTER_LINEAR)
        target_image = torch.tensor(target_image)
        # resized_target = torch.tensor(resized_target)

        ego_velocity = torch.tensor(ego_velocity)


        sample = {'local_pic': local_pic, 'frame': frame, 'ego_velo': ego_velocity}
        return sample, target_image, pointnum


class Dis_Dataset_hercules(Dataset):
    def __init__(self, df, frame_df, base_dir = '/workspace/data/HeRCULES/Library_01_Day/pmap_dataset/', target_len=1500):
        self.df = df      #prob_dataset
        self.base_dir = base_dir
        self.frame_df = frame_df
        self.lidar_depth = base_dir + 'lidar_inter2d_matrix/'
        # self.lidar_depth = base_dir + 'lidar_matrix/'
        # self.lidar_3dpoints_dir = '/workspace/data/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/l_down_512/'

        self.local_pics_dir = '/workspace/data/HeRCULES/Library_01_Day/stereo_left/'
        self.prob_map_dir = base_dir + 'pmap_image_adaptive_covariance_5frames/'
        self.edge_image = base_dir + 'edge_image/'
        self.r_in_dir = base_dir + 'r_in/'
        self.depth_image_dir =  base_dir + 'radar_depth_5frames_pic/'
        self.depth_image_5frames_dir =  base_dir + 'radar_depth_5frames_pic/'
        self.target_len = target_len
        self.lidar_mask_dir = base_dir + 'flat_mask_npz/'
        #self.target_dir = base_dir + 'radar_2d_map_pro/'
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        frame, pointnum, ego_velocity = self.df[idx,:]
        frame = int(frame)
        frame1,timestamp,cam_left_time,cam_right_time,radar_continental_time,lidar_time,odom_time = self.frame_df[frame]
        radar_pcd = np.load(f'/workspace/data/HeRCULES/Library_01_Day/pmap_dataset/r_in_5frames/{frame}.npy')
        pointnum = torch.tensor(len(radar_pcd)).float()



        # local_lidar_depth = np.load(self.lidar_depth + f'{frame}.npz')['depth']
        # local_lidar_depth = torch.tensor(local_lidar_depth, dtype=torch.float32)

        # flat_mask = np.load(self.lidar_mask_dir + f'{frame}.npz')['flat_mask']
        # flat_mask = torch.tensor(flat_mask, dtype=torch.float32)

        # target_depth_image = cv2.imread(self.depth_image_dir+f'{frame}.png', cv2.IMREAD_GRAYSCALE)
        # target_depth_image = torch.tensor(target_depth_image)
        # mask = (target_depth_image > 0).to(torch.uint8)

        # target_depth_image_5frames = cv2.imread(self.depth_image_5frames_dir+f'{frame}.png', cv2.IMREAD_GRAYSCALE)
        # target_depth_image_5frames = torch.tensor(target_depth_image_5frames)
        # target_depth_image_5frames[mask.bool()] = 0

        local_pic = np.array(cv2.imread(self.local_pics_dir + str(cam_left_time) + '.png')[440:700,0:1440])
        local_pic = torch.tensor(local_pic)

        target_image = cv2.imread(self.prob_map_dir+f'{frame}.jpg', cv2.IMREAD_GRAYSCALE)
        target_image = torch.tensor(target_image)

        ego_velocity = torch.tensor(ego_velocity)

        sample = {'local_pic': local_pic, 'frame': frame, 'ego_velo': ego_velocity}
        return sample, target_image, pointnum







class RCS_Dataset_VoD(Dataset):
    def __init__(self, df, rc, base_dir = '/workspace/data/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/'):
        self.df = df      #prob_dataset
        self.base_dir = base_dir
        self.range_image = base_dir + 'ablation/range_image_1/'
        self.local_pics_dir = '/workspace/data/VoD_dataset/view_of_delft_PUBLIC/radar/training/image_2/'
        self.edge_image = base_dir + 'edge_image/'
        self.rc = rc
        #self.target_dir = base_dir + 'radar_2d_map_pro/'


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        frame, index, x,y,z,u,v,v_r,rcs = self.df[idx,:]
        frame, index, u, v = int(frame), int(index), int(u), int(v)
        rc = self.rc
        ego_velocity = np.linalg.norm(np.load('/workspace/data/VoD_dataset/view_of_delft_PUBLIC/radar/training/radar_velo.npy')[frame])

        range_image = np.array(cv2.imread(self.range_image + f'{frame}_{index}.jpg',cv2.IMREAD_GRAYSCALE))
        range_image = torch.tensor(range_image)

        local_pic_dir = self.base_dir + f'RCS_patch_{rc}/{frame}_{index}.jpg'
        #local_pic_dir = self.base_dir + f'RCS_patch/VoD/{frame}_{index}.jpg'
        local_pic = None
        if os.path.exists(local_pic_dir):
            local_pic = cv2.imread(local_pic_dir)
        else:
            local_pic_tmp = crop_image(cv2.imread(self.local_pics_dir + str(frame).zfill(5) + '.jpg'),u,v,rc)
            cv2.imwrite(local_pic_dir, local_pic_tmp)
            local_pic = np.array(local_pic_tmp)
        local_pic = torch.tensor(local_pic)

        # edge_pic_dir = self.base_dir + f'edge_patch_{rc}/{frame}_{index}.jpg'
        # #edge_pic_dir = self.base_dir + f'edge_patch/{frame}_{index}.jpg'
        # edge_image = None
        # if os.path.exists(edge_pic_dir):
        #     edge_image = cv2.imread(edge_pic_dir, cv2.IMREAD_GRAYSCALE)
        # else:
        #     edge_image_tmp = cv2.imread(self.edge_image + f'{frame}.jpg', cv2.IMREAD_GRAYSCALE)
        #     edge_image = crop_image(edge_image_tmp, u, v, rc, True)
        #     cv2.imwrite(edge_pic_dir, edge_image)
        # edge_image = np.array(edge_image)
        # edge_image = torch.tensor(edge_image)

        rcs = torch.tensor(rcs)

        radarpoint = torch.tensor([x, y, z, ego_velocity, v_r])

        #sample = {'local_pic': local_pic, 'edge_image': edge_image, 'range_image': range_image, 'frame': frame, 'index':index, 'radarpoint':radarpoint}
        # sample = {'frame': frame, 'index':index, 'radarpoint':radarpoint}
        sample = {'local_pic': local_pic, 'frame': frame, 'index':index, 'radarpoint':radarpoint}
        return sample, rcs


class RCS_Dataset_VoD_test(Dataset):
    def __init__(self, df, rc, base_dir = '/workspace/data/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/'):
        self.df = df      #prob_dataset
        self.base_dir = base_dir
        self.rc = rc
        self.range_image = base_dir + 'range_image_rcs/'
        self.local_pics_dir = '/workspace/data/VoD_dataset/view_of_delft_PUBLIC/radar/training/image_2/'
        #self.edge_image = base_dir + 'edge_image/'
        #self.target_dir = base_dir + 'radar_2d_map_pro/'


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        frame, index, x,y,z,u,v,v_r,rcs = self.df[idx,:]
        frame, index, u, v = int(frame), int(index), int(u), int(v)
        rc = self.rc
        ego_velocity = np.linalg.norm(np.load('/workspace/data/VoD_dataset/view_of_delft_PUBLIC/radar/training/radar_velo.npy')[frame])

        range_image = np.array(cv2.imread(self.range_image + f'{frame}_{index}.jpg',cv2.IMREAD_GRAYSCALE))
        range_image = torch.tensor(range_image)

        local_pic_dir = self.base_dir + f'RCS_gen_patch_{rc}/{frame}_{index}.jpg'
        #local_pic_dir = self.base_dir + f'RCS_patch/VoD/{frame}_{index}.jpg'
        local_pic = None
        if os.path.exists(local_pic_dir):
            local_pic = cv2.imread(local_pic_dir)
        else:
            os.makedirs(self.base_dir + f'RCS_gen_patch_{rc}',exist_ok=True)
            local_pic_tmp = crop_image(cv2.imread(self.local_pics_dir + str(frame).zfill(5) + '.jpg'),u,v,rc)
            cv2.imwrite(local_pic_dir, local_pic_tmp)
            local_pic = np.array(local_pic_tmp)
        #local_pic = np.array(cv2.imread(self.aug_image_dir + str(int(frame)) + '.jpg'))
        local_pic = torch.tensor(local_pic)

        # edge_pic_dir = self.base_dir + f'edge_patch_{rc}/{frame}_{index}.jpg'
        # #edge_pic_dir = self.base_dir + f'edge_patch/{frame}_{index}.jpg'
        # edge_image = None
        # if os.path.exists(edge_pic_dir):
        #     edge_image = cv2.imread(edge_pic_dir, cv2.IMREAD_GRAYSCALE)
        # else:
        #     edge_image_tmp = cv2.imread(self.edge_image + f'{frame}.jpg', cv2.IMREAD_GRAYSCALE)
        #     edge_image = crop_image(edge_image_tmp, u, v, rc, True)
        #     cv2.imwrite(edge_pic_dir, edge_image)
        # edge_image = np.array(edge_image)
        # edge_image = torch.tensor(edge_image)

        rcs = torch.tensor(rcs)

        radarpoint = torch.tensor([x, y, z, ego_velocity, v_r])

        #sample = {'local_pic': local_pic, 'edge_image': edge_image, 'range_image': range_image, 'frame': frame, 'index':index, 'radarpoint':radarpoint}
        sample = {'local_pic': local_pic, 'range_image': range_image, 'frame': frame, 'index':index, 'radarpoint':radarpoint}
        return sample, rcs


class RCS_Dataset_astyx(Dataset):
    def __init__(self, df, base_dir = 'D:/Astyx dataset/dataset_astyx_hires2019/'):
        self.df = df      #prob_dataset
        self.base_dir = base_dir
        self.range_image = base_dir + 'ablation/newest/range_image_1/'
        self.local_pics_dir = base_dir + 'camera_front/'
        self.edge_image = base_dir + 'edge_image/'
        #self.target_dir = base_dir + 'radar_2d_map_pro/'


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        frame, index, x,y,z,u,v,v_r,rcs = self.df[idx,:]
        frame, index, u, v = int(frame), int(index), int(u), int(v)
        #ego_velocity = np.linalg.norm(np.load('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/radar_velo.npy')[frame])
        rc = 300
        range_image = np.array(cv2.imread(self.range_image + f'{frame}_{index}.jpg',cv2.IMREAD_GRAYSCALE))
        range_image = torch.tensor(range_image)

        local_pic_dir = self.base_dir + f'RCS_patch_{rc}/{frame}_{index}.jpg'

        #local_pic_dir = self.base_dir + f'RCS_patch/{frame}_{index}.jpg'
        local_pic = None
        if os.path.exists(local_pic_dir):
            #local_pic = cv2.imread(local_pic_dir)
            # if grayscale needed:
            gray_image = cv2.imread(local_pic_dir, cv2.IMREAD_GRAYSCALE)
            local_pic = cv2.merge([gray_image, gray_image, gray_image])
        else:
            os.makedirs(self.base_dir + f'RCS_patch_{rc}', exist_ok=True)
            local_pic_tmp = crop_image(cv2.imread(self.local_pics_dir + str(frame).zfill(6) + '.jpg'), u, v, rc)
            cv2.imwrite(local_pic_dir, local_pic_tmp)
            local_pic = np.array(local_pic_tmp)
        local_pic = torch.tensor(local_pic)


        edge_pic_dir = self.base_dir + f'edge_patch_{rc}/{frame}_{index}.jpg'
        #edge_pic_dir = self.base_dir + f'edge_patch/{frame}_{index}.jpg'
        edge_image = None
        if os.path.exists(edge_pic_dir):
            edge_image = cv2.imread(edge_pic_dir, cv2.IMREAD_GRAYSCALE)
        else:
            os.makedirs(self.base_dir + f'edge_patch_{rc}',exist_ok=True)
            edge_image_tmp = cv2.imread(self.edge_image + f'{frame}.jpg', cv2.IMREAD_GRAYSCALE)
            edge_image = crop_image(edge_image_tmp, u, v, rc, True)
            cv2.imwrite(edge_pic_dir, edge_image)
        edge_image = np.array(edge_image)
        edge_image = torch.tensor(edge_image)

        rcs = torch.tensor(rcs)

        radarpoint = torch.tensor([x, y, z, v_r])

        sample = {'local_pic': local_pic, 'edge_image': edge_image, 'range_image': range_image, 'frame': frame, 'index':index, 'radarpoint':radarpoint}
        return sample, rcs



class RCS_Dataset_Snail(Dataset):
    def __init__(self, df, frame_df, base_dir = 'D:/snail_radar/20231208/data4/'):
        self.df = df      #prob_dataset
        self.frame_df = frame_df
        self.base_dir = base_dir
        self.range_image = base_dir + 'ablation/newest/range_image_1/'
        self.local_pics_dir = base_dir + 'zed2i/left/'
        self.edge_image = base_dir + 'edge_image/'
        #self.target_dir = base_dir + 'radar_2d_map_pro/'


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rc = 100
        frame, index, x,y,z,u,v,v_r,rcs = self.df[idx,:]
        frame, index, u, v = int(frame), int(index), int(u), int(v)
        #ego_velocity = np.linalg.norm(np.load('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/radar_velo.npy')[frame])
        frame = int(frame)
        # print(self.frame_df[frame],self.frame_df[frame].shape)
        timestamp, cam_left_time, cam_right_time, radar_arg_time, radar_eagle_time, lidar_time, odom_time = \
        self.frame_df[frame]
        range_image = np.array(cv2.imread(self.range_image + f'{frame}_{index}.jpg',cv2.IMREAD_GRAYSCALE))
        range_image = torch.tensor(range_image)



        #local_pic_dir = self.base_dir + f'RCS_patch_{rc}/{frame}_{index}.jpg'
        #os.makedirs(self.base_dir + f'RCS_patch_{rc}', exist_ok=True)
        local_pic_dir = self.base_dir + f'RCS_patch/{frame}_{index}.jpg'
        local_pic = None
        if os.path.exists(local_pic_dir):
            # if grayscale needed:
            # gray_image = cv2.imread(local_pic_dir, cv2.IMREAD_GRAYSCALE)
            # gray_3_channel = cv2.merge([gray_image, gray_image, gray_image])
            local_pic = cv2.imread(local_pic_dir)
        else:
            print('warning')
            local_pic_tmp = crop_image(cv2.imread(self.local_pics_dir + str(cam_left_time) + '.jpg'), u, v, rc)
            cv2.imwrite(local_pic_dir, local_pic_tmp)
            local_pic = np.array(local_pic_tmp)

        local_pic = torch.tensor(local_pic)


        #edge_pic_dir = self.base_dir + f'edge_patch_{rc}/{frame}_{index}.jpg'
        #os.makedirs(self.base_dir + f'edge_patch_{rc}', exist_ok=True)
        edge_pic_dir = self.base_dir + f'edge_patch/{frame}_{index}.jpg'
        edge_image = None
        if os.path.exists(edge_pic_dir):
            edge_image = cv2.imread(edge_pic_dir, cv2.IMREAD_GRAYSCALE)
        else:
            edge_image_tmp = cv2.imread(self.edge_image + f'{frame}.jpg', cv2.IMREAD_GRAYSCALE)
            edge_image = crop_image(edge_image_tmp, u, v, rc, True)
            cv2.imwrite(edge_pic_dir, edge_image)
        edge_image = np.array(edge_image)
        edge_image = torch.tensor(edge_image)

        rcs = torch.tensor(rcs)

        radarpoint = torch.tensor([x, y, z, v, v_r])

        sample = {'local_pic': local_pic, 'edge_image': edge_image, 'range_image': range_image, 'frame': frame, 'index':index, 'radarpoint':radarpoint}
        return sample, rcs


class RCS_Dataset_Snail_eagle(Dataset):
    def __init__(self, df, frame_df, base_dir = 'D:/snail_radar/20231208/data4/'):
        self.df = df      #prob_dataset
        self.frame_df = frame_df
        self.base_dir = base_dir
        self.range_image = base_dir + 'eagle/ablation/newest/range_image_1/'
        self.local_pics_dir = base_dir + 'zed2i/left/'
        self.edge_image = base_dir + 'edge_image/'
        #self.target_dir = base_dir + 'radar_2d_map_pro/'


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rc = 100
        frame, index, x,y,z,u,v,v_r,rcs = self.df[idx,:]
        frame, index, u, v = int(frame), int(index), int(u), int(v)
        #ego_velocity = np.linalg.norm(np.load('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/radar_velo.npy')[frame])
        frame = int(frame)
        # print(self.frame_df[frame],self.frame_df[frame].shape)
        timestamp, cam_left_time, cam_right_time, radar_arg_time, radar_eagle_time, lidar_time, odom_time = \
        self.frame_df[frame]
        range_image = np.array(cv2.imread(self.range_image + f'{frame}_{index}.jpg',cv2.IMREAD_GRAYSCALE))
        range_image = torch.tensor(range_image)



        local_pic_dir = self.base_dir + f'eagle/RCS_patch_{rc}/{frame}_{index}.jpg'
        os.makedirs(self.base_dir + f'eagle/RCS_patch_{rc}', exist_ok=True)
        #local_pic_dir = self.base_dir + f'RCS_patch/{frame}_{index}.jpg'
        local_pic = None
        if os.path.exists(local_pic_dir):
            local_pic = cv2.imread(local_pic_dir)
        else:
            print('warning')
            local_pic_tmp = crop_image(cv2.imread(self.local_pics_dir + str(cam_left_time) + '.jpg'), u, v, rc)
            cv2.imwrite(local_pic_dir, local_pic_tmp)
            local_pic = np.array(local_pic_tmp)

        local_pic = torch.tensor(local_pic)


        edge_pic_dir = self.base_dir + f'eagle/edge_patch_{rc}/{frame}_{index}.jpg'
        os.makedirs(self.base_dir + f'eagle/edge_patch_{rc}', exist_ok=True)
        #edge_pic_dir = self.base_dir + f'eagle/edge_patch/{frame}_{index}.jpg'
        edge_image = None
        if os.path.exists(edge_pic_dir):
            edge_image = cv2.imread(edge_pic_dir, cv2.IMREAD_GRAYSCALE)
        else:
            edge_image_tmp = cv2.imread(self.edge_image + f'{frame}.jpg', cv2.IMREAD_GRAYSCALE)
            edge_image = crop_image(edge_image_tmp, u, v, rc, True)
            cv2.imwrite(edge_pic_dir, edge_image)
        edge_image = np.array(edge_image)
        edge_image = torch.tensor(edge_image)

        rcs = torch.tensor(rcs)

        radarpoint = torch.tensor([x, y, z, v, v_r])

        sample = {'local_pic': local_pic, 'edge_image': edge_image, 'range_image': range_image, 'frame': frame, 'index':index, 'radarpoint':radarpoint}
        return sample, rcs
    

class RCS_Dataset_MSC(Dataset):
    def __init__(self, df, rc, base_dir = '/workspace/data/MSC-Rad4R/URBAN_D0/pmap_dataset/'):
        self.df = df     
        self.base_dir = base_dir
        # self.range_image = base_dir + 'ablation/range_image_1/'
        self.range_image = base_dir + 'range_image/'
        self.local_pics_dir = '/workspace/data/MSC-Rad4R/URBAN_D0/1_IMAGE/LEFT/'
        # self.edge_image = base_dir + 'edge_image/'
        self.rc = rc
        #self.target_dir = base_dir + 'radar_2d_map_pro/'


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        frame, index, x,y,z,u,v,v_r,rcs = self.df[idx,:]
        frame, index, u, v = int(frame), int(index), int(u), int(v)
        # rc = 100
        df = pd.read_csv('/workspace/data/MSC-Rad4R/URBAN_D0/pmap_dataset/pmap_dataset.csv')
        ego_velocity = df.loc[df['Frame'] == frame, 'Velocity'].iloc[0]
        #ego_velocity = np.linalg.norm(np.load('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/radar_velo.npy')[frame])

        # range_image = np.array(cv2.imread(self.range_image + f'{frame}_{index}.jpg',cv2.IMREAD_GRAYSCALE))
        # range_image = torch.tensor(range_image)

        local_pic_dir = self.base_dir + f'RCS_patch_{self.rc}/{frame}_{index}.jpg'
        #local_pic_dir = self.base_dir + f'RCS_patch/VoD/{frame}_{index}.jpg'
        local_pic = None
        if os.path.exists(local_pic_dir):
            local_pic = cv2.imread(local_pic_dir)
        else:
            os.makedirs(self.base_dir + f'RCS_patch_{self.rc}/{frame}_{index}', exist_ok=True)
            local_pic_tmp = crop_image(cv2.imread(self.local_pics_dir + str(frame).zfill(6) + '.png'),u,v,self.rc)
            cv2.imwrite(local_pic_dir, local_pic_tmp)
            local_pic = np.array(local_pic_tmp)
        #local_pic = np.array(cv2.imread(self.aug_image_dir + str(int(frame)) + '.jpg'))
        local_pic = torch.tensor(local_pic)

        # edge_pic_dir = self.base_dir + f'edge_patch_{self.rc}/{frame}_{index}.jpg'
        # #edge_pic_dir = self.base_dir + f'edge_patch/{frame}_{index}.jpg'
        # edge_image = None
        # if os.path.exists(edge_pic_dir):
        #     edge_image = cv2.imread(edge_pic_dir, cv2.IMREAD_GRAYSCALE)
        # else:
        #     os.makedirs(self.base_dir + f'edge_patch_{self.rc}', exist_ok=True)
        #     edge_image_tmp = cv2.imread(self.edge_image + f'{frame}.jpg', cv2.IMREAD_GRAYSCALE)
        #     edge_image = crop_image(edge_image_tmp, u, v, self.rc, True)
        #     cv2.imwrite(edge_pic_dir, edge_image)
        # edge_image = np.array(edge_image)
        # edge_image = torch.tensor(edge_image)

        rcs = torch.tensor(rcs)

        radarpoint = torch.tensor([x, y, z, ego_velocity, v_r])

        #sample = {'local_pic': local_pic, 'edge_image': edge_image, 'range_image': range_image, 'frame': frame, 'index':index, 'radarpoint':radarpoint}
        sample = {'local_pic': local_pic,'frame': frame, 'index':index, 'radarpoint':radarpoint}
        return sample, rcs