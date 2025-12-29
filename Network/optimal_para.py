import pandas as pd
import numpy as np
import os
import cv2
import time
from concurrent.futures import ThreadPoolExecutor


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


def process_image_set(data_set, frame_df, rc, base_dir, image_dir, edge_image_dir, set_type, is_snail_radar=False):
    total_len = len(data_set)
    last_output = 0
    start_time = time.time()
    for i in range(total_len):

        frame, index, x, y, z, u, v, v_r, rcs = data_set[i, :]
        frame, index = int(frame), int(index)
        u, v = int(u), int(v)

        # Determine directories and file paths based on dataset type
        if not os.path.exists(f'{base_dir}/RCS_patch_{rc}'):
            os.mkdir(f'{base_dir}/RCS_patch_{rc}')
        local_pic_dir = f'{base_dir}/RCS_patch_{rc}/{frame}_{index}.jpg'
        if is_snail_radar:
            timestamp, cam_left_time, cam_right_time, radar_arg_time, radar_eagle_time, lidar_time, odom_time = \
            frame_df[frame]
            image_path = f'{image_dir}/left/{cam_left_time}.jpg'
        else:
            image_path = f'{image_dir}/{str(frame).zfill(5)}.jpg'

        # Crop and save local image
        #if not os.path.exists(local_pic_dir):
        local_pic_tmp = crop_image(cv2.imread(image_path), u, v, rc)
        cv2.imwrite(local_pic_dir, local_pic_tmp)

        # Process edge image
        if not os.path.exists(f'{base_dir}/edge_patch_{rc}'):
            os.mkdir(f'{base_dir}/edge_patch_{rc}')
        edge_pic_dir = f'{base_dir}/edge_patch_{rc}/{frame}_{index}.jpg'
        edge_image_tmp = cv2.imread(f'{edge_image_dir}/{frame}.jpg', cv2.IMREAD_GRAYSCALE)
        #print(edge_image_tmp.shape)
        #if not os.path.exists(edge_pic_dir):
        edge_image = crop_image(edge_image_tmp, u, v, rc, True)
        #print(edge_image.shape)
        #print('--------------------------------------')
        cv2.imwrite(edge_pic_dir, edge_image)

        # Progress tracking
        progress = int((i + 1) / total_len * 100)
        if progress // 10 > last_output:
            elapsed_time = time.time() - start_time
            print(f'{base_dir} {set_type} {rc}: {progress}% completed, elapsed time: {elapsed_time:.2f} seconds')
            last_output = progress // 10


def main():
    VoD_train = pd.read_csv('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/RCS_dataset_train.csv').values[:, 1:]
    VoD_test = pd.read_csv('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/RCS_dataset_test.csv').values[:, 1:]

    SR_train = pd.read_csv('D:/snail_radar/20231208/data4/eagle/RCS_dataset_train.csv').values[:, 1:]
    SR_test = pd.read_csv('D:/snail_radar/20231208/data4/eagle/RCS_dataset_test.csv').values[:, 1:]

    columns_to_read_as_str = ['timestamp', 'cam_left_time', 'cam_right_time', "radar_arg_time", "radar_eagle_time",
                              'lidar_time', 'odom_time']
    frame_df = pd.read_csv('D:/snail_radar/20231208/data4/frames_timestamps.csv', usecols=columns_to_read_as_str,
                           dtype=str).values

    #rc_values = [50, 150]
    rc_values = [50,150, 200, 250,300]

    # Define directory paths for each dataset
    #VoD_base_dir = 'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset'
    SR_base_dir = 'D:/snail_radar/20231208/data4/eagle/'
    #image_dir = 'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/image_2'
    #edge_image_dir_VoD = 'D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/edge_image'
    edge_image_dir_SR = 'D:/snail_radar/20231208/data4/edge_image'
    SR_image_dir = 'D:/snail_radar/20231208/data4/zed2i'

    with ThreadPoolExecutor() as executor:
        futures = []
        for rc in rc_values:
            print(f'Starting tasks for rc={rc}')

            # futures.append(executor.submit(process_image_set, VoD_train, frame_df, rc, VoD_base_dir,
            #                                image_dir, edge_image_dir_VoD, 'training', is_snail_radar=False))
            # futures.append(executor.submit(process_image_set, VoD_test, frame_df, rc, VoD_base_dir,
            #                                image_dir, edge_image_dir_VoD, 'test', is_snail_radar=False))
            futures.append(executor.submit(process_image_set, SR_train, frame_df, rc, SR_base_dir,
                                           SR_image_dir, edge_image_dir_SR, 'training', is_snail_radar=True))
            futures.append(executor.submit(process_image_set, SR_test, frame_df, rc, SR_base_dir,
                                           SR_image_dir, edge_image_dir_SR, 'test', is_snail_radar=True))

        # Ensure all tasks complete
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
