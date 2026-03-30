"""Evaluation script for DisNet-based point distribution estimation.

The model inference and metric computation logic were intentionally kept unchanged.
"""

import argparse
import os
import random
import re

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Cls_Reg_Net import *
from DisNet import DisNet, KLDivergenceLoss
from dataset.dataset import (
    Dis_Dataset_VoD,
    Dis_Dataset_astyx,
    Dis_Dataset_MSC,
    # Dis_Dataset_Snail,
    # Dis_Dataset_Snail_eagle,
)

DEVICE = "cuda:3"
DEFAULT_VOD_BASE_DIR = os.environ.get(
    "VOD_PMAP_BASE_DIR",
    "/workspace/data/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset",
)
DEFAULT_ASTYX_BASE_DIR = os.environ.get(
    "ASTYX_BASE_DIR",
    "/workspace/data/Astyx/dataset_astyx_hires2019",
)
# DEFAULT_SNAIL_BASE_DIR = os.environ.get(
#     "SNAIL_BASE_DIR",
#     "/workspace/data/SR/data4",
# )
# DEFAULT_SNAIL_EAGLE_BASE_DIR = os.environ.get(
#     "SNAIL_EAGLE_BASE_DIR",
#     "/workspace/data/SR/data4/eagle",
# )


def seed_everything(seed=2024):
    """Seed common random number generators for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_dataset_base_dir(dataset: str) -> str:
    mapping = {
        "VoD": DEFAULT_VOD_BASE_DIR,
        "astyx": DEFAULT_ASTYX_BASE_DIR,
        # "snail_radar": DEFAULT_SNAIL_BASE_DIR,
        # "snail_radar_eagle": DEFAULT_SNAIL_EAGLE_BASE_DIR,
    }
    if dataset not in mapping:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return mapping[dataset]


def dataset_path(dataset: str, *parts: str) -> str:
    return os.path.join(get_dataset_base_dir(dataset), *parts)


def frame_index(filename):
    """Parse a file name following the pattern 'frame_index.jpg'."""
    match = re.search(r"(\d+)_(\d+)\.jpg", filename)
    if not match:
        raise ValueError(
            f"Filename {filename} does not match the expected pattern 'frame_index.jpg'"
        )
    frame = int(match.group(1))
    index = int(match.group(2))
    return frame, index


def files2df(files, df):
    """Collect rows from a dataframe that correspond to a list of image file names."""
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
    """Normalize model outputs to grayscale and save them as image files."""
    os.makedirs(save_dir, exist_ok=True)

    batch_size, channels, height, width = output.shape
    x_min = output.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
    x_max = output.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]

    output = (output - x_min) / (x_max - x_min + 1e-8)
    output = output.clamp(0, 1)

    for i in range(batch_size):
        img = output[i, 0].cpu().detach().numpy()
        img = (img * 255).astype(np.uint8)

        filename = f"{frame[i]}.jpg"
        filepath = os.path.join(save_dir, filename)
        cv2.imwrite(filepath, img)


def pixel_mse_loss(output_gray, criterion, target):
    """Compute pixel-space MSE after min-max normalization."""
    x_min = output_gray.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
    x_max = output_gray.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    output_gray = (output_gray - x_min) / (x_max - x_min + 1e-8)
    output_gray = output_gray.clamp(0, 1)

    loss0 = criterion(output_gray, target / 255)
    return loss0


def load_frame_timestamps(dataset: str):
    """Load frame timestamp metadata when required by a dataset."""
    if dataset not in {"snail_radar", "snail_radar_eagle"}:
        return None

    columns_to_read_as_str = [
        "timestamp",
        "cam_left_time",
        "cam_right_time",
        "radar_arg_time",
        "radar_eagle_time",
        "lidar_time",
        "odom_time",
    ]
    csv_path = dataset_path("snail_radar", "frames_timestamps.csv")
    return pd.read_csv(csv_path, usecols=columns_to_read_as_str, dtype=str).values


def get_test_image_dir(dataset: str, log: str) -> str:
    """Return the directory used to save visualization images during evaluation."""
    if dataset == "VoD":
        return dataset_path(dataset, "exp_2", f"test_image_{log}")
    if dataset == "astyx":
        return dataset_path(dataset, "ablation", f"test_image_{log}")
    # if dataset == "snail_radar":
    #     return dataset_path(dataset, "ablation", "test_image_wo_F")
    # if dataset == "snail_radar_eagle":
    #     return dataset_path(dataset, "ablation", "test_image_wo_v")
    raise ValueError(f"Unsupported dataset: {dataset}")


def test_dis(args, dataset, log, weights=None):
    """Run evaluation on the validation split and export predicted visualizations."""
    base_dir = get_dataset_base_dir(dataset)
    frame_df = load_frame_timestamps(dataset)
    device = torch.device(args.device)

    df = pd.read_csv(os.path.join(base_dir, "prob_dataset.csv")).values
    train_data = pd.read_csv(os.path.join(base_dir, "train_dataset.csv")).values
    valid_data = pd.read_csv(os.path.join(base_dir, "test_dataset.csv")).values

    train_dataloader, test_dataloader = None, None

    if dataset == "VoD":
        train_dataloader = DataLoader(
            dataset=Dis_Dataset_VoD(train_data),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
        )
    elif dataset == "astyx":
        train_dataloader = DataLoader(
            dataset=Dis_Dataset_astyx(train_data),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=4,
        )
    # elif dataset == "snail_radar":
    #     train_dataloader = DataLoader(
    #         dataset=Dis_Dataset_Snail(train_data, frame_df),
    #         batch_size=args.batch_size,
    #         shuffle=True,
    #         drop_last=False,
    #         num_workers=4,
    #     )
    # elif dataset == "snail_radar_eagle":
    #     train_dataloader = DataLoader(
    #         dataset=Dis_Dataset_Snail_eagle(train_data, frame_df),
    #         batch_size=args.batch_size,
    #         shuffle=True,
    #         drop_last=False,
    #         num_workers=4,
    #     )

    model = torch.load(weights).to(device)
    criterion = torch.nn.MSELoss(reduction="mean")
    kl_loss = KLDivergenceLoss()
    min_val_loss = 10_000_000

    model.eval()

    if dataset == "VoD":
        valid_dataloader = DataLoader(
            dataset=Dis_Dataset_VoD(valid_data),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
        )
    elif dataset == "astyx":
        valid_dataloader = DataLoader(
            dataset=Dis_Dataset_astyx(valid_data),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=4,
        )
    # elif dataset == "snail_radar":
    #     valid_dataloader = DataLoader(
    #         dataset=Dis_Dataset_Snail(valid_data, frame_df),
    #         batch_size=args.batch_size,
    #         shuffle=True,
    #         drop_last=False,
    #         num_workers=4,
    #     )
    # elif dataset == "snail_radar_eagle":
    #     valid_dataloader = DataLoader(
    #         dataset=Dis_Dataset_Snail_eagle(valid_data, frame_df),
    #         batch_size=args.batch_size,
    #         shuffle=True,
    #         drop_last=False,
    #         num_workers=4,
    #     )

    total_val_loss = 0
    total_val_loss1, total_val_loss2 = 0, 0
    total_val_loss0 = 0
    total_val_num = 0
    total_loss2_percentage = 0
    cnt = 0
    new_df = pd.DataFrame(columns=["Frame", "Estimated_PointNum"])

    with torch.no_grad():
        for idx, (batch_dict, target, pointnum) in enumerate(valid_dataloader):
            target = target.to(device).unsqueeze(1).float()
            pointnum = pointnum.to(device)
            frame = batch_dict["frame"].reshape(len(target), -1).cpu().detach().numpy()

            output_gray, output_num = model(batch_dict)
            output_gray, output_num = output_gray.float(), output_num.float()

            output_num = output_num.reshape(-1, 1)
            pointnum = pointnum.reshape(-1, 1)
            num_est = output_num.cpu().detach().numpy() * 1000

            for i in range(len(num_est)):
                new_data = pd.DataFrame(
                    {"Frame": [frame[i, 0]], "Estimated_PointNum": [num_est[i, 0]]}
                )
                if not new_data.empty and not new_data.isna().all(axis=None):
                    new_df = pd.concat([new_df, new_data], ignore_index=True)

            loss1 = kl_loss(output_gray, target)
            error = abs(output_num * 1000 - pointnum) / pointnum
            loss2_percentage = criterion(error, torch.zeros_like(error))
            total_loss2_percentage += loss2_percentage.item()

            save_images(output_gray, batch_dict["frame"], get_test_image_dir(dataset, log))

            loss = loss1 + loss2_percentage
            total_val_loss += loss.item()
            total_val_loss1 += loss1.item()
            cnt += 1

        total_val_loss /= cnt
        print("validation set:")
        print("min total loss = ", min_val_loss)
        print(f"avg KL Divergence loss: ", total_val_loss1 / cnt)
        print(f"avg number percentage loss: ", total_loss2_percentage / cnt)


if __name__ == "__main__":
    seed_everything()
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=DEVICE, help="GPU device to use.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    args = parser.parse_args()

    weights = None
    dataset = "VoD"
    log_name = "radar_depth_w_smooth_loss_2"
    print(f"testing on {log_name}")

    if dataset == "VoD":
        weights = os.path.join(
            get_dataset_base_dir(dataset),
            "exp_RPGen",
            "weights",
            f"model_best_{dataset}_{log_name}",
            f"model_best_{dataset}_{log_name}_e28.pth",
        )
    elif dataset == "astyx":
        weights = os.path.join(
            get_dataset_base_dir(dataset),
            "exp_2",
            "weights",
            "ablation_wo_F_model_best.pth",
        )
    # elif dataset == "snail_radar":
    #     weights = os.path.join(
    #         get_dataset_base_dir(dataset),
    #         "exp_2",
    #         "weights",
    #         "ablation_wo_F_model_best.pth",
    #     )
    # elif dataset == "snail_radar_eagle":
    #     weights = os.path.join(
    #         get_dataset_base_dir(dataset),
    #         "exp_2",
    #         "weights",
    #         "ablation_wo_v_model_best.pth",
    #     )

    if os.path.exists(weights):
        test_dis(args, dataset, log_name, weights)
    else:
        print("weight not found.")
