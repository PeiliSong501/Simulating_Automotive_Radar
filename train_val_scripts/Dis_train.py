"""Training script for DisNet-based point distribution estimation.

"""

import argparse
import logging
import multiprocessing
import os
import random
import re
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from DisNet import DisNet, KLDivergenceLoss, MultiTaskLoss, NormalizedLoss
from dataset.dataset import (
    Dis_Dataset_VoD,
    Dis_Dataset_astyx,
    Dis_Dataset_MSC,
    Dis_Dataset_hercules,
    Dis_Dataset_Snail,
    Dis_Dataset_Snail_eagle,
)

torch.cuda.empty_cache()

DEVICE = "cuda:2"
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
DEFAULT_HERCULES_BASE_DIR = os.environ.get(
    "HERCULES_PMAP_BASE_DIR",
    "/workspace/data/HeRCULES/Library_01_Day/pmap_dataset",
)
DEFAULT_HERCULES_ROOT_DIR = os.environ.get(
    "HERCULES_ROOT_DIR",
    "/workspace/data/HeRCULES/Library_01_Day",
)
DEFAULT_MSC_BASE_DIR = os.environ.get(
    "MSC_PMAP_BASE_DIR",
    "/workspace/data/MSC-Rad4R/URBAN_D0/pmap_dataset",
)

def setup_logging(log_file):
    """Configure file and console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(),
        ],
    )


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
        "MSC": DEFAULT_MSC_BASE_DIR,
        "astyx": DEFAULT_ASTYX_BASE_DIR,
        # "snail_radar": DEFAULT_SNAIL_BASE_DIR,
        # "snail_radar_eagle": DEFAULT_SNAIL_EAGLE_BASE_DIR,
        "hercules": DEFAULT_HERCULES_BASE_DIR,
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

    x_min_t = target.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
    x_max_t = target.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    target_norm = (target - x_min_t) / (x_max_t - x_min_t + 1e-8)
    target_norm = target_norm.clamp(0, 1)

    loss0 = criterion(output_gray, target_norm)
    return loss0


def load_frame_timestamps(dataset: str):
    """Load frame timestamp metadata when required by a dataset."""
    if dataset in {"snail_radar", "snail_radar_eagle"}:
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

    if dataset == "hercules":
        columns_to_read_as_str = [
            "frame",
            "timestamp",
            "cam_left_time",
            "cam_right_time",
            "radar_continental_time",
            "lidar_time",
            "odom_time",
        ]
        csv_path = os.path.join(DEFAULT_HERCULES_ROOT_DIR, "frames_timestamps.csv")
        return pd.read_csv(csv_path, usecols=columns_to_read_as_str, dtype=str).values

    return None


def get_original_size(dataset: str):
    """Return the target image size expected by the model."""
    if dataset == "VoD":
        return [644, 1935]
    if dataset == "MSC":
        return [540, 720 - 77]
    if dataset == "hercules":
        return [260, 1440]
    if dataset == "astyx":
        return [618, 2048]
    # if dataset in {"snail_radar", "snail_radar_eagle"}:
    #     return [316, 640]
    raise ValueError(f"Unsupported dataset: {dataset}")


def build_train_dataloader(dataset: str, train_data, frame_df, batch_size: int):
    """Create the training dataloader for the selected dataset."""
    if dataset == "VoD":
        return DataLoader(
            dataset=Dis_Dataset_VoD(train_data),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=4,
        )
    if dataset == "MSC":
        return DataLoader(
            dataset=Dis_Dataset_MSC(train_data),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=4,
        )
    if dataset == "hercules":
        return DataLoader(
            dataset=Dis_Dataset_hercules(train_data, frame_df),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=4,
        )
    if dataset == "astyx":
        return DataLoader(
            dataset=Dis_Dataset_astyx(train_data),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=4,
        )
    raise ValueError(f"Unsupported dataset in current script: {dataset}")


def build_valid_dataloader(dataset: str, valid_data, frame_df, batch_size: int):
    """Create the validation dataloader for the selected dataset."""
    if dataset == "VoD":
        return DataLoader(
            dataset=Dis_Dataset_VoD(valid_data),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=4,
        )
    if dataset == "MSC":
        return DataLoader(
            dataset=Dis_Dataset_MSC(valid_data),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=4,
        )
    if dataset == "hercules":
        return DataLoader(
            dataset=Dis_Dataset_hercules(valid_data, frame_df),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=4,
        )
    if dataset == "astyx":
        return DataLoader(
            dataset=Dis_Dataset_astyx(valid_data),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=4,
        )
    raise ValueError(f"Unsupported dataset in current script: {dataset}")


def get_test_image_dir(dataset: str, log: str) -> str:
    """Return the directory used to save visualization images during validation."""
    if dataset == "VoD":
        return dataset_path(dataset, "exp_2", f"test_image_{log}")
    if dataset == "astyx":
        return dataset_path(dataset, "ablation", "test_image_wo_v")
    if dataset == "MSC":
        return dataset_path(dataset, "exp_2", f"test_image_{log}")
    if dataset == "hercules":
        return dataset_path(dataset, "exp_2", f"test_image_{log}")
    raise ValueError(f"Unsupported dataset in current script: {dataset}")


def train_dis(args, epochs, dataset, log=None, weights=None):
    """Train the distribution model and save the best checkpoint."""
    if log is not None:
        os.makedirs("./dis_log", exist_ok=True)
        log_file = f"./dis_log/training_log_{log}.txt"
        setup_logging(log_file)
        logging.info(f"Training started for dataset: {dataset}")
        logging.info(f"Batch size: {args.batch_size}, Epochs: {epochs}")

    base_dir = get_dataset_base_dir(dataset)
    frame_df = load_frame_timestamps(dataset)
    device = torch.device(args.device)

    df = pd.read_csv(os.path.join(base_dir, "prob_dataset.csv")).values
    train_data = pd.read_csv(os.path.join(base_dir, "train_dataset.csv")).values
    valid_data = pd.read_csv(os.path.join(base_dir, "test_dataset.csv")).values

    train_dataloader = build_train_dataloader(dataset, train_data, frame_df, args.batch_size)
    original_size = get_original_size(dataset)

    if weights is not None:
        model = torch.load(weights).to(device)
        print("Continue training from existing weights.")
    else:
        model = DisNet(original_size).to(device)

    criterion = torch.nn.MSELoss(reduction="mean")
    kl_loss = KLDivergenceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    min_val_loss = 10_000_000

    for e in range(epochs):
        len_td = len(train_dataloader)
        checkpoint = len_td * 0.10
        next_checkpoint = checkpoint
        last_time = time.time()

        if log is not None:
            logging.info(f"Epoch {e + 1}/{epochs} started.")

        print("epoch ", e)
        total_loss1, total_loss0 = 0, 0
        cnt = 0
        model.train()
        total_loss2_percentage = 0

        for idx, (batch_dict, target, pointnum) in enumerate(train_dataloader):
            target = target.to(device).unsqueeze(1).float()
            pointnum = pointnum.to(device)

            output_gray, output_num = model(batch_dict)
            output_gray, output_num = output_gray.float(), output_num.float()

            output_num = output_num.reshape(-1, 1)
            pointnum = pointnum.reshape(-1, 1)

            loss0 = pixel_mse_loss(output_gray, criterion, target)
            loss1 = kl_loss(output_gray, target)

            if dataset == "MSC":
                error = abs(output_num * 2500 - pointnum) / pointnum
            else:
                error = abs(output_num * 1000 - pointnum) / pointnum

            loss2_percentage = criterion(error, torch.zeros_like(error))
            total_loss2_percentage += loss2_percentage.item()

            loss = loss1 + loss2_percentage + loss0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss1 += loss1.item()
            total_loss0 += loss0.item()
            cnt += 1

            # Show progress every ~10% of the training loader.
            if idx + 1 >= next_checkpoint:
                current_time = time.time()
                elapsed_time = current_time - last_time
                last_time = current_time
                while next_checkpoint <= idx + 1:
                    next_checkpoint += checkpoint
                percent = (idx + 1) / len_td * 100
                print(
                    f"Processed {percent:.1f}% of batches. "
                    f"Time elapsed: {elapsed_time:.2f} seconds."
                )

        if log is not None:
            logging.info(f"avg KL Divergence loss of epoch {e}: {total_loss1 / cnt}")
            logging.info(
                f"avg number percentage loss of epoch {e}: {total_loss2_percentage / cnt}"
            )
            logging.info(f"sum pixel mse loss of epoch {e}: {total_loss0 / cnt}")

        print("--------------------------------------------------------------------------------")

        # ---------------------------------- validation ----------------------------------
        model.eval()
        valid_dataloader = build_valid_dataloader(dataset, valid_data, frame_df, args.batch_size)

        total_val_loss = 0
        total_val_loss1 = 0
        total_val_loss0 = 0
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
                if dataset == "MSC":
                    num_est = output_num.cpu().detach().numpy() * 2500

                for i in range(len(num_est)):
                    new_data = pd.DataFrame(
                        {"Frame": [frame[i, 0]], "Estimated_PointNum": [num_est[i, 0]]}
                    )
                    if not new_data.empty and not new_data.isna().all(axis=None):
                        new_df = pd.concat([new_df, new_data], ignore_index=True)

                loss1 = kl_loss(output_gray, target)
                loss0 = pixel_mse_loss(output_gray, criterion, target)

                error = abs(output_num * 1000 - pointnum) / pointnum
                loss2_percentage = criterion(error, torch.zeros_like(error))

                save_images(output_gray, batch_dict["frame"], get_test_image_dir(dataset, log))

                loss = loss1 + loss2_percentage + loss0

                total_val_loss += loss.item()
                total_val_loss1 += loss1.item()
                total_loss2_percentage += loss2_percentage.item()
                total_val_loss0 += loss0.item()
                cnt += 1

            total_val_loss /= cnt
            print("validation set:")

            if log is not None:
                logging.info(f"Validation total loss = {min_val_loss}")
                logging.info(f"avg KL Divergence loss:  {total_val_loss1 / cnt}")
                logging.info(f"avg number percentage loss:  {total_loss2_percentage / cnt}")
                logging.info(f"sum of pixel mse loss:  {total_val_loss0 / cnt}")

            if total_val_loss < min_val_loss:
                min_val_loss = total_val_loss
                logging.info(f"min total loss = {total_val_loss}, model saved!")

                weight_dir = os.path.join(base_dir, "exp_2", "weights", f"model_best_{log}")
                os.makedirs(weight_dir, exist_ok=True)

                modelsave = os.path.join(weight_dir, f"model_best_{log}_e{e}.pth")
                torch.save(model, modelsave)
                new_df.to_csv(os.path.join(base_dir, "exp_2", f"estimated_pointnum_{log}.csv"))


if __name__ == "__main__":
    seed_everything()
    torch.cuda.empty_cache()
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=DEVICE, help="GPU device to use.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    args = parser.parse_args()

    epochs = 50
    dataset = "hercules"
    weights = None
    log_name = f"{dataset}_dis_net"
    print("running", log_name)

    if dataset == "astyx":
        weights = os.path.join(
            get_dataset_base_dir(dataset),
            "exp_2",
            "weights",
            "ablation_wo_F_model_best.pth",
        )
    else:
        weights = os.path.join(
            get_dataset_base_dir(dataset),
            "exp_2",
            "weights",
            f"model_best_{log_name}.pth",
        )

    if os.path.exists(weights):
        train_dis(args, epochs, dataset, log_name, weights)
    else:
        train_dis(args, epochs, dataset, log_name)
