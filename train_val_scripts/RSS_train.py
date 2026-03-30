"""Training script for RSS/RCS regression.
Results for test set is directly saved in this script.

"""

import argparse
import logging
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from RSSNet import RSSNet
from dataset.dataset import (
    RSS_Dataset_VoD,
    RSS_Dataset_astyx,
    RSS_Dataset_MSC,
    # RSS_Dataset_Snail,
    # RSS_Dataset_Snail_eagle,
)


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
DEFAULT_MSC_BASE_DIR = os.environ.get(
    "MSC_PMAP_BASE_DIR",
    "/workspace/data/MSC-Rad4R/URBAN_D0/pmap_dataset",
)


def setup_logging(log_file: str) -> None:
    """Configure file and console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(),
        ],
    )


def seed_everything(seed: int = 2024) -> None:
    """Seed common random number generators for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_dataset_base_dir(dataset: str) -> str:
    """Return the root directory for a supported dataset."""
    mapping = {
        "VoD": DEFAULT_VOD_BASE_DIR,
        "astyx": DEFAULT_ASTYX_BASE_DIR,
        # "snail_radar": DEFAULT_SNAIL_BASE_DIR,
        # "snail_radar_eagle": DEFAULT_SNAIL_EAGLE_BASE_DIR,
        "MSC": DEFAULT_MSC_BASE_DIR,
    }
    if dataset not in mapping:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return mapping[dataset]


def dataset_path(dataset: str, *parts: str) -> str:
    """Build a path under a dataset-specific root directory."""
    return os.path.join(get_dataset_base_dir(dataset), *parts)


def get_rcs_range(dataset: str) -> tuple[float, float]:
    """Return min/max values used for label normalization."""
    if dataset == "VoD":
        return -74.01181, 39.788345
    if dataset == "astyx":
        return 45.0, 95.0
    # if dataset == "snail_radar":
    #     return 108.0, 178.0
    # if dataset == "snail_radar_eagle":
    #     return 0.0, 38.13
    if dataset == "MSC":
        return 1.5049268, 54.761814
    raise ValueError(f"Unsupported dataset: {dataset}")


def load_frame_timestamps(dataset: str):
    """Load frame timestamp metadata when a dataset requires it."""
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


def build_dataloaders(args, dataset: str, rc: int, frame_df=None):
    """Create train and validation dataloaders for the selected dataset."""
    if dataset == "VoD":
        train_data = pd.read_csv(dataset_path(dataset, "RCS_dataset_train.csv")).values[:, 1:]
        valid_data = pd.read_csv(dataset_path(dataset, "RCS_dataset_test.csv")).values[:, 1:]
        train_dataloader = DataLoader(
            dataset=RSS_Dataset_VoD(train_data, rc),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=4,
        )
        valid_dataloader = DataLoader(
            dataset=RSS_Dataset_VoD(valid_data, rc),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=4,
        )
    elif dataset == "astyx":
        train_data = pd.read_csv(dataset_path(dataset, "RCS_dataset_train.csv")).values[:, 1:]
        valid_data = pd.read_csv(dataset_path(dataset, "RCS_dataset_test.csv")).values[:, 1:]
        train_dataloader = DataLoader(
            dataset=RSS_Dataset_astyx(train_data),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=4,
        )
        valid_dataloader = DataLoader(
            dataset=RSS_Dataset_astyx(valid_data),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=4,
        )
    elif dataset == "snail_radar":
        train_data = pd.read_csv(dataset_path(dataset, "RCS_dataset_train.csv")).values[:, 1:]
        valid_data = pd.read_csv(dataset_path(dataset, "RCS_dataset_test.csv")).values[:, 1:]
        train_dataloader = DataLoader(
            dataset=RSS_Dataset_Snail(train_data, frame_df),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=4,
        )
        valid_dataloader = DataLoader(
            dataset=RSS_Dataset_Snail(valid_data, frame_df),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=4,
        )
    elif dataset == "snail_radar_eagle":
        train_data = pd.read_csv(dataset_path(dataset, "RCS_dataset_train.csv")).values[:, 1:]
        valid_data = pd.read_csv(dataset_path(dataset, "RCS_dataset_test.csv")).values[:, 1:]
        train_dataloader = DataLoader(
            dataset=RSS_Dataset_Snail_eagle(train_data, frame_df),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=4,
        )
        valid_dataloader = DataLoader(
            dataset=RSS_Dataset_Snail_eagle(valid_data, frame_df),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=4,
        )
    elif dataset == "MSC":
        train_data = pd.read_csv(dataset_path(dataset, "RCS_dataset_train.csv")).values[:, 1:]
        valid_data = pd.read_csv(dataset_path(dataset, "RCS_dataset_test.csv")).values[:, 1:]
        train_dataloader = DataLoader(
            dataset=RSS_Dataset_MSC(train_data, rc),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=4,
        )
        valid_dataloader = DataLoader(
            dataset=RSS_Dataset_MSC(valid_data, rc),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=4,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return train_dataloader, valid_dataloader


def build_model(dataset: str, batch_size: int, device: str, weights: str | None = None):
    """Create or resume the regression model."""
    if weights is not None:
        model = torch.load(weights).to(device)
        print("Continue training from existing weights.")
        return model

    if dataset == "VoD":
        return RSSNet(batch_size, 5).to(device)
    if dataset == "astyx":
        return RSSNet(batch_size, 4).to(device)
    if dataset in {"snail_radar", "snail_radar_eagle", "MSC"}:
        return RSSNet(batch_size, 5).to(device)
    raise ValueError(f"Unsupported dataset: {dataset}")


def train_rcs(args, epochs, dataset, ablation, rc, weights=None):
    """Train the RSS/RCS regression model and save the best checkpoint."""
    os.makedirs("./rcs_log", exist_ok=True)
    log_file = f"./rcs_log/training_log_{dataset}_{ablation}.txt"
    setup_logging(log_file)
    logging.info(f"Training started for dataset: {dataset}")
    logging.info(f"Batch size: {args.batch_size}, Epochs: {epochs}")

    base_dir = get_dataset_base_dir(dataset)
    min_value, max_value = get_rcs_range(dataset)
    frame_df = load_frame_timestamps(dataset)

    train_dataloader, valid_dataloader = build_dataloaders(args, dataset, rc, frame_df)
    model = build_model(dataset, args.batch_size, args.device, weights)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    min_val_loss = 1_000_000.0

    for e in range(epochs):
        logging.info(f"Epoch {e + 1}/{epochs} started.")
        l_td = len(train_dataloader)
        checkpoint = l_td * 0.10
        next_checkpoint = checkpoint
        last_time = time.time()
        print("epoch ", e)

        total_loss = 0
        cnt = 0
        model.train()

        for idx, (batch_dict, rcs_values) in enumerate(train_dataloader):
            current_batch_size = batch_dict["radarpoint"].size(0)
            label = torch.unsqueeze(rcs_values.to(args.device).float(), 1)

            pred = model(batch_dict, current_batch_size)
            pred, label = pred.reshape(-1, 1), label.reshape(-1, 1)

            pred = (pred - min_value) / (max_value - min_value)
            label = (label - min_value) / (max_value - min_value)
            loss = criterion(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            cnt += 1

            # Show progress every ~10% of the training loader.
            if idx + 1 >= next_checkpoint:
                current_time = time.time()
                elapsed_time = current_time - last_time
                last_time = current_time
                while next_checkpoint <= idx + 1:
                    next_checkpoint += checkpoint
                percent = (idx + 1) / l_td * 100
                print(
                    f"Processed {percent:.1f}% of batches. "
                    f"Time elapsed: {elapsed_time:.2f} seconds."
                )

        logging.info(f"avg training loss of epoch {e}: {total_loss / cnt}")

        # ---------------------------------- validation ----------------------------------
        model.eval()
        total_val_loss = 0
        total_val_loss1 = 0
        cnt = 0
        predictions = []
        labels = []
        frames = []
        indices = []

        with torch.no_grad():
            for idx, (batch_dict, rcs_values) in enumerate(valid_dataloader):
                current_batch_size = batch_dict["radarpoint"].size(0)
                label = torch.unsqueeze(rcs_values.to(args.device).float(), 1)
                pred = model(batch_dict, current_batch_size)

                loss1 = criterion(pred, label)
                predictions.append(pred.cpu().numpy())
                labels.append(label.cpu().numpy())
                frames.extend(batch_dict["frame"].cpu().numpy())
                indices.extend(batch_dict["index"].cpu().numpy())

                pred, label = pred.reshape(-1, 1), label.reshape(-1, 1)
                pred = (pred - min_value) / (max_value - min_value)
                label = (label - min_value) / (max_value - min_value)
                loss = criterion(pred, label)

                total_val_loss += loss.item()
                total_val_loss1 += loss1.item()
                cnt += 1

            total_val_loss /= cnt
            total_val_loss1 /= cnt

            if total_val_loss < min_val_loss:
                min_val_loss = total_val_loss
                weights_dir = os.path.join(base_dir, "ablation", "weights")
                os.makedirs(weights_dir, exist_ok=True)

                modelsave = os.path.join(
                    weights_dir,
                    f"RCS_{dataset}_model_best_{ablation}.pth",
                )
                torch.save(model, modelsave)

                logging.info(f"min loss = {min_val_loss}")
                logging.info(f"total_loss_1:{total_val_loss1}")
                logging.info("Model Saved!")

                predictions = np.concatenate(predictions)
                labels = np.concatenate(labels)
                df_results = pd.DataFrame(
                    {
                        "Frame": frames,
                        "Index": indices,
                        "True Labels": labels.flatten(),
                        "Predictions": predictions.flatten(),
                    }
                )
                df_results.to_csv(
                    os.path.join(
                        base_dir,
                        "ablation",
                        f"RCS_validation_results_ablation_{dataset}_{ablation}.csv",
                    ),
                    index=False,
                )
                logging.info("Validation results saved.")


if __name__ == "__main__":
    # Example:
    # df = pd.read_csv(dataset_path("MSC", "pmap_dataset.csv"))
    # average_pointnum = df["PointNum"].mean()
    # print(f"The average value of 'PointNum' is: {average_pointnum}")

    seed_everything()
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:3", help="GPU device to use.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    args = parser.parse_args()

    epochs = 40
    weights = None
    dataset = "VoD"
    rc = 100
    ablation = f"para_{rc}_1"   
    '''
    2 parameters, including radius of image patch(pixels) and local lidar points (meters)
    For optimal parameters, please refer to our paper.
    '''
    
    print(f"running on dataset {dataset}, ablation study {ablation}")

    weights = os.path.join(
        get_dataset_base_dir(dataset),
        "ablation",
        "weights",
        f"RCS_{dataset}_model_best_{ablation}.pth",
    )

    if os.path.exists(weights):
        train_rcs(args, epochs, dataset, ablation, rc, weights)
    else:
        train_rcs(args, epochs, dataset, ablation, rc)
