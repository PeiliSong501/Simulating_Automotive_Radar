# Reproduction Guide

This repository currently contains three practical stages for simulated automotive radar generation:

1. **Dis-Net**: predict the radar point distribution / probability map.
2. **Depth and Doppler generation**: recover 3D structure and assign depth / velocity.
3. **RSS-Net**: predict radar reflectivity.

> Note: Due to code loss caused by a server crash, some project content is still slightly incomplete. This project is still being updated.

---

## 1. Overall pipeline

The intended simulation order is:

1. Run **Dis-Net**.
2. Generate **depth** and **Doppler velocity**.
3. Run **RSS-Net**.

In other words:

- **Dis-Net** determines **where radar points should appear**.
- **Depth / velocity generation** determines **how far the points are** and **their Doppler-related motion**.
- **RSS-Net** determines **the reflectivity / RSS value** of each generated radar point.

To start, you should download related datasets and organize them using the default settings..
---

## 2. Stage-by-stage requirements

### 2.1 Dis-Net

#### Purpose
Dis-Net is used to predict the radar point distribution / probability map.

#### Main script
- `train_val_scripts/Dis_train.py`

#### Main model file
- `models/DisNet.py`

#### Dataset file
- `dataset/dataset.py`

#### Files required before running Dis-Net
For the current codebase, the training script expects the following CSV files under the dataset base directory:

- `prob_dataset.csv`
- `train_dataset.csv`
- `test_dataset.csv`

For VoD, the dataset loader also reads:

- local camera images from `image_2/`
- probability-map target images (needs to be generated in gt_generation.py)
- ego velocity from `radar_velo.npy`(needs to be generated in velocity_estimation.py)

So, before running Dis-Net, the required inputs are:

- train / test CSV split files
- probability-map supervision images
- local camera images
- ego velocity data

#### Where these files are generated
These files are produced by the preprocessing / GT-generation pipeline, mainly from:

- `gt_generation.py`

#### How to run
```bash
cd train_val_scripts
python Dis_train.py --device {DEVICE} --batch_size 4
```

#### Important notes
- In the current public code, the `dataset` variable is still set inside `Dis_train.py`.
- You may need to edit dataset-related paths directly in the script.

---

### 2.2 Depth and Doppler generation

#### Purpose
This stage reconstructs depth and assigns Doppler-related velocity after the Dis-Net output is available.

#### Main files
- `depth_generation_vod.py`
- `dynamic_objects_stats.py`
- `velocity_estimation.py`
- `lidar_interpolation.py`
- `lidar_depth_map_builder.py`

#### What this stage uses
This stage conceptually uses:

- Dis-Net outputs
- camera intrinsics `K`
- radar-to-camera transform `T_radar`
- LiDAR-derived depth maps or interpolated depth maps
- radar / ego velocity information

#### What each file mainly does
- `Network/depth_generation_vod.py`: reconstructs 3D radar points from 2D positions and depth values.
- `dynamic_objects_stats.py`: estimates ego velocity for radar and velocity of dynamic objects.
- `velocity_estimation.py`: estimates target velocity using least-squares / RANSAC-based methods.
- `lidar_interpolation.py`: interpolates or densifies local depth.
- `lidar_depth_map_builder.py`: prepares / builds LiDAR depth maps.

#### Recommended rename
The current name `lidar_depth_map_builder.py` is not very descriptive. A better name would be:

#### How to run
This stage is not yet a fully standardized single command in the current public code. In practice, it is run through the corresponding scripts after Dis-Net outputs are available.

A practical order is:

1. Use Dis-Net results to determine radar-point image locations / distributions.
2. Use `depth_generation_vod.py` to recover 3D structure.
3. Use `dynamic_objects_stats.py` to obtain ego velo and stats of velocity for all dynamic objects (prerequisite of `velocity_estimation.py`).
4. Use `velocity_estimation.py` to estimate or assign Doppler velocity.

> At the moment, this stage may require manual editing of file paths or frame-processing logic inside the scripts.

---

### 2.3 RSS-Net

#### Purpose
RSS-Net predicts the reflectivity / RSS value of each generated radar point.

#### Related scripts
- `train_val_scripts/RSS_train.py`
- `models/RSSNet.py`
- `RSS_dataset_train.csv`
- `RSS_dataset_test.csv`

#### Main script
- `train_val_scripts/RSS_train.py`

#### Main model file
- `models/RSSNet.py`

#### Dataset file
- `dataset/dataset.py`

#### Files required before running RSS-Net
For VoD, the current training script expects:

- `RSS_dataset_train.csv`
- `RSS_dataset_test.csv`

In addition, the dataset loader also reads:

- `range_image/`
- local image patches such as `RCS_patch_<rc>/`
- `edge_image/`
- ego velocity from `radar_velo.npy`

So, before running RSS-Net, the required inputs are:

- RSS/RCS train/test CSV files
- generated `range_image`
- generated local patches
- generated edge images
- radar / ego velocity information

#### Where these files are generated
These files are mainly produced by:

- `gt_generation_p.py`
- related patch / range-image generation code
- auxiliary preprocessing scripts

#### How to run
```bash
cd train_val_scripts
python RCS_train.py --device {DEVICE} --batch_size 8
```

#### Important notes
- You may need to edit the `dataset`, `rc`, and path settings directly inside the script.

---

## 3. Practical execution order

### Step 1. Run preprocessing / GT generation
Run the preprocessing and GT-generation scripts first.

Main related files:

- `gt_generation_p.py`
- `lidar_interpolation.py`
- `lidar_down.py`
- dataset-specific preparation scripts if needed

This stage should generate the assets required later, including:

- `prob_dataset.csv`
- `train_dataset.csv`
- `test_dataset.csv`
- `RCS_dataset_train.csv`
- `RCS_dataset_test.csv`
- probability-map images
- `range_image/`
- local patches
- `edge_image/`

---

### Step 2. Run Dis-Net
```bash
cd train_val_scripts
python train.py --device {DEVICE} --batch_size 4
```

This stage uses:

- `prob_dataset.csv`
- `train_dataset.csv`
- `test_dataset.csv`
- local images
- probability-map targets
- `radar_velo.npy`

Output:

- predicted radar point distribution / probability map

---

### Step 3. Generate depth and Doppler velocity
Use the depth / velocity generation scripts after Dis-Net results are available.

Main related files:

- `depth_generation_vod.py`
- `velocity_estimation.py`

Conceptually:

1. Use the Dis-Net result to determine radar-point image locations.
2. Recover depth / 3D structure.
3. Estimate or assign Doppler velocity.

This stage currently may require manual path editing and script-specific adjustments.

---

### Step 4. Run RSS-Net
```bash
cd train_val_scripts
python RCS_train.py --device cuda:0 --batch_size 8
```

This stage uses:

- `RCS_dataset_train.csv`
- `RCS_dataset_test.csv`
- `range_image/`
- local patches
- `radar_velo.npy`

Output:

- predicted RSS, which is reflectivity values (may or may not be RCS)

---

