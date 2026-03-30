# Simulating Automotive Radar with LiDAR and Camera Inputs

This repository contains the source code for the IROS 2025 paper:

**Simulating Automotive Radar with LiDAR and Camera Inputs**

📄 **arXiv:** https://arxiv.org/abs/2503.08068  
📘 **IEEE Xplore:** https://ieeexplore.ieee.org/document/11247276  
> *Note: the IEEE Xplore version contains minor typos; please refer to the arXiv version for the most accurate text.*

The corresponding video is available on YouTube (https://www.youtube.com/watch?v=dybbkPePHD4&t=1s) and bilibili (https://www.bilibili.com/video/BV1AdKwzLEPW/?spm_id_from=333.1387.0.0&vd_source=eb4269e8fa059487e2a91ce411765468).

---

## Overview

This project focuses on simulating automotive radar signals, including **pitch, yaw, range, velocity, and reflectivity**, by leveraging multimodal sensor inputs such as **LiDAR** and **camera data**.

The simulator is designed to model radar observations in complex driving environments and serves as a research tool for perception, sensor fusion, and learning-based radar understanding.

In addition to the network architectures and training pipelines, this codebase also provides a development toolkit for processing several **4D automotive radar datasets**, such as **VoD** and **MSC-Rad4R**.

---

## Reproduction Guide

A reproduction guide is provided in the following file:

- [`README_reproduction_guide.md`](./readme_reproduction_guide.md)

This document describes:

- the overall simulation pipeline
- the execution order of **Dis-Net → depth & Doppler generation → RSS-Net**
- the files required for each stage
- the scripts currently used to run each step

Please refer to it before running the code.

---

## Code Status

**Last updated: March 2026**

The repository has now been partially reorganized and currently has **basic readability and usability for reproduction purposes**. Compared with the initial release, the code structure is now clearer in several parts of the project.

However, please note that the current version is still not fully polished.

### Current status

- the codebase now has **improved readability compared with the earlier public version**
- several components have been reorganized to make the main pipeline easier to follow
- the repository is **still under cleanup and incremental maintenance**

### Important note

Due to a server crash, part of the original project files was lost. As a result:

- some scripts or supporting code **may** still be incomplete
- some parts of the repository are still being reorganized

This repository is therefore released in a **partially recovered and continuously updated** state.

---

## Future Updates

The repository will continue to be updated in the future. Planned improvements include:

- further cleanup of the project structure
- better code comments and readability
- more complete reproduction instructions
- continued restoration of files affected by the server crash

✉️ If you have urgent questions or require clarification for reproduction purposes, feel free to contact the author via email: **peilisong@mail.nankai.edu.cn**

Thank you for your interest, and please stay tuned for future updates.

---

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@INPROCEEDINGS{11247276,
  author={Song, Peili and Song, Dezhen and Yang, Yifan and Lan, Enfan and Liu, Jingtai},
  booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Simulating Automotive Radar with Lidar and Camera Inputs}, 
  year={2025},
  volume={},
  number={},
  pages={11112-11119},
  keywords={Point cloud compression;Laser radar;Neural networks;Radar detection;Radar;Radar imaging;Millimeter wave radar;Doppler radar;Research and development;Automotive engineering},
  doi={10.1109/IROS60139.2025.11247276}
}