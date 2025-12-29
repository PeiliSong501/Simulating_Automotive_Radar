This repository contains the source code for the IROS 2025 paper:

**Simulating Automotive Radar with LiDAR and Camera Inputs**

📄 **arXiv:** https://arxiv.org/abs/2503.08068  
📘 **IEEE Xplore:** https://ieeexplore.ieee.org/document/11247276  
> *Note: the IEEE Xplore version contains minor typos; please refer to the arXiv version for the most accurate text.*

The corresponding video is available at YouTube (https://www.youtube.com/watch?v=dybbkPePHD4&t=1s), and bilibili (https://www.bilibili.com/video/BV1AdKwzLEPW/?spm_id_from=333.1387.0.0&vd_source=eb4269e8fa059487e2a91ce411765468).

---

## Overview

This project focuses on simulating automotive radar signals (pitch, yaw, range, velocity, reflectivity) by leveraging multimodal sensor inputs, including **LiDAR** and **camera data**.  
The simulator is designed to model radar observations in complex driving environments and serves as a research tool for perception, sensor fusion, and learning-based radar understanding.

In addition to the network architectures and training pipelines, this codebase also provides a development toolkit for processing several **4D automotive radar datasets**, such as **VoD** and **MSC-Rad4R**.

---

## Code Status

⚠️ **Important Notice**

Since I am extremely busy these days, the current version of this repository:

- is **not well organized**
- contains **comments of poor quality**
- may have **suboptimal code readability**

The code is released **as-is** to support reproducibility of the paper results.

---

## Future Updates

A **cleaned and refactored version** of the codebase will be released in the future, including:

- clearer project structure
- detailed code comments
- usage instructions and examples
- configuration explanations

✉️ If you have urgent questions or require clarification for reproduction purposes, feel free to contact the author via email: **peilisong@mail.nankai.edu.cn**

Please stay tuned for updates.
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
  doi={10.1109/IROS60139.2025.11247276}}
