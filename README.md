# Software Simulation of a BLE/WiFi-based Indoor Tracking System

## Overview
This project implements a comprehensive software simulation of a BLE/WiFi-based indoor tracking system, built on an edge-cloud architecture. The simulation models key aspects of the system, including signal generation, location processing, resource management, and a user-facing visualization dashboard.


## Prerequisites

- Python 3.x


## Installation

### 1. Clone the repository:
```bash
git clone https://github.com/23CSE362-edge-computing-2025-26-odd/capstone-project-03_cloud9
```

### 2. Install dependencies:
```bash
pip install -r requirements.txt
```


## Dataset Acknowledgement

This project uses the [CampusRSSI dataset (PerCom25 Artifacts)](https://github.com/jinyi-yoon/CampusRSSI/tree/main), which provides real-world Wi-Fi RSSI measurements for indoor localization.

### Selected Environment
- Medium-Obs (Office): 16 APs over a 9.9 × 9.9 m² area, enclosed by concrete walls, glass partitions, and pillars.
- Data was collected using EFM-Networks ipTIME N104Q-i access points operating on IEEE 802.11n (2.4 GHz, 150 Mbps).
- RSSI measurements were taken every 30 cm using LG G Pad 3 10.1 (LG-X760) devices with a custom Android app.
- Pedestrian movement was not restricted to ensure realistic conditions.

### Citation
If you use the CampusRSSI dataset in your work, please cite the following:
```bibtex
@inproceedings{yoon2024ganloc,
  title={GAN-Loc: Empowering Indoor Localization for Unknown Areas via Generative Fingerprint Map},
  author={Yoon, JinYi and You, Yeawon and Kang, Dayeon and Kim, Jeewoon and Lee, HyungJune},
  booktitle={2024 21st Annual IEEE International Conference on Sensing, Communication, and Networking (SECON)},
  pages={282--290},
  year={2024},
  organization={IEEE}
}

@inproceedings{you2025collagemap,
  title={CollageMap: Tailoring Generative Fingerprint Map via Obstacle-Aware Adaptation for Site-Survey-Free Indoor Localization},
  author={You, Yeawon and Yoon, JinYi and Kang, Dayeon and Kim, Jeewoon and Lee, HyungJune},
  booktitle={2025 IEEE International Conference on Pervasive Computing and Communications (PerCom)},
  year={2025},
  organization={IEEE}
}
```
