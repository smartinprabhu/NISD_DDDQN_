# Elevating UAV Multi Object Tracking Capabilities

This repository contains the code and resources for the research paper titled "Elevating UAV Multi Object Tracking Capabilities: Synergistic Integration of YOLOv8, Nano Instance Segmentation, and Dueling Double Deep Q Network."

## Overview

This research work proposes a robust methodology for enhancing Unmanned Aerial Vehicle (UAV) multi-object tracking capabilities. The integration of YOLOv8, Nano instance segmentation, and Dueling Double Deep Q Network (DDDQN) addresses challenges in object detection, size variations, and unexpected movements.

## Methodology

### YOLOv8
[Link to YOLOv8 Repository](https://github.com/AlexeyAB/darknet)

The YOLOv8 model is employed for real-time object detection, providing a foundation for the multi-object tracking system.

### Nano Instance Segmentation
[Link to Nano Instance Segmentation Repository]

Nano instance segmentation enhances the precision of object delineation, contributing to more accurate tracking results.

### Dueling Double Deep Q Network (DDDQN)
[Link to DDDQN Repository]

The DDDQN algorithm is utilized for reinforcement learning, improving the tracking agent's decision-making in dynamic environments.

## Experiments and Results

### Datasets
- UAVDT (Unmanned Aerial Vehicle Detection and Tracking)
- VisDrone

### Hyperparameters and Performance

The research explores different hyperparameters, including epsilon and discount factors, showcasing their impact on loss, accuracy, reward, and episode lengths.

| Dataset   | Epsilon | Discount Factor | Loss  | Accuracy | Reward | Episodes |
|-----------|---------|------------------|-------|----------|--------|----------|
| UAVDT     | 0.1     | 0.9              | 0.001 | 90%      | 95     | 100      |
| UAVDT     | 0.1     | 0.99             | 0.002 | 85%      | 85     | 150      |
| VisDrone  | 0.1     | 0.999            | 0.003 | 85%      | 90     | 200      |

### Conclusion
The proposed methodology showcases superior performance in UAV multi-object tracking, addressing challenges in various scenarios, such as disaster response, agriculture, and traffic monitoring.

## Citation

If you find this work useful, please cite our paper:

@article{your_paper_reference,
title={Elevating UAV Multi Object Tracking Capabilities: Synergistic Integration of YOLOv8, Nano Instance Segmentation, and Dueling Double Deep Q Network},
author={Your Name and Co-Authors},
journal={Journal Name},
year={Year},
publisher={Publisher}
}

## Getting Started

Follow the instructions below to set up and run the code on your local machine.

### Prerequisites

Make sure you have the following prerequisites installed on your machine:

- Python (>=3.6)
- [List any other dependencies, libraries, or tools required]

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/NISD.git
   cd NISD
2. **Install dependencies:**
   pip install -r requirements.txt
   
## Configuration :

1.Configure YOLOv8 :

Download the YOLOv8 repository from here.

Follow the instructions provided in the YOLOv8 repository to set up and configure YOLOv8 for your specific use case.

Place the YOLOv8 files in the yolov8 directory within this repository.

2. Configure Nano Instance Segmentation:

[Provide instructions or a link to the Nano Instance Segmentation repository]

Follow the repository's instructions to configure and set up Nano Instance Segmentation within the nano_instance_segmentation directory.

3. Configure DDDQN:

[Provide instructions or a link to the DDDQN repository]

Follow the repository's instructions to configure and set up DDDQN within the dddqn directory.


## Dataset Preparation
Download Datasets:

Download the UAVDT and VisDrone datasets from their respective sources.

Ensure the datasets are organized in the following structure:


├── datasets
│   ├── UAVDT
│   │   ├── Folders
│   │       ├── Images
│   │   
│   └── VisDrone
│       ├── images
│       └── ...
└── ...

## Running Experiments
1. Training,Testing:

 ```bash 
python train.py --config config/training_config.yaml
python test.py --config config/testing_config.yaml
