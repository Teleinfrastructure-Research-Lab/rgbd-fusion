# Semantic-Aware RGBD Compression

This repository provides the official implementation of the methods described in the paper:

**"Exploring Semantic-Aware Compression of RGBD Images Using Conventional Codecs"**  
Presented at the 60th International Scientific Conference on Information, Communication and Energy Systems and Technologies (ICEST 2025), Ohrid, North Macedonia.

## Overview

This project implements RGBD image compression using conventional codecs (e.g., JPEG2000), leveraging depth image colorization and semantic-aware filtering. It includes:

- Several depth colorization strategies (Quantization, Intel-Hue, PSK, Range-Demux, Bytesplit)
- Two fusion methods for RGB and depth (Concatenation and PCA-based fusion)
- Semantic-aware compression to enhance perceptual quality in human regions

>**Note:** The paper refers to *Kinect’s built-in segmentation* for semantic-aware compression. However, the provided code uses **YOLO-based person segmentation**, as the dataset used in this implementation did not include segmentation masks. The results should be equivalent in quality and effectiveness.

## Installation

Make sure you have Python 3.8 or newer. To install the dependencies, run:

```bash
cd hue-depth-encoding
pip install -e ".[dev]"
cd ..
pip install -e .
```
## YOLO Weights for Segmentation

To enable semantic-aware segmentation using YOLO, you need to download pre-trained YOLOv8 weights.

**Download YOLOv8 weights from:**  
[https://docs.ultralytics.com/models/yolov8/#models](https://docs.ultralytics.com/models/yolov8/#models)

Place the downloaded weights (e.g., `yolov8n-seg.engine`, `yolov8n-seg.pt`, etc.) in an appropriate directory, or modify the code to load them from your chosen path.

---

# Citation

```
@INPROCEEDINGS{bozhilov2025exploring,
  author={Bozhilov, Ivaylo B. and Petkova, Radostina R. and Tonchev, Krasimir T. and Manolova, Agata H.},
  booktitle={2025 60th International Scientific Conference on Information, Communication and Energy Systems and Technologies (ICEST)}, 
  title={Exploring Semantic-Aware Compression of RGBD Images Using Conventional Codecs}, 
  year={2025},
  volume={},
  number={},
  pages={1-4},
  abstract={RGBD data from 3D capture devices includes synchronized color and depth images, forming a four-channel format that traditional codecs (e.g., JPEG2000, H.264) are not designed to compress. Challenges include a lack of support for four-channel data, incompatibility with high bit-depth depth images, and synchronization overhead when transmitting separate streams. This paper investigates depth image colorization to make RGBD data compatible with standard codecs. We compare multiple colorization methods and propose two fusion strategies—simple spatial concatenation and PCA-based fusion—to combine RGB and depth data into a single image. Furthermore, we introduce a semantic-aware compression approach that leverages person segmentation from the Kinect to guide the encoding process. Our results show that incorporating semantic information significantly improves compression efficiency and rate-distortion performance.},
  keywords={Image segmentation;Image coding;Codecs;Three-dimensional displays;Semantics;Rate-distortion;Transform coding;Synchronization;Standards;Principal component analysis;RGBD;Compression;Semantic},
  doi={10.1109/ICEST66328.2025.11098421},
  ISSN={2603-3267},
  month={June},}
```