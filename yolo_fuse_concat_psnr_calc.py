import numpy as np
import pandas as pd
import hashlib
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from ultralytics import YOLO
from ultralytics.yolo.utils import ROOT, yaml_load
from ultralytics.yolo.utils.checks import check_yaml

from utils.visualize import visualize_fusion
from Holotwin.Stream.FileReader import FileReader
from utils.utils import compress_with_jpeg, decompress_jpeg, calculate_psnr_masked, calculate_psnr_masked_skimage, depth_from_xyz, masked_blur
from methods.colorize import encode_depth_map, decode_depth_map
from methods.fuse import concat, unconcat

def short_dict_hash(d: dict, length: int = 8) -> str:
    dict_str = json.dumps(d, sort_keys=True)
    hash_obj = hashlib.sha256(dict_str.encode())
    return hash_obj.hexdigest()[:length]

def process(input_image):
    original_shape = input_image.shape[:2]  # (height, width)

    # Resize for YOLO inference
    input_resized = cv2.resize(input_image, (640, 640))
    results = model(input_resized, device=0)
    result = results[0]

    # Get bounding boxes, class IDs, and confidence scores
    bb = np.array(result.boxes.xyxy.cpu(), dtype="int")
    cls = np.array(result.boxes.cls.cpu(), dtype="int")
    scores = np.array(result.boxes.conf.cpu(), dtype="float")

    # Create a combined mask (starts empty)
    combined_mask = np.zeros((640, 640), dtype=np.uint8)

    # Process masks if they exist
    if result.masks is not None:
        for i, mask in enumerate(result.masks.data):
            if cls[i] == 0 and scores[i] > 0.6:  # Class 0 = person
                binary_mask = mask.cpu().numpy()
                combined_mask = np.logical_or(combined_mask, binary_mask)

    # Convert to binary (0 or 1)
    combined_mask = combined_mask.astype(np.uint8)

    # Resize the combined mask back to original image size
    mask_resized = cv2.resize(combined_mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)

    return mask_resized


CLASSES = yaml_load(check_yaml('coco128.yaml'))['names']
model = YOLO('yolov8n-seg.engine')

method = "intel_hue"
method_kwargs = {"d_min": 0.0,
                 "d_max": 1.0}
# method_kwargs = {}
# method_kwargs = {"num_ch": 1}
# method_kwargs = {"intervals": [(0.0,0.31),(0.3,0.61),(0.6,1.0)]}
jpeg_quality_levels = [40, 38,36, 35,33,31, 30,28,26, 25,23,21, 20,18,16, 15,13,11, 10,8,7,6,5,4,3,2,1]

height = 424
width = 512
channels = 3

data_path = "data/validateCap2person.txt"
columns = ['method', 'kwargs', 'PSNR depth', 'PSNR RGB', 'PSNR SA depth', 'PSNR SA RGB', 'rate', 'rate SA', 'jpeg_quality_level']
data = []
start_sample = 625
max_samples = 1
fr_step = 1


for quality_level in tqdm(jpeg_quality_levels, total = len(jpeg_quality_levels)):
    sample_count = 0
    with FileReader(data_path) as fr:
        for i,frame in enumerate(fr):
            if i < start_sample:
                continue
            if sample_count>=max_samples:
                break
            if i%fr_step != 0:
                continue
            sample_count+=1

            xyz_image = frame.vertexAttributes[0].data.reshape(height, width, channels)
            xyz_image = np.nan_to_num(xyz_image, nan=0.0, posinf=1, neginf=0.0)
            rgb_image = frame.vertexAttributes[1].data.reshape(height, width, channels).copy()
            rgb_image = rgb_image[:, :, ::-1]
            mask = process(rgb_image)
            depth = depth_from_xyz(xyz_image, 0.001, 7.0, True)
            d_min, d_max = 0.0, 1.0
            if method =="psk":
                depth = ((depth.astype(np.float32))*(2**13)).astype(np.uint16)

            ###COLORIZE
            colorized = encode_depth_map(depth, method, **method_kwargs)
            ###
            ### FUSE
            packed = concat(rgb_image, colorized)
            packed_mask = np.concatenate([mask,mask], axis = 1)
            packed_blured = masked_blur(packed, packed_mask, (4,4))
            ###

            jpeg_bytes = compress_with_jpeg(packed, quality_level, "rates")
            bits = len(jpeg_bytes)*8
            packed_decompressed = decompress_jpeg(jpeg_bytes)

            jpeg_bytes_blured = compress_with_jpeg(packed_blured, quality_level, "rates")
            bits_blured = len(jpeg_bytes_blured)*8
            packed_blured_decompressed = decompress_jpeg(jpeg_bytes_blured)

            ### FUSE
            rgb_rec, colorized_decompressed = unconcat(packed_decompressed)
            rgb_blured_rec, colorized_blured_decompressed = unconcat(packed_blured_decompressed)
            ###
            ###RESTORE
            depth_rec = decode_depth_map(colorized_decompressed, method, **method_kwargs)
            depth_blured_rec = decode_depth_map(colorized_blured_decompressed, method, **method_kwargs)
            if method =="psk":
                depth_rec = depth_rec/depth_rec.max()
                depth_rec = depth_rec * (2**13)
                depth_blured_rec = depth_blured_rec/depth_blured_rec.max()
                depth_blured_rec = depth_blured_rec * (2**13)
            ###
            
            psnr_depth = calculate_psnr_masked(depth, depth_rec, mask)
            psnr_rgb = calculate_psnr_masked_skimage(rgb_image, rgb_rec, mask)

            psnr_sa_depth = calculate_psnr_masked(depth, depth_blured_rec, mask)
            psnr_sa_rgb = calculate_psnr_masked_skimage(rgb_image, rgb_blured_rec, mask)


            # visualize_fusion(
            #     img1=rgb_image,
            #     img2=depth,
            #     combined=packed_blured_decompressed,
            #     rec1=rgb_blured_rec,
            #     rec2=depth_blured_rec
            # )

            data_point = {
                "method": method,
                "kwargs": str(method_kwargs),
                "PSNR depth": psnr_depth,
                "PSNR RGB": psnr_rgb,
                "PSNR SA depth": psnr_sa_depth,
                "PSNR SA RGB": psnr_sa_rgb,
                "rate": bits/(width*height),
                "rate SA": bits_blured/(width*height),
                'jpeg_quality_level': quality_level
            }
            data.append(data_point)

df = pd.DataFrame(data, columns=columns)

kwargs_hash = short_dict_hash(method_kwargs)
csv_name = f"output_data/yolo_fuse_concat/{method}_{kwargs_hash}.csv"
df.to_csv(csv_name, index=False)