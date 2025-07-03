import numpy as np
import pandas as pd
import hashlib
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.visualize import visualize_fusion
from Holotwin.Stream.FileReader import FileReader
from utils.utils import compress_with_jpeg, decompress_jpeg, calculate_psnr, depth_from_xyz
from methods.colorize import encode_depth_map, decode_depth_map

def short_dict_hash(d: dict, length: int = 8) -> str:
    dict_str = json.dumps(d, sort_keys=True)
    hash_obj = hashlib.sha256(dict_str.encode())
    return hash_obj.hexdigest()[:length]

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
columns = ['method', 'kwargs', 'PSNR', 'PSNR_RAW', 'rate', 'jpeg_quality_level']
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
            depth = depth_from_xyz(xyz_image, 0.001, 7.0, True)
            d_min, d_max = 0.0, 1.0
            if method =="psk":
                depth = ((depth.astype(np.float32))*(2**13)).astype(np.uint16)

            ###COLORIZE
            colorized = encode_depth_map(depth, method, **method_kwargs)
            ###

            jpeg_bytes = compress_with_jpeg(colorized, quality_level, "rates")
            bits = len(jpeg_bytes)*8
            colorized_decompressed = decompress_jpeg(jpeg_bytes)

            ###RESTORE
            restored_raw = decode_depth_map(colorized, method, **method_kwargs)
            restored = decode_depth_map(colorized_decompressed, method, **method_kwargs)
            if method =="psk":
                restored = restored/restored.max()
                restored = restored * (2**13)
            ###
            
            psnr_depth_raw = calculate_psnr(depth, restored_raw)
            psnr_depth_compressed = calculate_psnr(depth, restored)

            # print(f"PSNR Depth RAW: {psnr_depth_raw:.2f} dB")
            # print(f"PSNR Depth Compressed: {psnr_depth_compressed:.2f} dB")
            # print(f"bpp: {bits/(width*height)}")
            # visualize_fusion(
            #     img1=depth,
            #     img2=depth,
            #     combined=colorized_decompressed,
            #     rec1=restored_raw,
            #     rec2=restored
            # )

            data_point = {
                "method": method,
                "kwargs": str(method_kwargs),
                "PSNR": psnr_depth_compressed,
                "PSNR_RAW": psnr_depth_raw,
                "rate": bits/(width*height),
                'jpeg_quality_level': quality_level
            }
            data.append(data_point)

df = pd.DataFrame(data, columns=columns)


plt.figure(figsize=(8, 6))
plt.scatter(df['rate'], df['PSNR'], alpha=0.7)

plt.title(f'R-D for {method}')
plt.xlabel('bpp')
plt.ylabel('PSNR [db]')
plt.grid(True)
plt.tight_layout()
plt.show()


kwargs_hash = short_dict_hash(method_kwargs)
csv_name = f"output_data/colorize/{method}_{kwargs_hash}.csv"
df.to_csv(csv_name, index=False)