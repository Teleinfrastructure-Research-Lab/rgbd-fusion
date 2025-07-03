import pandas as pd
import glob
import os
import numpy as np
from scipy import interpolate, integrate

# csv_dir_path = "output_data/yolo_fuse_concat/"
# output_csv = "output_data/yolo_fuse_concat_bdpsnr.csv"

csv_dir_path = "output_data/yolo_fuse_pca/"
output_csv = "output_data/yolo_fuse_pca_bdpsnr.csv"

method_colors = {
    'intel_hue': 'red',
    'bytesplit': 'blue',
    'quantize': 'green',
    'psk': 'orange',
    'range_demux': 'magenta',
}

def load_and_prepare(csv_dir):
    """Load and clean CSVs from a directory"""
    all_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    df_list = [pd.read_csv(f) for f in all_files]
    df = pd.concat(df_list, ignore_index=True)

    # Replace infs with 100
    for col in ['PSNR depth', 'PSNR RGB', 'PSNR SA depth', 'PSNR SA RGB']:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], 100)

    return df

def bd_psnr(R1, PSNR1, R2, PSNR2):
    R1 = np.array(R1)
    R2 = np.array(R2)
    PSNR1 = np.array(PSNR1)
    PSNR2 = np.array(PSNR2)
    sorted_idx1 = np.argsort(R1)
    R1 = R1[sorted_idx1]
    PSNR1 = PSNR1[sorted_idx1]
    R1, unique_idx1 = np.unique(R1, return_index=True)
    PSNR1 = PSNR1[unique_idx1]
    sorted_idx2 = np.argsort(R2)
    R2 = R2[sorted_idx2]
    PSNR2 = PSNR2[sorted_idx2]
    R2, unique_idx2 = np.unique(R2, return_index=True)
    PSNR2 = PSNR2[unique_idx2]
    log_R1 = np.log(R1)
    log_R2 = np.log(R2)
    p1 = interpolate.PchipInterpolator(log_R1, PSNR1)
    p2 = interpolate.PchipInterpolator(log_R2, PSNR2)
    min_int = max(log_R1[0], log_R2[0])
    max_int = min(log_R1[-1], log_R2[-1])
    if min_int >= max_int:
        raise ValueError("No overlapping range between the two RD curves.")
    int1 = integrate.quad(p1, min_int, max_int)[0]
    int2 = integrate.quad(p2, min_int, max_int)[0]
    avg_diff = (int2 - int1) / (max_int - min_int)
    return avg_diff

columns = ['method', 'BD-PSNR Depth', 'BD-PSNR Color']
df = load_and_prepare(csv_dir_path)
result_df = pd.DataFrame(columns=columns)
for method in method_colors.keys():
    rates = df[df["method"] == method]["rate"].to_numpy()
    rates_sa = df[df["method"] == method]["rate SA"].to_numpy()
    psnr_depth = df[df["method"] == method]["PSNR depth"].to_numpy()
    psnr_color = df[df["method"] == method]["PSNR RGB"].to_numpy()
    psnr_sa_depth = df[df["method"] == method]["PSNR SA depth"].to_numpy()
    psnr_sa_color = df[df["method"] == method]["PSNR SA RGB"].to_numpy()

    bd_psnr_depth = bd_psnr(rates, psnr_depth, rates_sa, psnr_sa_depth)
    bd_psnr_color = bd_psnr(rates, psnr_color, rates_sa, psnr_sa_color)
    
    new_row = {"method": method,
               "BD-PSNR Depth": bd_psnr_depth,
               "BD-PSNR Color": bd_psnr_color}
    result_df = pd.concat([result_df, pd.DataFrame([new_row])], ignore_index=True)

result_df.to_csv(output_csv, index=False)
    

