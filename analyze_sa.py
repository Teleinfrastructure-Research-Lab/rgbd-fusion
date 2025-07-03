import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np

# Define directories
csv_dir_1 = 'output_data/yolo_fuse_concat/'
csv_dir_2 = 'output_data/yolo_fuse_pca/'

# Define method colors
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

# Load datasets
df1 = load_and_prepare(csv_dir_1)
df2 = load_and_prepare(csv_dir_2)

# General plotting function
def plot_rd_curve(metric_col, metric_sa_col, rate_col, rate_sa_col, title):
    plt.figure(figsize=(10, 7))

    for method, color in method_colors.items():
        # First dataset: solid line (normal PSNR) + solid line w/ stars (SA PSNR)
        df1_method = df1[df1['method'] == method].sort_values(by=rate_col)
        if not df1_method.empty:
            # pass
            plt.plot(df1_method[rate_col], df1_method[metric_col],
                     label=method, color=color, marker='o', linestyle='-', alpha=0.9)

            if metric_sa_col in df1_method.columns and rate_sa_col in df1_method.columns:
                plt.plot(df1_method[rate_sa_col], df1_method[metric_sa_col],
                         label=f"{method} SA", color=color, marker='*', linestyle='-', markersize=10, alpha=0.9)

        # Second dataset: dashed line (normal PSNR) + dashed line w/ stars (SA PSNR)
        df2_method = df2[df2['method'] == method].sort_values(by=rate_col)
        if not df2_method.empty:
            pass
            # plt.plot(df2_method[rate_col], df2_method[metric_col],
            #          label=f"{method} (pca)", color=color, marker='x', linestyle='--', alpha=0.6)

            # if metric_sa_col in df2_method.columns and rate_sa_col in df2_method.columns:
            #     plt.plot(df2_method[rate_sa_col], df2_method[metric_sa_col],
            #              label=f"{method} SA (pca)", color=color, marker='*', linestyle='--', markersize=10, alpha=0.6)

    plt.title(title)
    plt.xlabel('Rate (bpp)')
    plt.ylabel(f'{metric_col} [dB]')
    plt.ylim(10, 75)
    plt.grid(True, which='both')
    # plt.legend()
    plt.tight_layout()
    plt.show()

# Plot for depth
plot_rd_curve(
    metric_col='PSNR depth',
    metric_sa_col='PSNR SA depth',
    rate_col='rate',
    rate_sa_col='rate SA',
    title='Rate-PSNR Curve (Depth)'
)

# Plot for RGB
plot_rd_curve(
    metric_col='PSNR RGB',
    metric_sa_col='PSNR SA RGB',
    rate_col='rate',
    rate_sa_col='rate SA',
    title='Rate-PSNR Curve (RGB)'
)
