import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np

# Paths to your two directories
csv_dir_1 = 'output_data/fuse_concat/'
csv_dir_2 = 'output_data/fuse_pca/'

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

    for col in ['PSNR depth', 'PSNR RGB']:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], 100)

    return df

# Load both datasets
df1 = load_and_prepare(csv_dir_1)
df2 = load_and_prepare(csv_dir_2)

# Function to plot one metric
def plot_rd_curve(metric_col, title):
    plt.figure(figsize=(10, 7))

    for method, color in method_colors.items():
        # First dataset (solid)
        df1_method = df1[df1['method'] == method].sort_values(by='rate')
        if not df1_method.empty:
            plt.plot(df1_method['rate'], df1_method[metric_col],
                     label=method, color=color, marker='o', linestyle='-', alpha=0.9)

        # Second dataset (dashed)
        df2_method = df2[df2['method'] == method].sort_values(by='rate')
        if not df2_method.empty:
            plt.plot(df2_method['rate'], df2_method[metric_col],
                     label=f"{method} (pca)", color=color, marker='x', linestyle='--', alpha=0.6)

    plt.title(title)
    plt.xlabel('Rate (bpp)')
    plt.ylabel(f'{metric_col} [dB]')
    plt.ylim(10, 75)
    plt.grid(True, which='both')
    # plt.legend()s
    plt.tight_layout()
    plt.show()

# Plot both metrics
plot_rd_curve('PSNR depth', 'Rate-PSNR Curve (Depth)')
plot_rd_curve('PSNR RGB', 'Rate-PSNR Curve (RGB)')
