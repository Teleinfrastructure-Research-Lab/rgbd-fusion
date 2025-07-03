import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np

csv_dir = 'output_data/colorize/'

# Load all CSVs
all_files = glob.glob(os.path.join(csv_dir, "*.csv"))
df_list = [pd.read_csv(f) for f in all_files]
df = pd.concat(df_list, ignore_index=True)

# Replace inf values in PSNR column with 100
df['PSNR'] = df['PSNR'].replace([np.inf, -np.inf], 100)

# Define method colors
method_colors = {
    'intel_hue': 'red',
    'bytesplit': 'blue',
    'quantize': 'green',
    'psk': 'orange',
    'range_demux': 'magenta',
}

plt.figure(figsize=(10, 7))

for method, color in method_colors.items():
    method_df = df[df['method'] == method].sort_values(by='rate')
    if not method_df.empty:
        plt.plot(method_df['rate'], method_df['PSNR'], label=method, color=color, marker='o', alpha=0.9)

plt.title('Rate-PSNR Curve by Method')
plt.xlabel('Rate (bpp)')
plt.ylabel('PSNR [dB]')
plt.ylim(15, 75)  # Set y-axis range
plt.grid(True, which='both')
# plt.legend()
plt.tight_layout()
plt.show()
