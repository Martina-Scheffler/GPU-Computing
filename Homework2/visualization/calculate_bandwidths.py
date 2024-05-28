import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.rcParams.update({'font.size': 26})

# read in data from files
df_simple = pd.read_csv("output/analyze_bandwidth_0.csv", header=None, sep=';', names=range(11), skip_blank_lines=False)
df_coalesced = pd.read_csv("output/analyze_bandwidth_0.csv", header=None, sep=';', names=range(11))
df_diagonal = pd.read_csv("output/analyze_bandwidth_0.csv", header=None, sep=';', names=range(11))

# calculate bandwidth
tile_dimensions = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
block_rows = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

size = 2
td_id = 0

for row in range(df_simple.shape[0]):
    if np.isnan(df_simple.iat[row, 0]):
        size *= 2
        td_id = 0
    else:
        print(f'Size: {size}')
        col = 0
        while  col < df_simple.shape[1] and not np.isnan(df_simple.iat[row, col]):
            # formula here
            effective_bandwidth = ((size * size * 2 * 4) / 10**9) / (df_simple.iat[row, col] * 10**(-3))
            print(f'TD: {tile_dimensions[td_id]}, BR: {block_rows[col]}, EB: {effective_bandwidth}')
            col += 1
        print()
        td_id += 1




# plt.figure(figsize=(20, 10))

# plt.plot(range(1, 13), df_simple_00['mean'], color='tab:blue', linestyle='--', linewidth=4, label='Simple Transpose -O0')
# plt.plot(range(1, 13), df_simple_01['mean'], color='tab:red', linestyle='--', linewidth=4, label='Simple Transpose -O1')
# plt.plot(range(1, 13), df_simple_02['mean'], color='tab:green', linestyle='--', linewidth=4, label='Simple Transpose -O2')
# plt.plot(range(1, 13), df_simple_03['mean'], color='tab:orange', linestyle='--', linewidth=4, label='Simple Transpose -O3')

# plt.plot(range(1, 13), df_block_00['mean'], color='tab:blue', linestyle='-', linewidth=4, label='Block Transpose -O0')
# plt.plot(range(1, 13), df_block_01['mean'], color='tab:red', linestyle='-', linewidth=4, label='Block Transpose -O1')
# plt.plot(range(1, 13), df_block_02['mean'], color='tab:green', linestyle='-', linewidth=4, label='Block Transpose -O2')
# plt.plot(range(1, 13), df_block_03['mean'], color='tab:orange', linestyle='-', linewidth=4, label='Block Transpose -O3')

# plt.xlabel("Matrix Dimension: $2^{N} \\times 2^{N}$")
# plt.ylabel('Execution Time [ms]')
# plt.legend()
# plt.savefig('./visualization/execution_time_comparison.png', dpi=600)
# plt.show()
