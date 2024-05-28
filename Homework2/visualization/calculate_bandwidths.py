import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.rcParams.update({'font.size': 26})


def print_bandwidths(df):
    # calculate bandwidth
    tile_dimensions = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    block_rows = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    size = 2
    td_id = 0
    skipped_already = False
    max_bandwidth = 0

    for row in range(df.shape[0]):
        if not skipped_already and np.isnan(df.iat[row, 0]):
            size *= 2
            td_id = 0
            skipped_already = True
        elif not np.isnan(df.iat[row, 0]):
            skipped_already = False
            print(f'Size: {size}')
            col = 0
            while col < df.shape[1] and not np.isnan(df.iat[row, col]):
                # formula here
                effective_bandwidth = ((size * size * 2 * 4) / 10**9) / (df.iat[row, col] * 10**(-3))
                if(effective_bandwidth > max_bandwidth):
                    max_bandwidth = effective_bandwidth
                print(f'TD: {tile_dimensions[td_id]}, BR: {block_rows[col]}, EB: {effective_bandwidth}')
                col += 1
            print()
            td_id += 1
    
    print(f'Max. EBW: {max_bandwidth}')
    return max_bandwidth

# read in data from files
df_simple = pd.read_csv("output/analyze_bandwidth_0.csv", header=None, sep=';', names=range(11), skip_blank_lines=False)
df_coalesced = pd.read_csv("output/analyze_bandwidth_1.csv", header=None, sep=';', names=range(11), skip_blank_lines=False)
df_diagonal = pd.read_csv("output/analyze_bandwidth_2.csv", header=None, sep=';', names=range(11), skip_blank_lines=False)


max_simple = print_bandwidths(df=df_simple)
max_coalesced = print_bandwidths(df=df_coalesced)
max_diagonal = print_bandwidths(df=df_diagonal)



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
