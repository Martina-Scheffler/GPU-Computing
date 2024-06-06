#!/usr/bin/python3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.rcParams.update({'font.size': 26})


def print_bandwidths(df):
    # calculate bandwidth
    tile_dimensions = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    block_rows = [1, 2, 4, 8, 16, 32]

    size = 2
    td_id = 0
    
    max_bandwidths = {np.log2(size): 0}
    min_execution = {np.log2(size): np.inf}

    for row in range(df.shape[0]): # next matrix dimension after a newline
        if np.isnan(df.iat[row, 0]):
            size *= 2
            if (size <= 4096):  # max. size is 4096
                max_bandwidths[np.log2(size)] = 0
                min_execution[np.log2(size)] = np.inf
                
            td_id = 0  # start anew with TD
            
        else:
            print(f'Size: {size}')
            
            col = 0
            while col < df.shape[1] and not np.isnan(df.iat[row, col]):  # go through existing BR
                # calculate effective bandwidth
                effective_bandwidth = ((size * size * 2 * 4) / 10**9) / (df.iat[row, col] * 10**(-3))
                
                # find max. effective bandwidth
                if(effective_bandwidth > max_bandwidths[np.log2(size)]):
                    max_bandwidths[np.log2(size)] = effective_bandwidth
                
                # find min. execution time
                if(df.iat[row, col] < min_execution[np.log2(size)]):
                    min_execution[np.log2(size)] = df.iat[row, col]
                
                # display effective bandwidth
                print(f'TD: {tile_dimensions[td_id]}, BR: {block_rows[col]}, EB: {effective_bandwidth}')
                col += 1
                
            print()
            td_id += 1
    
    print(f'Max. EBW: {max_bandwidths}')
    print(f'Min. ET: {min_execution}')
    return max_bandwidths, min_execution


# read in data from files
df_simple = pd.read_csv("output/analyze_bandwidth_0.csv", header=None, sep=';', names=range(11), skip_blank_lines=False)
df_coalesced = pd.read_csv("output/analyze_bandwidth_1.csv", header=None, sep=';', names=range(11), skip_blank_lines=False)
df_diagonal = pd.read_csv("output/analyze_bandwidth_2.csv", header=None, sep=';', names=range(11), skip_blank_lines=False)

# calculate max. bandwidths and min. execution times
max_simple, min_simple = print_bandwidths(df=df_simple)
max_coalesced, min_coalesced = print_bandwidths(df=df_coalesced)
max_diagonal, min_diagonal = print_bandwidths(df=df_diagonal)

# plot max. bandwidths
plt.figure(figsize=(20, 10))
plt.plot(max_simple.keys(), max_simple.values(), color='tab:blue', linewidth=4, label='Simple')
plt.plot(max_coalesced.keys(), max_coalesced.values(), color='tab:green', linewidth=4, label='Coalesced')
plt.plot(max_diagonal.keys(), max_diagonal.values(), color='tab:orange', linestyle='--', linewidth=4, label='Diagonal')
plt.plot(max_simple.keys(), len(max_simple.values()) * [933], color='tab:red', linewidth=4, label='max. BW')

plt.grid(True, 'both')
plt.xlabel("Matrix Dimension: $2^{N} \\times 2^{N}$")
plt.ylabel('Effective Bandwidth [GB/s]')
plt.legend()

plt.savefig('./visualization/effective_bandwidths.png', dpi=600)
plt.show()


# plot min. execution time from CPU vs. GPU
cpu_block_df = pd.read_csv("../Homework1/output/block_transpose_03.csv", header=None, sep=';')
cpu_block_df['mean'] = cpu_block_df.mean(axis=1)

cpu_simple_df = pd.read_csv("../Homework1/output/simple_transpose_03.csv", header=None, sep=';')
cpu_simple_df['mean'] = cpu_simple_df.mean(axis=1)


plt.figure(figsize=(20, 10))

plt.plot(range(1, 13), cpu_simple_df['mean'], color='tab:green', linestyle='-', linewidth=4, label='CPU Simple')
plt.plot(range(1, 13), cpu_block_df['mean'], color='tab:orange', linestyle='-', linewidth=4, label='CPU Block')

plt.plot(min_simple.keys(), min_simple.values(), color='tab:blue', linestyle='-', linewidth=4, label='GPU Simple')
plt.plot(min_coalesced.keys(), min_coalesced.values(), color='tab:red', linestyle='--', linewidth=4, label='GPU Coalesced')
plt.plot(min_diagonal.keys(), min_diagonal.values(), color='tab:purple', linestyle=':', linewidth=4, label='GPU Diagonal')

plt.xlabel("Matrix Dimension: $2^{N} \\times 2^{N}$")
plt.ylabel('Execution Time [ms]')

plt.grid(True, 'both')
plt.legend()

plt.savefig('./visualization/comparison_gpu_vs_cpu.png', dpi=600)
plt.show()


# Print execution times for a 2^12 x 2^12 matrix (data used in the paper)
print(cpu_simple_df['mean'].values[11])
print(cpu_block_df['mean'].values[11])
print(min_simple[12.0])
print(min_coalesced[12.0])
print(min_diagonal[12.0])