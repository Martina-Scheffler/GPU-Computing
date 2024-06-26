import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams.update({'font.size': 26})

df_simple_00 = pd.read_csv("output/simple_transpose_00.csv", header=None, sep=';')
df_simple_01 = pd.read_csv("output/simple_transpose_01.csv", header=None, sep=';')
df_simple_02 = pd.read_csv("output/simple_transpose_02.csv", header=None, sep=';')
df_simple_03 = pd.read_csv("output/simple_transpose_03.csv", header=None, sep=';')

df_block_00 = pd.read_csv("output/block_transpose_00.csv", header=None, sep=';')
df_block_01 = pd.read_csv("output/block_transpose_01.csv", header=None, sep=';')
df_block_02 = pd.read_csv("output/block_transpose_02.csv", header=None, sep=';')
df_block_03 = pd.read_csv("output/block_transpose_03.csv", header=None, sep=';')

# compute mean
df_simple_00['mean'] = df_simple_00.mean(axis=1)
df_simple_01['mean'] = df_simple_01.mean(axis=1)
df_simple_02['mean'] = df_simple_02.mean(axis=1)
df_simple_03['mean'] = df_simple_03.mean(axis=1)

df_block_00['mean'] = df_block_00.mean(axis=1)
df_block_01['mean'] = df_block_01.mean(axis=1)
df_block_02['mean'] = df_block_02.mean(axis=1)
df_block_03['mean'] = df_block_03.mean(axis=1)

plt.figure(figsize=(20, 10))

plt.plot(range(1, 13), df_simple_00['mean'], color='tab:blue', linestyle='--', linewidth=4, label='Simple Transpose -O0')
plt.plot(range(1, 13), df_simple_01['mean'], color='tab:red', linestyle='--', linewidth=4, label='Simple Transpose -O1')
plt.plot(range(1, 13), df_simple_02['mean'], color='tab:green', linestyle='--', linewidth=4, label='Simple Transpose -O2')
plt.plot(range(1, 13), df_simple_03['mean'], color='tab:orange', linestyle='--', linewidth=4, label='Simple Transpose -O3')

plt.plot(range(1, 13), df_block_00['mean'], color='tab:blue', linestyle='-', linewidth=4, label='Block Transpose -O0')
plt.plot(range(1, 13), df_block_01['mean'], color='tab:red', linestyle='-', linewidth=4, label='Block Transpose -O1')
plt.plot(range(1, 13), df_block_02['mean'], color='tab:green', linestyle='-', linewidth=4, label='Block Transpose -O2')
plt.plot(range(1, 13), df_block_03['mean'], color='tab:orange', linestyle='-', linewidth=4, label='Block Transpose -O3')

plt.xlabel("Matrix Dimension: $2^{N} \\times 2^{N}$")
plt.ylabel('Execution Time [ms]')
plt.legend()
plt.savefig('./visualization/execution_time_comparison.png', dpi=600)
plt.show()
