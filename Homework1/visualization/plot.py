import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df_simple = pd.read_csv("output/simple_transpose.csv", header=None, sep=';')
df_square = pd.read_csv("output/square_transpose.csv", header=None, sep=';')
df_block = pd.read_csv("output/block_transpose_32.csv", header=None, sep=';')

# compute mean
df_simple['mean'] = df_simple.mean(axis=1)
df_square['mean'] = df_square.mean(axis=1)
df_block['mean'] = df_block.mean(axis=1)

plt.plot(range(1, 13), df_simple['mean'], label='Simple Transpose')
plt.plot(range(1, 13), df_square['mean'], label='Square Transpose')
plt.plot(range(1, 13), df_block['mean'], label='Block Transpose')
plt.xlabel("Matrix Dimension: $2^{N} \\times 2^{N}$")
plt.ylabel('Execution Time [ms]')
plt.legend()
plt.show()
