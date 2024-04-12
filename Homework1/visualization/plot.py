import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df_simple = pd.read_csv("output/simple_transpose.csv", header=None, sep=';')

# compute mean
df_simple['mean'] = df_simple.mean(axis=1)

plt.plot(range(1, 13), df_simple['mean'][:])
plt.xlabel("Matrix Dimension: $2^{N} \\times 2^{N}$")
plt.ylabel('Execution Time [ms]')
plt.show()
