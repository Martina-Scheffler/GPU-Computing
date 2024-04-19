
import matplotlib.pyplot as plt
import pandas as pd

df_block_00_2 = pd.read_csv("output/block_transpose_00_2.csv", header=None, sep=';')
df_block_00_4 = pd.read_csv("output/block_transpose_00_4.csv", header=None, sep=';')
df_block_00_8 = pd.read_csv("output/block_transpose_00_8.csv", header=None, sep=';')
df_block_00_16 = pd.read_csv("output/block_transpose_00_16.csv", header=None, sep=';')
df_block_00_32 = pd.read_csv("output/block_transpose_00_32.csv", header=None, sep=';')
df_block_00_64 = pd.read_csv("output/block_transpose_00_64.csv", header=None, sep=';')
df_block_00_128 = pd.read_csv("output/block_transpose_00_128.csv", header=None, sep=';')
df_block_00_256 = pd.read_csv("output/block_transpose_00_256.csv", header=None, sep=';')
df_block_00_512 = pd.read_csv("output/block_transpose_00_512.csv", header=None, sep=';')
df_block_00_1024 = pd.read_csv("output/block_transpose_00_1024.csv", header=None, sep=';')
df_block_00_2048 = pd.read_csv("output/block_transpose_00_2048.csv", header=None, sep=';')

df_block_01_2 = pd.read_csv("output/block_transpose_01_2.csv", header=None, sep=';')
df_block_01_4 = pd.read_csv("output/block_transpose_01_4.csv", header=None, sep=';')
df_block_01_8 = pd.read_csv("output/block_transpose_01_8.csv", header=None, sep=';')
df_block_01_16 = pd.read_csv("output/block_transpose_01_16.csv", header=None, sep=';')
df_block_01_32 = pd.read_csv("output/block_transpose_01_32.csv", header=None, sep=';')
df_block_01_64 = pd.read_csv("output/block_transpose_01_64.csv", header=None, sep=';')
df_block_01_128 = pd.read_csv("output/block_transpose_01_128.csv", header=None, sep=';')
df_block_01_256 = pd.read_csv("output/block_transpose_01_256.csv", header=None, sep=';')
df_block_01_512 = pd.read_csv("output/block_transpose_01_512.csv", header=None, sep=';')
df_block_01_1024 = pd.read_csv("output/block_transpose_01_1024.csv", header=None, sep=';')
df_block_01_2048 = pd.read_csv("output/block_transpose_01_2048.csv", header=None, sep=';')

df_block_02_2 = pd.read_csv("output/block_transpose_02_2.csv", header=None, sep=';')
df_block_02_4 = pd.read_csv("output/block_transpose_02_4.csv", header=None, sep=';')
df_block_02_8 = pd.read_csv("output/block_transpose_02_8.csv", header=None, sep=';')
df_block_02_16 = pd.read_csv("output/block_transpose_02_16.csv", header=None, sep=';')
df_block_02_32 = pd.read_csv("output/block_transpose_02_32.csv", header=None, sep=';')
df_block_02_64 = pd.read_csv("output/block_transpose_02_64.csv", header=None, sep=';')
df_block_02_128 = pd.read_csv("output/block_transpose_02_128.csv", header=None, sep=';')
df_block_02_256 = pd.read_csv("output/block_transpose_02_256.csv", header=None, sep=';')
df_block_02_512 = pd.read_csv("output/block_transpose_02_512.csv", header=None, sep=';')
df_block_02_1024 = pd.read_csv("output/block_transpose_02_1024.csv", header=None, sep=';')
df_block_02_2048 = pd.read_csv("output/block_transpose_02_2048.csv", header=None, sep=';')

df_block_03_2 = pd.read_csv("output/block_transpose_03_2.csv", header=None, sep=';')
df_block_03_4 = pd.read_csv("output/block_transpose_03_4.csv", header=None, sep=';')
df_block_03_8 = pd.read_csv("output/block_transpose_03_8.csv", header=None, sep=';')
df_block_03_16 = pd.read_csv("output/block_transpose_03_16.csv", header=None, sep=';')
df_block_03_32 = pd.read_csv("output/block_transpose_03_32.csv", header=None, sep=';')
df_block_03_64 = pd.read_csv("output/block_transpose_03_64.csv", header=None, sep=';')
df_block_03_128 = pd.read_csv("output/block_transpose_03_128.csv", header=None, sep=';')
df_block_03_256 = pd.read_csv("output/block_transpose_03_256.csv", header=None, sep=';')
df_block_03_512 = pd.read_csv("output/block_transpose_03_512.csv", header=None, sep=';')
df_block_03_1024 = pd.read_csv("output/block_transpose_03_1024.csv", header=None, sep=';')
df_block_03_2048 = pd.read_csv("output/block_transpose_03_2048.csv", header=None, sep=';')

# compute mean
df_block_00_2['mean'] = df_block_00_2.mean(axis=1)
df_block_00_4['mean'] = df_block_00_4.mean(axis=1)
df_block_00_8['mean'] = df_block_00_8.mean(axis=1)
df_block_00_16['mean'] = df_block_00_16.mean(axis=1)
df_block_00_32['mean'] = df_block_00_32.mean(axis=1)
df_block_00_64['mean'] = df_block_00_64.mean(axis=1)
df_block_00_128['mean'] = df_block_00_128.mean(axis=1)
df_block_00_256['mean'] = df_block_00_256.mean(axis=1)
df_block_00_512['mean'] = df_block_00_512.mean(axis=1)
df_block_00_1024['mean'] = df_block_00_1024.mean(axis=1)
df_block_00_2048['mean'] = df_block_00_2048.mean(axis=1)

df_block_01_2['mean'] = df_block_01_2.mean(axis=1)
df_block_01_4['mean'] = df_block_01_4.mean(axis=1)
df_block_01_8['mean'] = df_block_01_8.mean(axis=1)
df_block_01_16['mean'] = df_block_01_16.mean(axis=1)
df_block_01_32['mean'] = df_block_01_32.mean(axis=1)
df_block_01_64['mean'] = df_block_01_64.mean(axis=1)
df_block_01_128['mean'] = df_block_01_128.mean(axis=1)
df_block_01_256['mean'] = df_block_01_256.mean(axis=1)
df_block_01_512['mean'] = df_block_01_512.mean(axis=1)
df_block_01_1024['mean'] = df_block_01_1024.mean(axis=1)
df_block_01_2048['mean'] = df_block_01_2048.mean(axis=1)

df_block_02_2['mean'] = df_block_02_2.mean(axis=1)
df_block_02_4['mean'] = df_block_02_4.mean(axis=1)
df_block_02_8['mean'] = df_block_02_8.mean(axis=1)
df_block_02_16['mean'] = df_block_02_16.mean(axis=1)
df_block_02_32['mean'] = df_block_02_32.mean(axis=1)
df_block_02_64['mean'] = df_block_02_64.mean(axis=1)
df_block_02_128['mean'] = df_block_02_128.mean(axis=1)
df_block_02_256['mean'] = df_block_02_256.mean(axis=1)
df_block_02_512['mean'] = df_block_02_512.mean(axis=1)
df_block_02_1024['mean'] = df_block_02_1024.mean(axis=1)
df_block_02_2048['mean'] = df_block_02_2048.mean(axis=1)

df_block_03_2['mean'] = df_block_03_2.mean(axis=1)
df_block_03_4['mean'] = df_block_03_4.mean(axis=1)
df_block_03_8['mean'] = df_block_03_8.mean(axis=1)
df_block_03_16['mean'] = df_block_03_16.mean(axis=1)
df_block_03_32['mean'] = df_block_03_32.mean(axis=1)
df_block_03_64['mean'] = df_block_03_64.mean(axis=1)
df_block_03_128['mean'] = df_block_03_128.mean(axis=1)
df_block_03_256['mean'] = df_block_03_256.mean(axis=1)
df_block_03_512['mean'] = df_block_03_512.mean(axis=1)
df_block_03_1024['mean'] = df_block_03_1024.mean(axis=1)
df_block_03_2048['mean'] = df_block_03_2048.mean(axis=1)

for i in range(12):
	x = range(1, 12)
	y = [df_block_00_2['mean'][i],
		df_block_00_4['mean'][i],
		df_block_00_8['mean'][i],
		df_block_00_16['mean'][i],
		df_block_00_32['mean'][i],
		df_block_00_64['mean'][i],
		df_block_00_128['mean'][i],
		df_block_00_256['mean'][i],
		df_block_00_512['mean'][i],
		df_block_00_1024['mean'][i],
		df_block_00_2048['mean'][i],
		]
	z = [df_block_01_2['mean'][i],
		df_block_01_4['mean'][i],
		df_block_01_8['mean'][i],
		df_block_01_16['mean'][i],
		df_block_01_32['mean'][i],
		df_block_01_64['mean'][i],
		df_block_01_128['mean'][i],
		df_block_01_256['mean'][i],
		df_block_01_512['mean'][i],
		df_block_01_1024['mean'][i],
		df_block_01_2048['mean'][i],
		]

	a = [df_block_02_2['mean'][i],
		df_block_02_4['mean'][i],
		df_block_02_8['mean'][i],
		df_block_02_16['mean'][i],
		df_block_02_32['mean'][i],
		df_block_02_64['mean'][i],
		df_block_02_128['mean'][i],
		df_block_02_256['mean'][i],
		df_block_02_512['mean'][i],
		df_block_02_1024['mean'][i],
		df_block_02_2048['mean'][i],
		]
 
	b = [df_block_03_2['mean'][i],
		df_block_03_4['mean'][i],
		df_block_03_8['mean'][i],
		df_block_03_16['mean'][i],
		df_block_03_32['mean'][i],
		df_block_03_64['mean'][i],
		df_block_03_128['mean'][i],
		df_block_03_256['mean'][i],
		df_block_03_512['mean'][i],
		df_block_03_1024['mean'][i],
		df_block_03_2048['mean'][i],
		]
 
	plt.figure()
	plt.plot(x, y, label='-O0')
	plt.plot(x, z, label='-O1')
	plt.plot(x, a, label='-O2')
	plt.plot(x, b, label='-O3')
	plt.xlabel(f'Block Size: $2^N \\times 2^N$')
	plt.ylabel('Execution Time [ms]')
	plt.title(f'Matrix Dimension: (2^{i+1})^2')
	plt.legend()
	plt.savefig(f'./visualization/block_size_comparison_{i+1}.png')
