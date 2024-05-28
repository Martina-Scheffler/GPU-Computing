# GPU Computing
## Homework 1
### Clone
```bash
ssh name.surname@marzola.disi.unitn.it
```
```bash
git clone https://github.com/Martina-Scheffler/GPU-Computing.git
```
```bash
cd GPU-Computing/Homework1/
```

### Run code to measure execution times for different compile flags
```bash
sbatch ./compile_flag_analysis_sbatch.sh
```


### Generate a plot 
Prerequisites:
- Python3
- Pandas
- Matplotlib

e.g. copy to a computer where Matplotlib is installed (into the output/ folder):
```bash
scp name.surname@marzola.disi.unitn.it:~/GPU-Computing/Homework1/output/* .
```
Generate the plot (call from Homework1/ folder):
```bash
python3 visualization/plot_compile_flags.py
```

### Analyze cache behavior
```bash
sbatch cache_analysis_sbatch.sh
```
which executes
```bash 
valgrind --tool=cachegrind --cache-sim=yes --cachegrind-out-file=valgrind/simple_transpose.out ./bin/simple_transpose 12
```
```bash 
valgrind --tool=cachegrind --cache-sim=yes --cachegrind-out-file=valgrind/simple_transpose.out ./bin/block_transpose 12
```

Then use either cg_annotate or kcachegrind to see details:
```bash
cg_annotate --show-percs=yes valgrind/simple_transpose.out
```
```bash
cg_annotate --show-percs=yes valgrind/block_transpose.out
```

```bash
kcachegrind valgrind/simple_transpose.out
```
```bash
kcachegrind valgrind/block_transpose.out
```

### Run single execution of one of the transpose algorithms
Modify transpose_sbatch.sh to set:
- Compile flag
- Algorithm
- Matrix dimension

Run:
```bash
sbatch transpose_sbatch.sh
```
which executes the algorithm and prints the execution time to the .out file.


## Homework 2
### Clone
```bash
ssh name.surname@marzola.disi.unitn.it
```
```bash
git clone https://github.com/Martina-Scheffler/GPU-Computing.git
```
```bash
cd GPU-Computing/Homework2/
```

### Run single transpose
In `transpose_sbatch.sh`, change `srun ./bin/transpose <power> <strategy> <tile dimension> <block rows>` to fit your 
desired values.
For example `srun ./bin/transpose 2 <strategy> 4 1` will transpose a 2^2 x 2^2 matrix using tile dimension 4 and block rows 1.

For strategy, the options are:
- 0 = simple kernel
- 1 = coalesced kernel
- 2 = diagonal kernel

Run:
```batch
sbatch transpose_sbatch.sh
```

### Analyzing parameters for multiple matrix dimensions
Run:
```batch
sbatch analyze_bandwidth_sbatch.sh
```
Then, copy to a machine with Python installed
```batch
scp name.surname@marzola.disi.unitn.it:~/GPU-Computing/Homework2/output/* .
```
Run the python script in the `Homework2` folder
```batch
python3 visualization/calculate_bandwidths.py
```