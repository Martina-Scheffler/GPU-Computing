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

e.g. copy to a computer where Matplotlib is installed:
```bash
scp name.surname@marzola.disi.unitn.it:~/GPU-Computing/Homework1/output/* .
```
Generate the plot:
```bash
python3 visualization/plot.py
```

### Analyze cache behavior
```bash
sbatch cache_analysis_sbatch.sh
```
Then use either cg_annotate or kcachegrind to see details:
```bash
cg_annotate --show-percs=yes simple_transpose.out
```
```bash
cg_annotate --show-percs=yes block_transpose.out
```

```bash
kcachegrind simple_transpose.out
```
```bash
kcachegrind block_transpose.out
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