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
Python3 visualization/plot.py
```

### Analyze cache behavior
e.g. for a (2^12 x 2^12) matrix:
```bash
valgrind --tool=cachegrind --cache-sim=yes --cachegrind-out-file=valgrind/simple_transpose.out ./bin/simple_transpose 12
```
```bash
valgrind --tool=cachegrind --cache-sim=yes --cachegrind-out-file=valgrind/block_transpose.out ./bin/block_transpose 12
```
