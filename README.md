# GPU Computing
## Final Project - Sparse Matrix Transposition
### Clone
```bash
ssh name.surname@marzola.disi.unitn.it
```
```bash
git clone https://github.com/Martina-Scheffler/GPU-Computing.git
```
```bash
cd GPU-Computing/FinalProject/
```

### Import Test Matrices (optional)
In order to load the test matrices in the programs, they were first converted from `.mtx` files to `.csv` files. 
During the process, also some wrong entries (value zero saved as non-zero) were removed and the number of non-zero elements adjusted.

This was done by commenting in the `main()` function in [`FinalProject/include/import_sparse_matrix.cpp`](FinalProject/include/import_sparse_matrix.cpp) and then executing:
```batch
g++ -o include/import include/import_sparse_matrix.cpp include/mmio.c 
```
followed by
```batch
./include/import 
```
The matrices are saved in the [`coo/`](FinalProject/test_matrices/coo/) and [`csr/`](FinalProject/test_matrices/csr/) folders with the structure:
1. rows
2. columns
3. number of non-zero elements
4. row offsets (CSR) or row indices (COO)
5. column indices
6. values of the non-zero elements


### Run
```batch
sbatch transpose_sparse_sbatch.sh
```
will run the command `srun ./bin/transpose <algorithm> <matrix>`, where the options for `<algorithm>` are:

- `0`: cuSPARSE CSR transpose by CSR-CSC conversion
- `1`: own kernel for CSR-CSC conversion
- `2`: own kernel for transposing a COO matrix
- `3`: own kernel for transposition by CSR-COO-T-CSR

and the `<matrix>` parameter is either an integer from `1` to `10` specifying a desired test matrix, or `all` to evaluate all of them in a loop. The matrices that can be used for testing and a description of them can be found in [`FinalProject/test_matrices/`](FinalProject/test_matrices/). 

After running, resulting transposed matrices are created in `test_matrices/tranposed/` and files for performance evaluation in `output/`.

### Test Correctness
Correctness of the tranposition algorithms can be tested using the Jupyter notebooks in [FinalProject/evaluation/tests/](FinalProject/evaluation/tests/).

For this, the desired matrices need to be copied to a computer where a Jupyter notebook can be run:
```bash
scp name.surname@marzola.disi.unitn.it:~/GPU-Computing/FinalProject/test_matrices/transposed/* .
```
and then the notebooks can be used to transpose the original matrix using `scipy.sparse` and checking for equality with the one obtained on the cluster.

- [test_coo_transpose](FinalProject/evaluation/tests/test_coo_transpose.ipynb) can be used to test correctness of algorithm `2` for COO transposition
- [test_csr_coo_convert](FinalProject/evaluation/tests/test_csr_coo_convert.ipynb) can be used to test conversion from CSR to COO format
- [test_csr_coo_csr_convert](FinalProject/evaluation/tests/test_csr_coo_csr_convert.ipynb) can be used to check a CSR matrix against itself after a CSR-COO-CSR conversion
- [test_csr_transpose](FinalProject/evaluation/tests/test_csr_transpose.ipynb) can be used to test correctness of CSR transposition, i.e. algorithms `0`, `1` and `3`

The correct output for all tests at the bottom should be `True`.


### Run All (CSR)
In order to run all transpose variants shown in the paper for the CSR matrices, run:
```bash
sbatch transpose_all_sbatch.sh
```
This will run algorithms `0`, `1` and `3` for `all` matrices.

### Evaluate
To copy the `output/` to a machine with Python to evaluate:
```bash
scp name.surname@marzola.disi.unitn.it:~/GPU-Computing/FinalProject/output/* .
```
Afterwards, run
```bash
python3 evaluation/visualize.py
```
from the `FinalProject/` folder. This generates plots for the execution times and effective bandwidths. Further, it prints the max. effective bandwidth reached on the data set and its percentage of the maximum bandwidth of the GPU.



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

e.g. copy to a computer where Matplotlib is installed (into the `output/` folder):
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
Then, copy to a machine with Python installed (into the `output/` folder)
```batch
scp name.surname@marzola.disi.unitn.it:~/GPU-Computing/Homework2/output/* .
```
Run the python script in the `Homework2` folder
```batch
python3 visualization/calculate_bandwidths.py
```
This will print all effective bandwidths for all configurations and generate two plots:
- A comparison of max. effective bandwidths between the kernels for different matrix sizes
- A comparison of execution times between CPU and GPU algorithms


