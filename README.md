# GPU Computing
## Cluster
### Access
```bash
ssh name.surname@marzola.disi.unitn.it
```
Example:
```bash
ssh martina.scheffler@marzola.disi.unitn.it
```

### Copying
```bash
scp path_to_local_file name.surname@marzola.disi.unitn.it:destination_path
```
Example:
```bash
scp Homework1/* martina.scheffler@marzola.disi.unitn.it:~/Homework1
```

## C++
### Makefile
```bash
make
```

### No Makefile
```bash
g++ -o simple_transpose simple_transpose.cpp
```

## Installing valgrind
```bash
sudo apt-get install valgrind
```

## Sbatch
### Run File
```bash
sbatch matrix_transpose_sbatch.sh
```

# Homework 1
## Clone
```bash
git clone https://github.com/Martina-Scheffler/GPU-Computing.git
```
```bash
cd GPU-Computing/Homework1/
```
```bash
./compile_flag_analysis.sh
```
