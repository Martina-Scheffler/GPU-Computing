# GPU Computing
## Cluster
### Access
```
$ ssh name.surname@marzola.disi.unitn.it
```
Example:
```
$ ssh martina.scheffler@marzola.disi.unitn.it
```

### Copying
```
$ scp path_to_local_file name.surname@marzola.disi.unitn.it:destination_path
```
Example:
```
$ scp Homework1/* martina.scheffler@marzola.disi.unitn.it:~/Homework1
```

## C++
### Makefile
```
$ make
```

### No Makefile
```
$ g++ -o simple_transpose simple_transpose.cpp
```

## Sbatch
### Run File
```
$ sbatch matrix_transpose_sbatch.sh
```