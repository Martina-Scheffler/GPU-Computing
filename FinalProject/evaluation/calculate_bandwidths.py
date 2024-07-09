#!/usr/bin/python3
import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np
plt.rcParams.update({'font.size': 26})

# read in data from files
def read_values(path):
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        milliseconds = float(reader.__next__()[0])
        rows = int(reader.__next__()[0])
        columns = int(reader.__next__()[0])
        nnz = int(reader.__next__()[0])
        buffer_size = int(reader.__next__()[0])  # Bytes
        
    return milliseconds, rows, columns
    

def effective_bandwidth(time, rows, columns):
    time *= 1e-3  # ms to s
    bytes = rows * columns * 4 * 2
    return ((bytes) / 1e9) / time
    


if __name__ == '__main__':
    ms, rows, columns = read_values('output/csr_cusparse_1.csv')
    print(effective_bandwidth(ms, rows, columns))
    