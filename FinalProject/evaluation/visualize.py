#!/usr/bin/python3
import matplotlib.pyplot as plt
import csv
plt.rcParams.update({'font.size': 26})

# import data from csv files
def import_csv(path):
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        milliseconds = float(reader.__next__()[0])
        rows = int(reader.__next__()[0])
        columns = int(reader.__next__()[0])
        nnz = int(reader.__next__()[0])
        try:
            buffer_size = int(reader.__next__()[0])  # Bytes
        except:
            buffer_size = 0
        
    return milliseconds, rows, columns, nnz, buffer_size
    

def effective_bandwidth(time, bytes):
    time *= 1e-3  # ms to s
    return ((bytes) / 1e9) / time
    

def run():
    execution_times_cusparse = []
    nnz_array = []
    buffer_array = []
    row_array = []
    column_array = []

    for i in range(1, 11):
        ms, rows, columns, nnz, bf = import_csv(f'./output/csr_cusparse_{i}.csv')
        execution_times_cusparse.append(ms)
        nnz_array.append(nnz)
        buffer_array.append(bf)
        row_array.append(rows)
        column_array.append(columns)
    
    execution_times_coo = []
    for i in range(1, 11):
        ms, rows, columns, nnz, _ = import_csv(f'./output/csr_coo_own_{i}.csv')
        execution_times_coo.append(ms)
    
    execution_times_csc = []
    for i in range(1, 11):
        ms, rows, columns, nnz, _ = import_csv(f'./output/csr_csc_own_{i}.csv')
        execution_times_csc.append(ms)
    
    plt.figure(figsize=(10, 10))
    plt.xlabel("Test Matrices")
    plt.ylabel("Execution Times [ms]")
    plt.scatter(range(1, 11), execution_times_cusparse, marker='o', linewidths=5, label="cuSPARSE")
    plt.scatter(range(1, 11), execution_times_coo, marker='*', linewidths=5, label="CSR-COO-CSR")
    plt.scatter(range(1, 11), execution_times_csc, marker='^', linewidths=5, label="CSR-CSC")
    plt.legend()
    plt.grid()
    plt.savefig('./evaluation/execution_times.png', dpi=600)
    plt.show()
    
    eb_cusparse = []
    eb_coo = []
    eb_csc = []

    for i in range(0, 10):
        bytes = row_array[i] * column_array[i] * 4 * 2
        eb_cusparse.append(effective_bandwidth(execution_times_cusparse[i], bytes))
        eb_coo.append(effective_bandwidth(execution_times_coo[i], bytes))
        eb_csc.append(effective_bandwidth(execution_times_csc[i], bytes))
    
    plt.figure(figsize=(10, 10))
    plt.xlabel("Test Matrices")
    plt.ylabel("Effective Bandwidth [GB/s]")
    plt.scatter(range(1, 11), eb_cusparse, marker='o', linewidths=5, label="cuSPARSE")
    plt.scatter(range(1, 11), eb_coo, marker='*', linewidths=5, label="CSR-COO-CSR")
    plt.scatter(range(1, 11), eb_csc, marker='^', linewidths=5, label="CSR-CSC")
    #plt.plot(range(1, 11), [933]*10, label="Max. Bandwidth")
    plt.legend()
    plt.grid()
    plt.savefig('./evaluation/effective_bandwidths.png', dpi=600)
    plt.show()
    
    # print max. effective bandwidths and percentage of max. GPU bandwidth
    print(f'cuSPARSE - Max. EBW: {max(eb_cusparse)}, %: {max(eb_cusparse) / 933 * 100}')
    print(f'CSR-COO-CSR - Max. EBW: {max(eb_coo)}, %: {max(eb_coo) / 933 * 100}')
    print(f'CSR-CSC - Max. EBW: {max(eb_csc)}, %: {max(eb_csc) / 933 * 100}')



if __name__ == '__main__':
    run()
    