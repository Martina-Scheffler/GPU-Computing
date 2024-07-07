CC = c++
LIB_HOME=.
LIBS=-L${LIB_HOME}/lib64
OPT = -std=c++14 -O2

NVCC = nvcc
NVCC_FLAGS =
NVCC_LIBS = 
TARGETDIR := bin
SRCDIR := src
INCDIR = include

INCLUDE_HEADERS := $(INCDIR)/mmio.h
INCLUDE_CPP := $(INCDIR)/mmio.c


all: $(TARGETDIR)/transpose.o $(TARGETDIR)/import_sparse.o

$(TARGETDIR)/transpose.o: $(SRCDIR)/transpose.cu
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS) 

$(TARGETDIR)/import_sparse.o: $(INCDIR)/import_sparse_matrix.cpp $(INCLUDE_HEADERS)
	mkdir -p $(@D)
	$(CC) $^ -o $@ $(LIBS) $(OPT) $(INCLUDE_CPP)

clean:
	$(TARGETDIR)/*
