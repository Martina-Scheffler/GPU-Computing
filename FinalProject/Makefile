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

INCLUDE_HEADERS := $(INCDIR)/mmio.h $(INCDIR)/import_sparse_matrix.h
INCLUDE_CPP := $(INCDIR)/mmio.c $(INCDIR)/import_sparse_matrix.cpp


all: $(TARGETDIR)/transpose.o $(TARGETDIR)/import_sparse.o $(TARGETDIR)/transpose

$(TARGETDIR)/transpose.o: $(SRCDIR)/transpose.cu
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS) 

$(TARGETDIR)/import_sparse.o: $(INCLUDE_CPP) $(INCLUDE_HEADERS)
	mkdir -p $(@D)
	$(CC) -o $@ -c $< $(LIBS) $(OPT)

$(TARGETDIR)/transpose:
	$(NVCC) -o $@ $(TARGETDIR)/transpose.o $(TARGETDIR)/import_sparse.o -lcudart -lcuda -lcusparse

clean:
	$(TARGETDIR)/*
