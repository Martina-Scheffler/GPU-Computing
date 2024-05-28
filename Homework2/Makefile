CC = c++
LIB_HOME=.
LIBS=-L${LIB_HOME}/lib64
OPT = -std=c++14 -O2

NVCC = nvcc
NVCC_FLAGS = -lcublas
NVCC_LIBS = 
TARGETDIR := bin
SRCDIR := src
INCDIR = include

INCLUDE_HEADERS := $(INCDIR)/matrix_generation.h
INCLUDE_CPP := $(INCDIR)/matrix_generation.cpp


all:  $(TARGETDIR)/transpose.o $(TARGETDIR)/matrix_generation.o $(TARGETDIR)/transpose

$(TARGETDIR)/transpose.o: $(SRCDIR)/transpose.cu
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS) 

$(TARGETDIR)/matrix_generation.o: $(INCLUDE_CPP) $(INCLUDE_HEADERS)
	mkdir -p $(@D)
	$(CC) -o $@ -c $< $(LIBS) $(OPT)

$(TARGETDIR)/transpose:
	$(NVCC) -o $@ $(TARGETDIR)/transpose.o $(TARGETDIR)/matrix_generation.o -lcudart -lcuda  -lcublas


clean:
	$(TARGETDIR)/*
