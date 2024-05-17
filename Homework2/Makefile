CC = c++
LIB_HOME=.
LIBS=-L${LIB_HOME}/lib64
OPT = -std=c++14

NVCC = nvcc
NVCC_FLAGS = 
NVCC_LIBS = 
TARGETDIR := bin
SRCDIR := src
INCDIR = include

INCLUDE_HEADERS := $(INCDIR)/matrix_generation.h
INCLUDE_CPP := $(INCDIR)/matrix_generation.cpp


all: $(TARGETDIR)/simple_transpose.o $(TARGETDIR)/matrix_generation.o $(TARGETDIR)/simple_transpose

$(TARGETDIR)/simple_transpose.o: $(SRCDIR)/simple_transpose.cu
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS) 

$(TARGETDIR)/matrix_generation.o: $(INCLUDE_CPP) $(INCLUDE_HEADERS)
	mkdir -p $(@D)
	$(CC) $^ -o $@ $(LIBS) $(OPT)

$(TARGETDIR)/simple_transpose:
	$(NVCC) -o $@ $(TARGETDIR)/simple_transpose.o $(TARGETDIR)/matrix_generation.o -lcudart -lcuda 


clean:
	$(TARGETDIR)/*
