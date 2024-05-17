CC = c++
LIB_HOME=.
LIBS=-L${LIB_HOME}/lib64
OPT = -std=c++1

NVCC = nvcc
NVCC_FLAGS = 
NVCC_LIBS = 

TARGETDIR := bin
SRCDIR := src
INCDIR = include

INCLUDE_HEADERS := $(INCDIR)/matrix_generation.h
INCLUDE_CPP := $(INCDIR)/matrix_generation.cpp


all: $(TARGETDIR)/simple_transpose 

$(TARGETDIR)/simple_transpose: $(TARGETDIR)/simple_transpose.o $(INCLUDE_HEADERS)
	mkdir -p $(@D)
	$(CC) -o $@ $(LIBS) $(OPT) $(INCLUDE_CPP) $^

$(TARGETDIR)/simple_transpose.o: $(SRCDIR)/simple_transpose.cu 
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS) 


clean:
	$(TARGETDIR)/*
