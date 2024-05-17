CC = nvcc
LIB_HOME=.
LIBS=-L${LIB_HOME}/lib64


TARGETDIR := bin
SRCDIR := src
INCDIR = include

INCLUDE_HEADERS := $(INCDIR)/matrix_generation.h
INCLUDE_CPP := $(INCDIR)/matrix_generation.cpp


all: module load cuda; $(TARGETDIR)/simple_transpose $(TARGETDIR)/block_transpose 

$(TARGETDIR)/simple_transpose: $(SRCDIR)/simple_transpose.cu $(INCLUDE_HEADERS)
	mkdir -p $(@D)
	$(CC) -o $@ $(LIBS) $(OPT) $(INCLUDE_CPP)

clean:
	$(TARGETDIR)/*
