CC = c++
LIB_HOME=.
LIBS=-L${LIB_HOME}/lib64

OPT = -std=c++14 -g $(USER_COMPILE_FLAGS)

TARGETDIR := bin
SRCDIR := src
INCDIR = include

INCLUDE_HEADERS := $(INCDIR)/mmio.h
INCLUDE_CPP := $(INCDIR)/mmio.c


all: $(TARGETDIR)/import_sparse

$(TARGETDIR)/import_sparse: $(INCDIR)/import_sparse_matrix.cpp $(INCLUDE_HEADERS)
	mkdir -p $(@D)
	$(CC) $^ -o $@ $(LIBS) $(OPT) $(INCLUDE_CPP)

clean:
	$(TARGETDIR)/*
