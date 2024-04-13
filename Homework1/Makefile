CC = c++
LIB_HOME=.
LIBS=-L${LIB_HOME}/lib64

OPT = -std=c++14 -g $(USER_COMPILE_FLAGS)

TARGETDIR := bin
SRCDIR := src
INCDIR = include

INCLUDE_HEADERS := $(INCDIR)/matrix_generation.h
INCLUDE_CPP := $(INCDIR)/matrix_generation.cpp


all: $(TARGETDIR)/simple_transpose $(TARGETDIR)/block_transpose 


$(TARGETDIR)/simple_transpose: $(SRCDIR)/simple_transpose.cpp $(INCLUDE_HEADERS)
	mkdir -p $(@D)
	$(CC) $^ -o $@ $(LIBS) $(OPT) $(INCLUDE_CPP)

$(TARGETDIR)/block_transpose: $(SRCDIR)/block_transpose.cpp $(INCLUDE_HEADERS)
	mkdir -p $(@D)
	$(CC) $^ -o $@ $(LIBS) $(OPT) $(INCLUDE_CPP)


clean:
	$(TARGETDIR)/*
