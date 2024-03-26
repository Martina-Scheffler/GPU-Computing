CC = c++
LIB_HOME=.
LIBS=-L${LIB_HOME}/lib64
INCLUDE = -Isrc
OPT = -std=c++14

BUILDDIR := obj
TARGETDIR := bin

all: $(TARGETDIR)/simple_transpose $(TARGETDIR)/square_transpose $(TARGETDIR)/block_transpose

$(TARGETDIR)/simple_transpose: simple_transpose.cpp $(OBJECTS)
	mkdir -p $(@D)
	$(CC) $^ -o $@ $(INCLUDE) $(LIBS) $(OPT)

$(TARGETDIR)/square_transpose: square_transpose.cpp $(OBJECTS)
	mkdir -p $(@D)
	$(CC) $^ -o $@ $(INCLUDE) $(LIBS) $(OPT)

$(TARGETDIR)/block_transpose: block_transpose.cpp $(OBJECTS)
	mkdir -p $(@D)
	$(CC) $^ -o $@ $(INCLUDE) $(LIBS) $(OPT)


clean:
	rm $(BUILDDIR)/*.o $(TARGETDIR)/*
