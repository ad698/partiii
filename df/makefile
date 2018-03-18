
# Makefile for df

#------------------------------------------------------
include ../Makefile.inc
#------------------------------------------------------
#------------------------------------------------------
# General compiler & options
SRC = src/
POT = pot/
OBJ = obj/
#------------------------------------------------------
# Set paths to other code

GENERALPATH = ../general/
CTPATH = $(GENERALPATH)coordtransforms/
POTPATH = ../pot/
AAPATH = ../aa/

GNUPLOTLIBPATH = $(GENERALPATH)gnuplot
NUMRECLIBPATH = $(GENERALPATH)jamestools/numrec/
JAMESLIBPATH = $(GENERALPATH)jamestools/jamestools/
POTLIBPATH = ../pot/
CUBALIBPATH = -L$(GENERALPATH)cuba
CUBA = $(CUBALIBPATH) -lcuba
AAINCPATH = $(AAPATH)inc/
AALIBPATH = $(AAPATH)lib/

ifeq ($(TORUS), 1)
	TORUSINCPATH = -I$(TORUSPATH)src/ -I$(TORUSPATH)src/utils/ -I$(TORUSPATH)src/pot
	TORUSLIBPATH = -L$(TORUSPATH)obj/ -L$(TORUSPATH)WDlib/obj/ -L$(EBFPATH)lib/
	TORUSLIBS = -lTorus -lOther -lPot -lWD -lebf_cpp
else
	TORUSINCPATH =
	TORUSLIBPATH =
	TORUSLIBS =
endif

ifeq ($(LAPACK), 1)
	LAPACKPATH=$(LAPACKPATH)
	LAPACKLIBS=-llapack -lblas
else
	LAPACKLIBPATH=
	LAPACKLIBS=
endif

#------------------------------------------------------
# Set include and library paths

INC = -I../../ -I../ -Iinc/ $(TORUSINCPATH) -I$(GENERALPATH) -I$(GSLPATH)include/ -I$(POTPATH)inc/ -I$(CTPATH)inc/ -I$(AAPATH)inc/

LIBPATHS =

LIB = -Wl,-rpath,$(TORUSPATH)obj/ -Wl,-rpath,$(TORUSPATH)WDlib/obj/  -Wl,-rpath,$(EBFPATH)lib/ -L$(GENERALPATH) -L$(NUMRECLIBPATH) -L$(JAMESLIBPATH) \
	$(CUBALIBPATH)\
	  -L$(AALIBPATH) -L$(GNUPLOTLIBPATH) $(TORUSLIBPATH) -L$(GSLPATH)lib/ -L$(POTLIBPATH)\
	  -L$(CTPATH) $(TORUSLIBS) -laa -lpot_js -lcoords \
	  -lpress_cp -lplot -lgsl -lgslcblas -lm $(EXTRACLIB) $(LAPACKLIBPATH) -ljamestools $(LAPACKLIBS) $(TORUSFLAG) $(LAPACKFLAG) $(LIBPATHS)

#------------------------------------------------------
# Set files

TESTS = test_df.exe

SELFCON = build_self_consistent.exe build_multi_self_consistent.exe

OBJECTS = $(OBJ)df.o $(OBJ)moments.o $(OBJ)self_consistent.o

HEADERS = $(wildcard inc/*.h)

#------------------------------------------------------

default: clean all

all:  $(SELFCON)

forceall: clean all

#------------------------------------------------------

$(OBJ)%.o : $(SRC)%.cpp ${HEADERS}
	$(CCOMPILER) $(OPTS) -c $< -o $@ $(INC)

#------------------------------------------------------

## SELF CONSISTENT

build_self_consistent.exe: $(OBJECTS) $(OBJ)build_self_consistent.o
	$(CCOMPILER) $(OPTS) -o build_self_consistent.exe $(OBJECTS) $(OBJ)build_self_consistent.o $(LIB) $(CUBA)

build_multi_self_consistent.exe: $(OBJECTS) $(OBJ)multicomponent_self_consistent.o
	$(CCOMPILER) $(OPTS) -o build_multi_self_consistent.exe $(OBJECTS) $(OBJ)multicomponent_self_consistent.o $(LIB) $(CUBA)

veldisp_self_consistent.exe: $(OBJECTS) $(OBJ)veldisp_self_consistent.o
	$(CCOMPILER) $(OPTS) -o veldisp_self_consistent.exe $(OBJECTS) $(OBJ)veldisp_self_consistent.o $(LIB) $(LIBPY) $(CUBA)

.PHONY: clean
clean:
	rm -rf obj/*.o; rm -rf *.so; rm -rf *.exe
