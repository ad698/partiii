
# Makefile for new_struct

#------------------------------------------------------
HOST=$(shell hostname)
ifneq (,$(findstring .ast.cam.ac.uk,$(HOST)))
	CCOMPILER = /opt/ioa/software/gcc/4.7.2/bin/g++
	DEBUG = -ggdb3 #-ffast-math -Ofast
	OPTS = -std=c++0x -Wall -O2 -fopenmp -fPIC $(DEBUG)
	GSLPATH = /opt/ioa/software/gsl/1.16/
	BOOSTINCPATH =/opt/ioa/software/boost/1.55.0/include
	BOOSTLIBPATH =-L/opt/ioa/software/boost/1.55.0/lib
	PYTHONINCPATH=/opt/ioa/software/python/2.7.8/include/python2.7/
	PYTHONLIBPATH=-L/opt/ioa/software/python/2.7.8/lib/
else
	CCOMPILER = g++-4.9
	OPTS = -std=c++0x -Wall -O3 -Ofast -ffast-math -fopenmp -fPIC
	BOOSTINCPATH =.
	BOOSTLIBPATH =
	PYTHONINCPATH=/Users/jls/anaconda/include/python2.7/
	PYTHONLIBPATH=-L/Users/jls/anaconda/lib/
	EXTRACLIB = -lc++
endif

#------------------------------------------------------
# General compiler & options
SRC = src/
POT = pot/
OBJ = obj/
FALCONOBJ=$(OBJ)falcONobj/
#------------------------------------------------------
# Set paths to other code

GENERALPATH = ../../general/
TORUSPATH = ../../Torus/
NEMOPATH = /data/jls/nemo/
FALCONPATH = $(NEMOPATH)usr/dehnen/falcON.P/

GNUPLOTLIBPATH = $(GENERALPATH)gnuplot
NUMRECLIBPATH = $(GENERALPATH)jamestools/numrec/
JAMESLIBPATH = $(GENERALPATH)jamestools/jamestools/

TORUSINCPATH = -I$(TORUSPATH)src/ -I$(TORUSPATH)WDlib/inc/
TORUSLIBPATH = -L$(TORUSPATH)obj/ -L$(TORUSPATH)WDlib/obj/

LAPACKPATH = $(GENERALPATH)lapack/
CUBALIBPATH = -L$(GENERALPATH)cuba -L$(GENERALPATH)cubature
CUBA = $(CUBALIBPATH) -lcuba -lcubature_sj

#------------------------------------------------------
# Set include and library paths

INC = -I../../ -Iinc/ $(TORUSINCPATH) -I$(PYTHONINCPATH) -I$(BOOSTINCPATH) -I$(GENERALPATH) -I$(GSLPATH)include/

LIB = -L$(GENERALPATH) -L$(NUMRECLIBPATH) -L$(JAMESLIBPATH) -L$(GNUPLOTLIBPATH) $(TORUSLIBPATH) -L$(GSLPATH)lib/ \
-lpress_cp -lTorus -lOther -lPot -lWD -lplot -lgsl -lgslcblas -lm $(EXTRACLIB) -L$(LAPACKPATH) -llapack -lblas -ljamestools
BOOSTLIB = -lboost_python -lpython2.7
LIBPY = $(PYTHONLIBPATH) $(BOOSTLIBPATH) $(BOOSTLIB)

#------------------------------------------------------
# Additional stuff for falcON programs

FALCONINC = -I$(FALCONPATH)inc/ -I$(FALCONPATH)inc/public/ -I$(FALCONPATH)utils/inc/ -I$(NEMOPATH)inc/ -I$(NEMOPATH)inc/max/ -I$(NEMOPATH)lib/ -DfalcON_SINGLE -DfalcON_NEMO -DfalcON_PROPER

FALCONLIB = -L$(FALCONPATH)lib/ -lfalcON.P -L$(FALCONPATH)utils/lib/ -lWDutils -L$(NEMOPATH)lib/ -lnemo++ -lnemo -DfalcON_SINGLE -DfalcON_NEMO -DfalcON_PROPER

#------------------------------------------------------
# Set files

TESTS = test.exe test_df.exe genfunc_test.exe analytic_test.exe spherical_test.exe stackel_fit_test.exe multipole_test.exe

SELFCON = build_self_consistent.exe veldisp_self_consistent.exe

LIBRARY = triax_py.so actest_py.so streamy_py.so streamy_err_py.so pal5_streamy.so

OBJECTS = $(OBJ)debug.o $(OBJ)coordsys.o $(OBJ)coordtransforms.o $(OBJ)potential.o $(OBJ)orbit.o $(OBJ)spherical_aa.o $(OBJ)stackel_aa.o  $(OBJ)get_closed_Rz.o $(OBJ)uv_orb.o $(OBJ)lmn_orb.o $(OBJ)it_torus.o $(OBJ)stackel_fit.o $(OBJ)Multipole.o $(OBJ)analytic_aa.o $(OBJ)genfunc_aa.o $(OBJ)adiabatic_aa.o $(OBJ)tables_aa.o

#------------------------------------------------------

default: clean all

all:  $(SELFCON)

forceall:	clean all

#------------------------------------------------------

$(OBJ)%.o : $(SRC)%.cpp
	$(CCOMPILER) $(OPTS) -c $< -o $@ $(INC)

$(FALCONOBJ)%.o : $(SRC)%.cpp
	$(CCOMPILER) $(OPTS) -c $< -o $@ $(INC) $(FALCONINC)

#------------------------------------------------------

## TESTS
analytic_test.exe: $(OBJECTS)
	$(CCOMPILER) $(OPTS) -o analytic_test.exe $(OBJECTS) $(LIB) $(LIBPY) $(CUBA)

genfunc_test.exe: $(OBJECTS)
	$(CCOMPILER) $(OPTS) -o genfunc_test.exe $(OBJECTS) $(LIB) $(LIBPY) $(CUBA)

test.exe: $(OBJECTS) $(OBJ)test.o
	$(CCOMPILER) $(OPTS) -o test.exe $(OBJECTS) $(OBJ)test.o $(LIB) $(LIBPY) $(CUBA)

spherical_test.exe: $(OBJECTS)
	$(CCOMPILER) $(OPTS) -o spherical_test.exe $(OBJECTS) $(LIB) $(LIBPY) $(CUBA)

stackel_fit_test.exe: $(OBJECTS)
	$(CCOMPILER) $(OPTS) -o stackel_fit_test.exe $(OBJECTS) $(LIB) $(CUBA) $(LIBPY)

multipole_test.exe: $(OBJECTS)
	$(CCOMPILER) $(OPTS) -o multipole_test.exe $(OBJECTS) $(LIB) $(CUBA) $(LIBPY)


test_df.exe: $(OBJECTS) $(OBJ)test_df.o $(OBJ)df.o $(OBJ)moments.o
	$(CCOMPILER) $(OPTS) -o test_df.exe $(OBJECTS) $(OBJ)test_df.o $(OBJ)df.o $(OBJ)moments.o $(LIB) $(LIBPY) $(CUBA)

## LIBRARY

triax_py.so: $(OBJECTS) $(OBJ)triax_py.o
	$(CCOMPILER) -fPIC -shared $(CFL) $(OBJECTS) $(OBJ)triax_py.o $(INC) $(LIB) $(LIBPY) -o triax_py.so

actest_py.so: $(OBJECTS) $(OBJ)actest_py.o
	$(CCOMPILER) -fPIC -shared $(OPTS) $(OBJECTS) $(OBJ)actest_py.o $(INC) $(LIB) $(LIBPY) $(CUBA) -o actest_py.so

streamy_py.so: $(OBJECTS) $(OBJ)streamy.o
	$(CCOMPILER) -fPIC -shared $(OPTS) $(OBJECTS) $(OBJ)streamy.o $(INC) $(LIB) $(LIBPY) $(CUBA) -o streamy_py.so

streamy_err_py.so: $(OBJECTS) $(OBJ)streamy_errors.o
	$(CCOMPILER) -fPIC -shared $(OPTS) $(OBJECTS) $(OBJ)streamy_errors.o $(INC) $(LIB) $(LIBPY) $(CUBA) -o streamy_err_py.so

pal5_streamy.so: $(OBJECTS) $(OBJ)pal5_streamy.o
	$(CCOMPILER) -fPIC -shared $(OPTS) $(OBJECTS) $(OBJ)pal5_streamy.o $(INC) $(LIB) $(LIBPY) $(CUBA) -o pal5_streamy.so

## SELF CONSISTENT

build_self_consistent.exe: $(OBJECTS) $(OBJ)df.o $(OBJ)moments.o $(OBJ)self_consistent.o $(OBJ)build_self_consistent.o
	$(CCOMPILER) $(OPTS) -o build_self_consistent.exe $(OBJECTS) $(OBJ)df.o $(OBJ)moments.o $(OBJ)self_consistent.o $(OBJ)build_self_consistent.o $(LIB) $(LIBPY) $(CUBA)

veldisp_self_consistent.exe: $(OBJECTS) $(OBJ)df.o $(OBJ)moments.o $(OBJ)self_consistent.o $(OBJ)veldisp_self_consistent.o
	$(CCOMPILER) $(OPTS) -o veldisp_self_consistent.exe $(OBJECTS) $(OBJ)df.o $(OBJ)moments.o $(OBJ)self_consistent.o $(OBJ)veldisp_self_consistent.o $(LIB) $(LIBPY) $(CUBA)

## OTHER

falcON_aa.exe: $(OBJECTS) $(FALCONOBJ)falcON_aa.o
	$(CCOMPILER) $(OPTS) -o falcON_aa.exe $(OBJECTS) $(FALCONOBJ)falcON_aa.o $(FALCONLIB) $(LIB) $(LIBPY) $(CUBA)

vl2.exe: $(OBJECTS)
	$(CCOMPILER) $(OPTS) -o vl2.exe $(OBJECTS) $(LIB) $(CUBA) $(LIBPY)

denis.exe: $(OBJECTS) $(OBJ)denis.o
	$(CCOMPILER) $(OPTS) -o denis.exe $(OBJECTS) $(OBJ)denis.o $(LIB) $(CUBA) $(LIBPY)

gaps.exe: $(OBJECTS) $(OBJ)gaps.o
	$(CCOMPILER) $(OPTS) -o gaps.exe $(OBJECTS) $(OBJ)gaps.o $(LIB) $(CUBA) $(LIBPY)

.PHONY: clean
clean:
	rm -rf obj/*.o; rm -rf *.so; rm -rf *.exe
