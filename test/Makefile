
CC = g++
CPPFLAGS = -I.. -g
GSL_LIBS = -lgsl -lgslcblas
ARMADILLO_LIBS = -llapack
LIBS = $(GSL_LIBS) $(ARMADILLO_LIBS)

linear.model: linear.model.test.cpp
	$(CC) $(CPPFLAGS) linear.model.test.cpp $(LIBS)
