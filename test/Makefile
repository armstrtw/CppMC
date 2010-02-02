
CC = g++
CPPFLAGS = -I.. -g
ARMADILLO_LIBS = -llapack
LIBS = $(ARMADILLO_LIBS)

linear.model: linear.model.test.cpp
	$(CC) $(CPPFLAGS) linear.model.test.cpp $(LIBS)
