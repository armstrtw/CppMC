
CC = g++
CPPFLAGS = -I.. -Wall -g
##CPPFLAGS = -I.. -Wall -O2
ARMADILLO_LIBS = -llapack
##ARMADILLO_LIBS = -lgoto2 -lpthread -lgfortran
LIBS = $(ARMADILLO_LIBS)

all: linear.model.test linear.model.benchmark linear.model.normal.coefs.test varying.coefs.test markov.switching.model

clean:
	rm -f linear.model.test linear.model.benchmark linear.model.normal.coefs.test varying.coefs.test markov.switching.model

linear.model.test: linear.model.test.cpp
	$(CC) $(CPPFLAGS) $(LIBS) linear.model.test.cpp -o linear.model.test

linear.model.benchmark: linear.model.benchmark.cpp
	$(CC) $(CPPFLAGS) $(LIBS) linear.model.benchmark.cpp -o linear.model.benchmark

linear.model.normal.coefs.test: linear.model.normal.coefs.test.cpp
	$(CC) $(CPPFLAGS) $(LIBS) linear.model.normal.coefs.test.cpp -o linear.model.normal.coefs.test

varying.coefs.test: varying.coefs.test.cpp
	$(CC) $(CPPFLAGS) $(LIBS) varying.coefs.test.cpp -o varying.coefs.test

markov.switching.model: markov.switching.model.cpp
	$(CC) $(CPPFLAGS) $(LIBS) markov.switching.model.cpp -o markov.switching.model
