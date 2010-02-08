
CC = g++
CPPFLAGS = -I.. -Wall -g
ARMADILLO_LIBS = -llapack
LIBS = $(ARMADILLO_LIBS)

all: linear.model.test linear.model.benchmark linear.model.normal.coefs.test varying.coefs.test

clean:
	rm linear.model.test linear.model.benchmark linear.model.normal.coefs.test

linear.model.test: linear.model.test.cpp
	$(CC) $(CPPFLAGS) $(LIBS) linear.model.test.cpp -o linear.model.test

linear.model.benchmark: linear.model.benchmark.cpp
	$(CC) $(CPPFLAGS) $(LIBS) linear.model.benchmark.cpp -o linear.model.benchmark

linear.model.normal.coefs.test: linear.model.normal.coefs.test.cpp
	$(CC) $(CPPFLAGS) $(LIBS) linear.model.normal.coefs.test.cpp -o linear.model.normal.coefs.test

varying.coefs.test: varying.coefs.test.cpp
	$(CC) $(CPPFLAGS) $(LIBS) varying.coefs.test.cpp -o varying.coefs.test
