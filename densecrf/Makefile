# update the path variables 

CC	= g++
#CFLAGS	= -W -Wall -O3
CFLAGS	= -W -Wall -g

INCLUDE	= -I/include

all:
	make clean
	make prog_refine_pascal

clean:
	rm -f *.a
	rm -f *.o
	rm -f prog_test_densecrf

libDenseCRF.a: libDenseCRF/bipartitedensecrf.cpp libDenseCRF/densecrf.cpp libDenseCRF/filter.cpp libDenseCRF/permutohedral.cpp libDenseCRF/util.cpp libDenseCRF/densecrf.h libDenseCRF/fastmath.h libDenseCRF/permutohedral.h libDenseCRF/sse_defs.h libDenseCRF/util.h
	$(CC) libDenseCRF/bipartitedensecrf.cpp libDenseCRF/densecrf.cpp libDenseCRF/filter.cpp libDenseCRF/permutohedral.cpp libDenseCRF/util.cpp -c $(CFLAGS)
	ar rcs libDenseCRF.a bipartitedensecrf.o densecrf.o filter.o permutohedral.o util.o

prog_test_densecrf: test_densecrf/simple_dense_inference.cpp libDenseCRF.a
	$(CC) test_densecrf/simple_dense_inference.cpp -o prog_test_densecrf $(CFLAGS) -L. -lDenseCRF

prog_refine_pascal: refine_pascal/dense_inference.cpp refine_pascal/dense_inference.h libDenseCRF.a
	$(CC) refine_pascal/dense_inference.cpp -o prog_refine_pascal $(CFLAGS) -L. -lDenseCRF -I./refine_pascal/

