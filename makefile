#ds hardcoded for readability
all: gemm_serial gemm_threaded gemm_omp gemm_blas gemm_simd gemm_mpi

gemm_serial: gemm_serial.cpp
	g++ -O3 -Wall -std=c++11 $< -o $@
	
gemm_threaded: gemm_threaded.cpp
	g++ -O3 -Wall -std=c++11 $< -o $@ -pthread -Wl,--no-as-needed
	
gemm_omp: gemm_omp.cpp
	g++ -O3 -Wall -std=c++11 $< -o $@ -fopenmp
	
gemm_blas: gemm_blas.cpp
	g++ -O3 -Wall -std=c++11 $< -o $@ -lf77blas #ETHZ: -lblas -llapack
	
gemm_simd: gemm_simd.cpp
	g++ -O3 -Wall -std=c++11 -ffast-math -ftree-vectorize -ftree-vectorizer-verbose=0 $< -o $@
	
gemm_mpi: gemm_mpi.cpp
	mpic++ -O3 -Wall -std=c++11 $< -o $@
	
.PHONY clean:
	rm -f gemm_serial gemm_threaded gemm_omp gemm_blas gemm_simd gemm_mpi

