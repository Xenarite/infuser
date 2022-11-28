build:
# g++ -std=c++20 -funroll-loops -fopenmp -march=native -Ofast -mavx2 ./src/main.cpp -o ./bin/infuser
	g++ -std=c++20 -DNOMPI -funroll-loops -fopenmp -march=native -Ofast -mavx2 ./src/main.cpp -o ./bin/infuser

buildcuda:
	nvcc -w --std=c++17 -use_fast_math -Xcompiler -fopenmp -Xcompiler -march=skylake -allow-unsupported-compiler -Xcompiler -mavx2 -gencode arch=compute_80,code=sm_80 ./src/gpu.cu -o ./bin/superfuser
mpi:
	mpicxx -std=c++20 -g -fopenmp -march=native -O3 -mavx2 ./src/main.cpp -o ./bin/infuser-mpi

mpicuda:
	clear
	# nvcc -w --std=c++17 -use_fast_math -Xcompiler -march=skylake -allow-unsupported-compiler -Xcompiler -mavx2 -gencode arch=compute_80,code=sm_80 -c -O3  -arch sm_80  -I/usr/lib/openmpi/include -I/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/comm_libs/openmpi/openmpi-3.1.5/include  ./src/infuser.cu
	# nvcc -lm  -lcudart -lcublas -L/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/comm_libs/openmpi/openmpi-3.1.5/lib -lmpi_cxx -lmpi -lopen-rte -lopen-pal -ldl -lnsl -lutil -lm infuser.o -o infuser
	nvcc -lcudart --std=c++17 -c -O3 src/infuser.cu -o infuser_cuda.o -gencode arch=compute_80,code=sm_80
	# nvcc -O3 infuser_cuda.o -o dlink.o -gencode arch=compute_80,code=sm_80 -dlink
	mpicxx -std=c++20 -fopenmp -L/opt/cuda/lib64/ -lcuda -lcudart -g -fopenmp -DENABLEGPU -march=native -O3 -mavx2 ./src/main.cpp infuser_cuda.o -o ./bin/infuser-mpi
