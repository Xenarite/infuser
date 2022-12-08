#pragma once
#include "common.h"
#include "graph.h"
#include "hyperfuser.h"

void* devcpy(void* host_t, size_t size);
void* get_dev(size_t size);
void fill_registers_dev(float* M, size_t n, size_t R, size_t J, size_t NH, size_t PROC_RANK, size_t PROC_SIZE);
void simulate_dev(float* M, graph_t g, size_t R, size_t J, int* X);
void harmonic_mean_dev(float* MG, float* M, float* mask, size_t n, int R, int J);
size_t get_max_dev(float* arr, size_t size);
void max_inplace(float* mask, float* registers, size_t size);
void hostcpy(void* host_ptr, void* dev_ptr, size_t size);
void devsync();
void maxsum_char_dev(float* estimates, char* hypers, char* mask, size_t N, int R);
void maxsum_float_dev(float* estimates, float* hypers, float* mask, size_t N, int R);

template<typename T>
bool checkequal(T* host_ptr, T* dev_ptr, size_t size) {
	auto host_dev_ptr = get_aligned<T>(size);
	hostcpy(host_dev_ptr.get(), dev_ptr, size * sizeof(T));
	return std::equal(host_ptr, host_ptr+size, host_dev_ptr.get());
}

template <typename graph_t>
std::vector<std::tuple<size_t, float>>
infuser_gpu(graph_t& g, int K, size_t R, size_t J, int* X, int NH = 4, bool harmonic = true) {
	int PROC_SIZE, PROC_RANK;
	MPI_Comm_size(MPI_COMM_WORLD, &PROC_SIZE);
	MPI_Comm_rank(MPI_COMM_WORLD, &PROC_RANK);

	float* _MG = (float*) get_dev(sizeof(float)*g.n);

	std::cerr << "RANK"<<PROC_RANK<<":"<<" local R set to " << R <<std::endl;
	std::cerr << "RANK"<<PROC_RANK<<":"<<" random offset is " << PROC_RANK * R <<std::endl;

	int* X_dev = (int*)devcpy(X, sizeof(int) * R);
	std::vector<std::tuple<size_t, float>> results;

	float* M = (float*) get_dev(sizeof(float) * g.n * R * J);
	float* mask = (float*)get_dev(sizeof(float) * R * J);
	float* MG = (float*)get_dev(sizeof(float) * g.n);

	fill_registers_dev(M, g.n, R, J, NH, PROC_RANK, PROC_SIZE);

	auto _g = g;
	_g.xadj = (decltype(_g.xadj))devcpy((void*)g.xadj, sizeof(g.xadj[0]) * (g.n + 1));
	_g.adj = (decltype(_g.adj))devcpy((void*)g.adj, sizeof(g.adj[0]) * (g.m));

	Timer t;
	simulate_dev(M, _g, R, J, X_dev);

	while (results.size() < K) {
		if (harmonic)
			harmonic_mean_dev(MG, M, mask, g.n, R, J);
		else
			maxsum_float_dev(MG, M, mask, g.n, R*J);
		devsync();
		MPI_Reduce(MG, _MG, g.n, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
		float val;
		size_t u;
		if (PROC_RANK == 0){
			u = get_max_dev(_MG, g.n);
		}
		MPI_Bcast(&u, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
		results.push_back(std::make_tuple(u, t.elapsed()));
		max_inplace(mask, M + u * R * J, R*J);

	}
	return results;
}