#pragma once
#include "common.h"
#include "graph.h"
#include "hyperfuser.h"

void* devcpy(void* host_t, size_t size);
void* get_dev(size_t size);
void fill_registers_dev(float* M, size_t n, size_t R, size_t J, size_t NH, size_t offset);
void simulate_dev(float* M, graph_t g, size_t R, size_t J, int* X);
void harmonic_mean_dev(float* MG, float* M, float* mask, size_t n, int R, int J);
size_t get_max_dev(float* arr, size_t size);
void max_inplace(float* mask, float* registers, size_t size);
void hostcpy(void* host_ptr, void* dev_ptr, size_t size);

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
infuser_gpu(graph_t& g, int K, size_t R, size_t J, int NH = 4, bool harmonic = true) {
	int PROC_SIZE = 1, PROC_RANK = 0;
#ifndef NOMPI
	MPI_Comm_size(MPI_COMM_WORLD, &PROC_SIZE);
	MPI_Comm_rank(MPI_COMM_WORLD, &PROC_RANK);
#endif

	auto M = get_aligned<float>(R * J * g.n);
	//std::vector<float> MG(g.n, 0);
	auto MG = get_aligned<float>(g.n);
#ifndef NOMPI
	std::vector<float> _MG(g.n, 0);
	float* _MG_dev = (float*) get_dev(sizeof(float)*g.n);
#else
#define _MG MG
#define _MG_dev MG_dev
#endif
	auto mask = get_aligned<float>(R * J);
	auto X = get_rands(R, PROC_RANK * R);  //////////////////////
	std::sort(X.get(), X.get() + R);
	int* X_dev = (int*)devcpy(X.get(), sizeof(int) * R);
	std::vector<std::tuple<size_t, float>> results;
	const size_t offset = g.n * R * J * PROC_RANK;
	float* M_dev = (float*) get_dev(sizeof(float) * g.n * R * J);
	float* mask_dev = (float*)get_dev(sizeof(float) * R * J);
	float* MG_dev = (float*)get_dev(sizeof(float) * g.n);

	fill_registers_dev(M_dev, g.n, R, J, NH, offset);

	auto _g = g;
	_g.xadj = (decltype(_g.xadj))devcpy((void*)g.xadj, sizeof(g.xadj[0]) * (g.n + 1));
	_g.adj = (decltype(_g.adj))devcpy((void*)g.adj, sizeof(g.adj[0]) * (g.m));

	Timer t;
	simulate_dev(M_dev, _g, R, J, X_dev);

	while (results.size() < K) {
		if (harmonic)
			harmonic_mean_dev(MG_dev, M_dev, mask_dev, g.n, R, J);
		else
			maxsum_float_dev(MG_dev, M_dev, mask_dev, g.n, R*J);
#ifndef NOMPI
		MPI_Reduce(MG_dev, _MG_dev, g.n, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
#endif
		size_t u;
		if (PROC_RANK == 0)
			u = get_max_dev(_MG_dev, g.n);
#ifndef NOMPI
		MPI_Bcast(&u, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
#endif
		results.push_back(std::make_tuple(u, t.elapsed()));
		max_inplace(mask_dev, M_dev + u * R * J, g.n);
	}
	return results;
}