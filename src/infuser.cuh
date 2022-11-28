#pragma once
#include "common.h"
#include "graph.h"
#include "mega.h"
void* devcpy(void* host_t, size_t size);
void* get_dev(size_t size);
void fill_registers_dev(float* M, size_t n, size_t R, size_t J, size_t NH,
	size_t offset);
void simulate_dev(float* M, graph_t g, size_t R, size_t J, int* X);
void harmonic_mean_dev(float* MG, float* M, float* mask, size_t n, int R,
	int J);
size_t get_max_dev(float* arr, size_t size);
void max_inplace(float* mask, float* registers, size_t size);
void hostcpy(void* host_ptr, void* dev_ptr, size_t size);
template<typename T>
std::vector<std::tuple<size_t, float>> infuser_gpu(T& g, int K, size_t R,
	size_t J, int NH = 4,
	bool harmonic = true) {
	int PROC_SIZE = 1, PROC_RANK = 0;
#ifndef NOMPI
	MPI_Comm_size(MPI_COMM_WORLD, &PROC_SIZE);
	MPI_Comm_rank(MPI_COMM_WORLD, &PROC_RANK);
#endif
	float* M, * mask;
	int* X;
	M = (float*)get_dev(sizeof(float) * R * J * g.n);
	float* MG = (float*)get_dev(sizeof(float) * g.n);
#ifndef NOMPI
	float* _MG = (float*)get_dev(sizeof(float) * g.n);
#else
#define _MG MG
#endif
	auto MG_HOST = get_aligned<float>(g.n);
	mask = (float*)get_dev(sizeof(float) * R * J);
	auto _X = get_rands(R, PROC_RANK * R);  //////////////////////
	std::sort(_X.get(), _X.get() + R);
	X = (int*) devcpy(_X.get(), sizeof(int) * R);
	std::vector<std::tuple<size_t, float>> results;
	const size_t offset = PROC_RANK * g.n * R * J;


	auto _g = g;
	_g.xadj = (decltype(_g.xadj))devcpy((void*)g.xadj, sizeof(g.xadj[0]) * (g.n + 1));
	_g.adj = (decltype(_g.adj))devcpy((void*)g.adj, sizeof(g.adj[0]) * (g.m));
	std::cout << sizeof(g.xadj[0]) << " " << sizeof(g.adj[0]) << std::endl;

	fill_registers_dev(M, g.n, R, J, NH, offset);
	std::cout << "here" << std::endl;

	std::cout << "filled" << std::endl;
	Timer t;
	simulate_dev(M, _g, R, J, X);
	//for (int i = 0; i < 100; i++) {
	//	float foo;
	//	hostcpy((void*)(&foo), (void*)(M + i), sizeof(float));
	//	std::cout << foo << std::endl;
	//}
	//exit(1);
	auto M_host = get_aligned<float>(g.n * R * J);
	vector<float>MG_host(g.n,0);
	hostcpy(M_host.get(), M, sizeof(float) * g.n * R * J);
	auto mask_host = get_aligned<float>(R * J);
	while (results.size() < K) {
		PARFORV(g, i)
			MG_host[i] = harmonic_mean(mask_host.get(), M_host.get() + (i * R * J), J, R);
		//harmonic_mean_dev(MG, M, mask, g.n, R, J);

		/*for (int i = 0; i < 100; i++) {
			float foo;
			hostcpy((void*)(&foo), (void*)(MG + i), sizeof(float));
			std::cout << foo << std::endl;
		}
		exit(1);
		std::cout << results.size() << " means calculated" << std::endl;*/

#ifndef NOMPI
		MPI_Reduce(MG, _MG, g.n, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
#endif
		size_t u;
		//if (PROC_RANK == 0) u = get_max_dev(_MG, g.n);
		u = std::distance(MG_host.begin(), std::max_element(MG_host.begin(), MG_host.end()));

#ifndef NOMPI
		MPI_Bcast(&u, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
#endif
		results.push_back(
			std::make_tuple((size_t)u, float(t.elapsed())));
		//max_inplace(mask, M + (u * R * J), R * J);
		 for (size_t i = 0; i < R * J; i++)
		   mask_host[i] = std::max(mask_host[i], M_host[u * R * J + i]);
	}
	return results;
}
