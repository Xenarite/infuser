#pragma once
#include "common.h"
#include "graph.h"
#include "mega.h"
#include <cuda_runtime.h>

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
bool checkequal(T* host_ptr, T* dev_ptr, size_t size) {
	auto host_dev_ptr = get_aligned<T>(size);
	hostcpy(host_dev_ptr.get(), dev_ptr, size * sizeof(T));
	return std::equal(host_ptr, host_ptr+size, host_dev_ptr.get());
}
//
//template<typename T>
//std::vector<std::tuple<size_t, float>> 
//infuser_gpu(T& g, int K, size_t R,
//	size_t J, int NH = 4,
//	bool harmonic = true) {
//	int PROC_SIZE = 1, PROC_RANK = 0;
//#ifndef NOMPI
//	MPI_Comm_size(MPI_COMM_WORLD, &PROC_SIZE);
//	MPI_Comm_rank(MPI_COMM_WORLD, &PROC_RANK);
//#endif
//	float* M, * mask;
//	int* X;
//	M = (float*)get_dev(sizeof(float) * R * J * g.n);
//	float* MG = (float*)get_dev(sizeof(float) * g.n);
//#ifndef NOMPI
//	float* _MG = (float*)get_dev(sizeof(float) * g.n);
//#else
//#define _MG MG
//#endif
//    vector<float>MG_HOST(g.n, 0);
//	mask = (float*)get_dev(sizeof(float) * R * J);
//	auto _X = get_rands(R, PROC_RANK * R);  //////////////////////
//	std::sort(_X.get(), _X.get() + R);
//	X = (int*) devcpy(_X.get(), sizeof(int) * R);
//	std::vector<std::tuple<size_t, float>> results;
//	const size_t offset = PROC_RANK * g.n * R * J;
//
//
//	auto _g = g;
//	_g.xadj = (decltype(_g.xadj))devcpy((void*)g.xadj, sizeof(g.xadj[0]) * (g.n + 1));
//	_g.adj = (decltype(_g.adj))devcpy((void*)g.adj, sizeof(g.adj[0]) * (g.m));
//	std::cout << sizeof(g.xadj[0]) << " " << sizeof(g.adj[0]) << std::endl;
//	auto M_host = get_aligned<float>(g.n * R * J);
//	const int bucketmask = J - 1;
//	PARFORVR(g, i, j, R) {
//		uint64_t val = ~((i + 1) * R + j);
//		float zeros = 0;
//		for (int x = 0; x < NH; x++) {
//			val = __hash64(val);
//			zeros += float(__builtin_clzll(val)) / NH;
//		}
//		auto bucket = __hash64(val) & bucketmask;
//		M_host[i * R * J + j * J + bucket] = float(zeros);
//	}
//
//	fill_registers_dev(M, g.n, R, J, NH, offset);
//	std::cout << (checkequal(M_host.get(), M, g.n*R*J)?"yes":"no") << std::endl;
//	std::cout << "filled" << std::endl;
//	Timer t;
//	simulate_2d(g, R, J, M_host.get(), _X.get());
//	//cuchk(cudaMemcpy(M, M_host.get(), g.n*R*J*sizeof(float), cudaMemcpyHostToDevice));
//
//	//simulate_dev(M, _g, R, J, X);
//	//std::cout << (checkequal(M_host.get(), M, g.n * R * J) ? "yes" : "no") << std::endl;
//	//exit(1);
//	//for (int i = 0; i < 100; i++) {
//	//	float foo;
//	//	hostcpy((void*)(&foo), (void*)(M + i), sizeof(float));
//	//	std::cout << foo << std::endl;
//	//}
//	//exit(1);
//	vector<float>MG_host(g.n,0);
//	//hostcpy(M_host.get(), M, sizeof(float) * g.n * R * J);
//	auto mask_host = get_aligned<float>(R * J);
//
//	while (results.size() < K) {
//		PARFORV(g, i)
//			MG_host[i] =harmonic_mean(mask_host.get(), M_host.get() + (i * R * J), J, R);
//
//#ifndef NOMPI
//		MPI_Reduce(MG.data(), _MG.data(), g.n, MPI_FLOAT, MPI_SUM, 0,
//			MPI_COMM_WORLD);
//#endif
//
//		size_t u;
//		if (PROC_RANK == 0)
//			u = std::distance(MG_HOST.begin(), std::max_element(MG_HOST.begin(), MG_HOST.end()));
//#ifndef NOMPI
//		MPI_Bcast(&u, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
//#endif
//		results.push_back(
//			std::make_tuple(u, t.elapsed()));
//		for (size_t i = 0; i < R * J; i++)
//			mask_host[i] = std::max(mask_host[i], M_host[u * R * J + i]);
//	}
//
//
//
////	while (results.size() < K) {
////		PARFORV(g, i)
////			MG_host[i] = harmonic_mean(mask_host.get(), M_host.get() + (i * R * J), J, R);
////		//harmonic_mean_dev(MG, M, mask, g.n, R, J);
////
////		/*for (int i = 0; i < 100; i++) {
////			float foo;
////			hostcpy((void*)(&foo), (void*)(MG + i), sizeof(float));
////			std::cout << foo << std::endl;
////		}
////		exit(1);
////		std::cout << results.size() << " means calculated" << std::endl;*/
////
////#ifndef NOMPI
////		MPI_Reduce(MG, _MG, g.n, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
////#endif
////		size_t u;
////		//if (PROC_RANK == 0) u = get_max_dev(_MG, g.n);
////		u = std::distance(MG_host.begin(), std::max_element(MG_host.begin(), MG_host.end()));
////
////#ifndef NOMPI
////		MPI_Bcast(&u, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
////#endif
////		results.push_back(
////			std::make_tuple((size_t)u, float(t.elapsed())));
////		//max_inplace(mask, M + (u * R * J), R * J);
////		 for (size_t i = 0; i < R * J; i++)
////		   mask_host[i] = std::max(mask_host[i], M_host[u * R * J + i]);
////	}
//	return results;
//}
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
#else
#define _MG MG
#define _MG_dev MG_dev
#endif
	auto mask = get_aligned<float>(R * J);
	auto X = get_rands(R, PROC_RANK * R);  //////////////////////
	std::sort(X.get(), X.get() + R);
	std::vector<int> iteration(g.n, 0);
	std::vector<std::tuple<size_t, float>> results;
	const size_t offset = g.n * R * J * PROC_RANK;
	float* M_dev = (float*) get_dev(sizeof(float) * g.n * R * J);
	fill_registers_dev(M_dev, g.n, R, J, NH, offset);
	hostcpy(M.get(), M_dev, sizeof(float) * g.n * R * J);
	//const int bucketmask = J - 1;
	//PARFORVR(g, i, j, R) {
	//	uint64_t val = ~((i + 1) * R + j);
	//	float zeros = 0;
	//	for (int x = 0; x < NH; x++) {
	//		val = __hash64(val);
	//		zeros += float(__builtin_clzll(val)) / NH;
	//	}
	//	auto bucket = __hash64(val) & bucketmask;
	//	M[i * R * J + j * J + bucket] = float(zeros);
	//}
	//simulate_simple(g, R, J, M.get(), X.get());
	int* X_dev = (int*)devcpy(X.get(), sizeof(int) * R);
	auto _g = g;
	_g.xadj = (decltype(_g.xadj))devcpy((void*)g.xadj, sizeof(g.xadj[0]) * (g.n + 1));
	_g.adj = (decltype(_g.adj))devcpy((void*)g.adj, sizeof(g.adj[0]) * (g.m));
	//bool res = checkequal(X.get(), X_dev, R);
	//std::cout << "before sim:" << res << std::endl;
	//simulate_simple(g, R, J, M.get(), X.get());
	Timer t;
	simulate_dev(M_dev, _g, R, J, X_dev);
	//std::cout << "after sim:" << checkequal(M.get(), M_dev, g.n * R * J) << std::endl;

	//(cudaMemcpy(M.get(), M_dev, g.n * R*J*sizeof(float), cudaMemcpyDeviceToHost));

	
	float* mask_dev = (float*)get_dev(sizeof(float) * R * J);
	//(cudaMemcpy(M_dev, M.get(), g.n*R*J*sizeof(float), cudaMemcpyHostToDevice));
	//(cudaMemcpy(M.get(),M_dev, g.n*R*J*sizeof(float), cudaMemcpyHostToDevice));
	//M_dev = (float*)devcpy(M.get(), g.n * R * J * sizeof(float));
	float* MG_dev = (float*)get_dev(sizeof(float) * g.n);
	while (results.size() < K) {
		//(cudaMemcpy(mask_dev, mask.get(), R* J * sizeof(float), cudaMemcpyHostToDevice));
		harmonic_mean_dev(MG_dev, M_dev, mask_dev, g.n, R, J);
		//PARFORV(g, i)
		//	MG[i] = harmonic_mean(mask.get(), M.get() + (i * R * J), J, R);
		//(cudaMemcpy(MG.get(), MG_dev, g.n * sizeof(float), cudaMemcpyDeviceToHost));


#ifndef NOMPI
		MPI_Reduce(MG.data(), _MG.data(), g.n, MPI_FLOAT, MPI_SUM, 0,
			MPI_COMM_WORLD);
#endif

		size_t u;
		if (PROC_RANK == 0)
			u = get_max_dev(_MG_dev, g.n);
			//u = std::distance(_MG.get(), std::max_element(_MG.get(), _MG.get()+g.n));
#ifndef NOMPI
		MPI_Bcast(&u, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
#endif
		results.push_back(
			std::make_tuple(u, t.elapsed()));
		max_inplace(mask_dev, M_dev + u * R * J, g.n);
		//for (size_t i = 0; i < R * J; i++)
		//	mask[i] = std::max(mask[i], M[u * R * J + i]);
		//PARFORV(g, u)
		//	MG[u] = harmonic ? harmonic_mean(mask.get(), M.get() + (u * R * J), J, R)
		//	: maxsum(M.get() + (u * R * J),
		//		M.get() + ((u + 1ull) * R * J), mask.get());
	}
	return results;
}