#pragma once
#include "common.h"
#include "graph.h"
template <typename T>
float maxsum(T* start, T* end, T* mask) {
	float acc = 0;
	for (auto* i = start; i < end; i++) {
		acc += std::max(*i, *(mask++));
	}
	return acc;
}
template <typename T>
float harmonic_mean(T* x, T* y, size_t size, size_t R) {
	float total = 0;
	for (size_t r = 0; r < R; ++r) {
		float sum = 0;
		for (size_t i = 0; i < size; i++)
			sum += powf(
				2, -std::max(x[r * size + i],
					y[r * size + i]));  // 1.0/(1 << v);  // FIXME lookup
		total += 1.0f / sum;
	}
	return total / R;
}

template <class graph_t, typename T>
void simulate_2d(graph_t& g, const size_t R, const size_t J, T* __restrict M,
	int* __restrict rands) {
	const int ITER_LIMIT = 40;
	const float threshold = 0.02;
	std::vector<char> active(g.n, 1), nactive(g.n, 0);
	const size_t RJ = R * J;
	for (int iter = 0; iter < ITER_LIMIT; iter++) {
		PARFORV(g, u) {
			bool flag = false;
			for (size_t pos = g.xadj[u]; pos < g.xadj[u + 1]; pos++) {
				const auto e = g.adj[pos];
				if (!active[e.v]) continue;
				for (int r = 0; r < R; r++) {
					if (((rands[r] ^ e.hash) <= e.w)) {
						for (int j = 0; j < J; j++)
							if (M[e.v * RJ + r * J + j] > M[u * RJ + r * J + j]) {
								M[u * RJ + r * J + j] = M[e.v * RJ + r * J + j];
								flag = true;
							}
					}
				}
			}
			if (flag) nactive[u] = true;
		}
		size_t active_count = 0;
		int step = 100;
#pragma omp parallel for reduction(+ : active_count)
		for (int i = 0; i < g.n; i += step) {
			active_count += nactive[i];
		}
		double active_rate = double(active_count) * double(step) / g.n;
		if (active_rate <= threshold) {
			break;
		}
		swap(active, nactive);
		parfill(nactive, char(0));
	}
}
template <class graph_t, typename T>
void simulate_simple(graph_t& g, const size_t R, const size_t J, T* __restrict M,
	int* __restrict rands) {
	const int ITER_LIMIT = 40;
	const size_t RJ = R * J;
	for (int iter = 0; iter < ITER_LIMIT; iter++) {
		PARFORV(g, u) {
			for (size_t pos = g.xadj[u]; pos < g.xadj[u + 1]; pos++) {
				const auto e = g.adj[pos];
				for (int r = 0; r < R; r++) {
					if (((rands[r] ^ e.hash) <= e.w)) {
						for (int j = 0; j < J; j++)
							if (M[e.v * RJ + r * J + j] > M[u * RJ + r * J + j]) {
								M[u * RJ + r * J + j] = M[e.v * RJ + r * J + j];
							}
					}
				}
			}
		}
	}
}
template <typename graph_t, typename reg_t>
std::vector<std::tuple<size_t, float>> infuser_celf(graph_t& g, int K, size_t R,
	size_t J, int NH = 4,
	bool harmonic = true) {
	int PROC_SIZE, PROC_RANK;
	MPI_Comm_size(MPI_COMM_WORLD, &PROC_SIZE);
	MPI_Comm_rank(MPI_COMM_WORLD, &PROC_RANK);

	auto M = get_aligned<reg_t>(R * J * g.n);
	std::vector<float> MG(g.n, 0);
	auto mask = get_aligned<reg_t>(R * J);
	auto X = get_rands(R, PROC_RANK * R);  //////////////////////
	std::sort(X.get(), X.get() + R);
	std::vector<int> iteration(g.n, 0);
	std::vector<std::tuple<size_t, float>> results;
	auto cmp = [&](float left, float right) { return (MG[left] < MG[right]); };
	std::priority_queue<size_t, std::deque<size_t>, decltype(cmp)> q(cmp);

	const int bucketmask = J - 1;
	PARFORVR(g, i, j, R) {
		uint64_t val = ~((i + 1) * R + j);
		float zeros = 0;
		for (int x = 0; x < NH; x++) {
			val = __hash64(val);
			zeros += float(__builtin_clzll(val)) / NH;
		}
		auto bucket = __hash64(val) & bucketmask;
		M[i * R * J + j * J + bucket] = reg_t(zeros);
	}

	double time = omp_get_wtime();
	simulate_2d(g, R, J, M.get(), X.get());
	PARFORV(g, i)
		MG[i] = harmonic ? harmonic_mean(mask.get(), M.get() + (i * R * J), J, R)
		: maxsum(M.get() + (i * R * J),
			M.get() + ((i + 1ull) * R * J), mask.get());

	// MPI_Barrier(MPI_COMM_WORLD);

	if (PROC_RANK == 0) {
		FORV(g, i)
			q.push(i);
		float score = 0;
		while (results.size() < K) {
			size_t u = q.top();
			q.pop();
			if (iteration[u] == results.size()) {
				score += MG[u];
				results.push_back(
					std::make_tuple<size_t, float>((size_t)u, omp_get_wtime() - time));
				for (int i = 0; i < R * J; i++)
					mask[i] = std::max(mask[i], M[u * R * J + i]);
				u += INT64_MAX;
				MPI_Bcast(&u, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
			}
			else {
				MPI_Bcast(&u, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
				float local_score = (harmonic
					? harmonic_mean(mask.get(), M.get() + (u * R * J), J, R)
					: maxsum(M.get() + (u * R * J),
						M.get() + ((u + 1ull) * R * J), mask.get()));
				MG[u] = local_score - score;
				float local_mg = MG[u], global_mg;
				MPI_Reduce(&local_mg, &global_mg, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
				MG[u] = global_mg / PROC_SIZE;
				iteration[u] = results.size();
				q.push(u);
			}
		}
	}
	else {
		while (true) {
			size_t u = 0;
			MPI_Bcast(&u, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
			// std::cout << PROC_RANK << " looks for "<< u <<std::endl;
			if (u == UINT64_MAX) break;
			if (u > INT64_MAX) {
				for (int i = 0; i < R * J; i++)
					mask[i] = std::max(mask[i], M[(u - INT64_MAX) * R * J + i]);
				std::cout << (u - INT64_MAX) << " commited " << std::endl;
				continue;
			}
			MG[u] = harmonic ? harmonic_mean(mask.get(), M.get() + (u * R * J), J, R)
				: maxsum(M.get() + (u * R * J),
					M.get() + ((u + 1ull) * R * J), mask.get());
			float local_mg = MG[u], global_mg;
			MPI_Reduce(&local_mg, &global_mg, 1, MPI_FLOAT, MPI_SUM, 0,
				MPI_COMM_WORLD);
		}
	}
	if (PROC_RANK == 0) {
		uint64_t ullmax = UINT64_MAX;
		MPI_Bcast(&ullmax, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
	}
	return results;
}

template <typename reg_t>
std::vector<std::tuple<size_t, float>>
infuser(graph_t& g, int K, size_t R,
	size_t J, int NH = 4,
	bool harmonic = true) {
	int PROC_SIZE = 1, PROC_RANK = 0;
#ifndef NOMPI
	MPI_Comm_size(MPI_COMM_WORLD, &PROC_SIZE);
	MPI_Comm_rank(MPI_COMM_WORLD, &PROC_RANK);
#endif

	auto M = get_aligned<reg_t>(R * J * g.n);
	std::vector<float> MG(g.n, 0);
#ifndef NOMPI
	std::vector<float> _MG(g.n, 0);
#else
#define _MG MG
#endif
	auto mask = get_aligned<reg_t>(R * J);
	auto X = get_rands(R, PROC_RANK * R);  //////////////////////
	std::sort(X.get(), X.get() + R);
	std::vector<int> iteration(g.n, 0);
	std::vector<std::tuple<size_t, float>> results;

	const int bucketmask = J - 1;
	PARFORVR(g, i, j, R) {
		uint64_t val = ~((i + 1) * R + j);
		float zeros = 0;
		for (int x = 0; x < NH; x++) {
			val = __hash64(val);
			zeros += float(__builtin_clzll(val)) / NH;
		}
		auto bucket = __hash64(val) & bucketmask;
		M[i * R * J + j * J + bucket] = reg_t(zeros);
	}

	Timer t;
	simulate_2d(g, R, J, M.get(), X.get());
	while (results.size() < K) {
		PARFORV(g, i)
			MG[i] = harmonic ? harmonic_mean(mask.get(), M.get() + (i * R * J), J, R)
			: maxsum(M.get() + (i * R * J),
				M.get() + ((i + 1ull) * R * J), mask.get());

#ifndef NOMPI
		MPI_Reduce(MG.data(), _MG.data(), g.n, MPI_FLOAT, MPI_SUM, 0,
			MPI_COMM_WORLD);
#endif

		size_t u;
		if (PROC_RANK == 0)
			u = std::distance(_MG.begin(), std::max_element(_MG.begin(), _MG.end()));
#ifndef NOMPI
		MPI_Bcast(&u, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
#endif
		results.push_back(
			std::make_tuple(u, t.elapsed()));
		for (size_t i = 0; i < R * J; i++)
			mask[i] = std::max(mask[i], M[u * R * J + i]);
		PARFORV(g, u)
			MG[u] = harmonic ? harmonic_mean(mask.get(), M.get() + (u * R * J), J, R)
			: maxsum(M.get() + (u * R * J),
				M.get() + ((u + 1ull) * R * J), mask.get());
	}
	return results;
}
template <class graph_t>
double run_ic(graph_t& g, const uint32_t S, const size_t R, int32_t* rand_seeds,
	char* visited) {
	const size_t n = g.n, BLOCKSIZE = 64;
	const size_t Roffset = R / 8;
	std::fill(visited + S * Roffset, visited + (S + 1) * Roffset, UINT8_MAX);
	const int ITER_LIMIT = 50;
	std::vector<char> active(g.n, 0), nactive(g.n, 0);
	active[S] = 1;
	for (int iter = 0; iter < ITER_LIMIT; iter++) {
		bool cont = false;
#pragma omp parallel for schedule(dynamic, 8192) reduction(+:cont)// reduction(+:active_count) //
		FORV(g, u) {
			if (!active[u]) continue;
			for (auto pos = g.xadj[u]; pos < g.xadj[u + 1]; pos++) {
				auto e = g.adj[pos]; //FORL(g,u,e){//for (auto e : g.edges(u)) {
				bool flag = false;
				for (int r = 0; r < R; r += BLOCKSIZE) {
					const auto roffset = r / 8;
					const uint64_t curr_sims =
						*((uint64_t*)&(visited[u * (Roffset)+(roffset)]));
					uint64_t packed = 0;
					for (int b = 0; b < BLOCKSIZE; b++) {
						if ((rand_seeds[r + b] ^ e.hash) < e.w) {
							packed |= 1LL << b;
						}
					}
					const auto will_visit =
						packed & curr_sims &
						~(*(uint64_t*)&visited[e.v * (Roffset)+(roffset)]);
					if (will_visit) {
						*(uint64_t*)&visited[e.v * (Roffset)+(roffset)] |= will_visit;
						flag = true;
					}
				}
				if (flag) {
					nactive[e.v] = true;
					cont = true;
				}
			}
		}
		if (!cont) break;
		std::swap(active, nactive);
		parfill(nactive, char(0));
	}

	uint64_t* ptr = (uint64_t*)visited;
	size_t size = g.n * R / (sizeof(uint64_t) * 8);
	double score = 0;
#pragma omp parallel for reduction(+ : score)
	for (long long i = 0; i < size; i++) {
		score += __builtin_popcountll(ptr[i]);
	}

	return score / R;
}