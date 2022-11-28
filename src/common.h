#pragma once
#include <omp.h>
#ifndef NOMPI
#include <mpi.h>
#endif
#include <algorithm>
#include <chrono>
#include <climits>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <bit>
#if defined(_MSC_VER)
#include <nmmintrin.h>
#endif
#include <algorithm>
#if _MSC_VER >= 1910
#include <execution>
#else
#include <stdlib.h>

#include <parallel/algorithm>
#include <parallel/numeric>
void* _aligned_malloc(size_t size, size_t alignment) {
  void* p;
  if (posix_memalign(&p, alignment, size)) return NULL;
  return p;
}
void _aligned_free(void* p) { free(p); }

#endif

const uint32_t HASHMASK = INT32_MAX;

#define PARMAC _Pragma("omp parallel for schedule(dynamic, 8192)")
#define FORV(g, u) for (size_t u = 0; u < g.n; u++)
#define PARFORV(g, u) PARMAC FORV(g, u)
#define FORVR(g, u, r, lim) FORV(g, u) for (size_t r = 0; r < lim; r++)
#define PARFORVR(g, u, r, lim) PARMAC FORVR(g, u, r, lim)
#define FORE(g, u, e) FORV(g, u) for (auto __pos=g.xadj[u],&e=g.adj[__pos]; __pos<g.xadj[u+1]; __pos++, e=g.adj[__pos])
#define FORL(g, u, e) for (auto [__pos,&e] ={g.xadj[u],g.adj[g.xadj[u]]}; __pos<g.xadj[u+1]; __pos++, e=g.adj[__pos])
#define PARFORE(g, u, e) PARMAC FORE(g, u, e)
#define PARFORVRD(g, u, var) PARREDUX(var)

inline uint32_t __hash(uint64_t h) {
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdL;
  h ^= h >> 33;
  h *= 0xc4ceb9fe1a85ec53L;
  h ^= h >> 33;
  return h & HASHMASK;  // FFFF;
}
inline uint64_t __hash64(uint64_t h) {
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdL;
  h ^= h >> 33;
  h *= 0xc4ceb9fe1a85ec53L;
  h ^= h >> 33;
  return h;  // FFFF;
}
inline uint32_t edge_hash(const uint32_t x, const uint32_t y) {
  uint64_t h = (((uint64_t)x) << 32) | y;
  return __hash(h);
}
inline uint32_t __hash(const uint32_t x, const uint32_t y) {
  uint64_t h = (x > y) ? (((uint64_t)y) << 32) | x : (((uint64_t)x) << 32) | y;
  return __hash(h);
}

template <typename T>
void parfill(std::vector<T>& buf, T pattern) {
  const int len = buf.size();
#pragma omp parallel for
  for (int i = 0; i < len; i++) buf[i] = pattern;
}

template <class T>
std::unique_ptr<T[], decltype(&_aligned_free)> get_aligned(size_t elems,
                                                           size_t align = 64) {
  auto* ptr = static_cast<T*>(_aligned_malloc(sizeof(T) * elems, align));
  std::fill(ptr, ptr + elems, 0);
  return std::unique_ptr<T[], decltype(&_aligned_free)>(ptr, &_aligned_free);
}

std::unique_ptr<int[], decltype(&_aligned_free)> get_rands(size_t size,
                                                           size_t offset = 0) {
  std::default_random_engine e1(42);
  std::uniform_int_distribution<int> uniform_dist(0, INT_MAX);
  for (size_t i = 0; i < offset; i++)
    uniform_dist(e1);  // burn few to get deterministic values

  auto rand_seeds = get_aligned<int>(size);
  for (size_t i = 0; i < size; i++) rand_seeds[i] = uniform_dist(e1);
  return move(rand_seeds);
}

struct Timer{
	typedef std::chrono::high_resolution_clock clock_;
	typedef std::chrono::duration<double, std::ratio<1> > second_;
	std::chrono::time_point<clock_> beg_;
	Timer() : beg_(clock_::now()) {}
	void reset() { beg_ = clock_::now(); }
	double elapsed() const { return std::chrono::duration_cast<second_> (clock_::now() - beg_).count(); }
} t;