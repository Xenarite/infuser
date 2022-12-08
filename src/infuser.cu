#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>

#include <iostream>

#include "graph.h"

inline void assert_gpu(cudaError_t code, const char* file, int line) {
	if (code != cudaSuccess) {
		std::cerr << "CUDAERR:" << cudaGetErrorString(code) << " " << file << ":" << line
			<< std::endl;
		abort();
	}
}
#define cuchk(ans) \
  { assert_gpu((ans), __FILE__, __LINE__); }
void* get_dev(size_t size) {
	void* t;
	cuchk(cudaMalloc((void**)&t, size));
	cuchk(cudaMemset(t, 0, size));
	return t;
}
void devsync(){
	cudaDeviceSynchronize();
}


void* devcpy(void* host_t, size_t size) {
	void* dev_t;
	cuchk(cudaMalloc((void**)&dev_t, size));
	cuchk(cudaMemcpy(dev_t, host_t, size, cudaMemcpyHostToDevice));
	return dev_t;
}
void hostcpy(void* host_ptr, void* dev_ptr, size_t size) {
	cuchk(cudaMemcpy(host_ptr, dev_ptr, size, cudaMemcpyDeviceToHost));
}
__device__ inline uint64_t dev_hash64(uint64_t h) {
	h ^= h >> 33;
	h *= 0xff51afd7ed558ccdL;
	h ^= h >> 33;
	h *= 0xc4ceb9fe1a85ec53L;
	h ^= h >> 33;
	return h;
}
__global__ void fill_hypers_kernel(float* M, const size_t n, const size_t R,
	const size_t J, const size_t NH,
	const size_t PROC_RANK=0, const size_t PROC_SIZE=1) {
	// size_t R = blockDim.x;
	const int bucketmask = J - 1;
	for (size_t i = blockIdx.x; i < n; i += (gridDim.x)) {
		for (size_t j = threadIdx.x; j < R; j += blockDim.x) {
			uint64_t val = ~((i + 1) * (R*PROC_SIZE) + (R*PROC_RANK) + j);
			float zeros = 0;
			for (int x = 0; x < NH; x++) {
				val = dev_hash64(val);
				zeros += float(__clzll(val)) / NH;
			}
			auto bucket = dev_hash64(val) & bucketmask;
			M[i * R * J + j * J + bucket] = float(zeros);
		}
	}
}

void fill_registers_dev(float* M, size_t n, size_t R, size_t J, size_t NH,
	size_t PROC_RANK, size_t PROC_SIZE) {
	fill_hypers_kernel << <n, 32 >> > (M, n, R, J, NH, PROC_RANK, PROC_SIZE);
	cuchk(cudaDeviceSynchronize());
	cuchk(cudaPeekAtLastError());
}

__global__ void simulate_kernel(float* M, graph_t g, int R, int J, int* X, char* L, char iter) {
	for (size_t i = blockIdx.x; i < g.n; i += (gridDim.x)) {
		for (size_t r = threadIdx.x; r < R; r += blockDim.x) {
			const auto x = X[r];
			for (size_t j = 0; j < J; j++) {
				auto reg = M[i * R * J + r * J + j];
				for (size_t pos = g.xadj[i]; pos < g.xadj[i + 1]; pos++) {
					const auto e = g.adj[pos];
					if (L[e.v] < iter) continue;
					if (((e.hash ^ x) <= e.w))
						reg = max(reg, M[e.v * R * J + r * J + j]);
				}
				if (M[i * R * J + r * J + j] != reg) {
					M[i * R * J + r * J + j] = reg;
					L[i] = iter + 1;
				}
			}
		}
	}
}

const int ITER_LIMIT = 40;
const float CONV_PT = 0.01;

void simulate_dev(float* M, graph_t g, size_t R, size_t J, int* X) {
	char* L = (char*) get_dev(sizeof(char) * g.n);
	for (int i = 0; i < ITER_LIMIT; i++) {
		simulate_kernel << <1024, 32 >> > (M, g, R, J, X, L, i);
		size_t count = thrust::count(
			thrust::device_ptr<char>(L), 
			thrust::device_ptr<char>(L + g.n), 
			char(i));
		if (count < CONV_PT * g.n)
			break;
	}
	cuchk(cudaFree(L));
	cuchk(cudaDeviceSynchronize());
	cuchk(cudaPeekAtLastError());
}

template <typename T>
__inline__ __device__ T reduce_warp(T val) {
	for (int offset = (warpSize >> 1); offset > 0;
		offset = (offset >> 1))  // div 2 -> shift 1
		val += __shfl_down_sync(0xFFFFFFFF, val, offset);
	return val;
}
template <typename T>
__inline__ __device__ T reduce_block(T val) {
	static __shared__ T s[32];
	int lane = threadIdx.x & 0x1f;  // last 5 bits== mod 32
	int warp = threadIdx.x >> 5;    // shift 5 -> div 32
	val = reduce_warp(val);
	if (lane == 0) s[warp] = val;
	__syncthreads();
	val = (threadIdx.x < blockDim.x / warpSize) ? s[lane] : 0;
	if (warp == 0) val = reduce_warp(val);
	return val;
}

template <typename T>
__inline__ __device__ T reduce_warp(T val, int size) {
	for (int offset = 1; offset < size; offset <<= 1)
		val += __shfl_down_sync(0xFFFFFFFF, val, offset);
	return val;
}

template <typename T, typename T2>
__global__ void
harmonic_sum_kernel(T2* estimates, T* M, T* mask, size_t N, int R, int J) {
	const auto RJ = R * J;
	for (size_t u = blockIdx.x; u < N; u += (gridDim.x)) {
		if (threadIdx.x == 0) {
			estimates[u] = 0;
		}
		float sum = 0;
		for (size_t pos = threadIdx.x; pos < RJ; pos += blockDim.x) {
			const auto r = pos / J;
			const auto val = powf(2, -max(M[u * RJ + pos], mask[pos]));
			sum += reduce_warp(val, J);
		}
		if (threadIdx.x % J == 0) {
			atomicAdd(estimates + u, (1.0f / sum) / R);
		}
	}
}

void harmonic_mean_dev(float* MG, float* M, float* mask, size_t n, int R, int J) {
	harmonic_sum_kernel <<< 1024, 32 >>> (MG, M, mask, n, R, J);
}
size_t get_max_dev(float* arr, size_t size) {
	auto max_elem = thrust::max_element(thrust::device_ptr<float>(arr),
		thrust::device_ptr<float>(arr) + size);
	size_t s = thrust::distance(thrust::device_ptr<float>(arr), max_elem);
	return s;
}
void max_inplace(float* mask, float* registers, size_t size) {
	thrust::transform(thrust::device_ptr<float>(registers),
		thrust::device_ptr<float>(registers + size),
		thrust::device_ptr<float>(mask),
		thrust::device_ptr<float>(mask), thrust::maximum<float>());
}
template<typename T, typename T2>
__global__
void maxsum_kernel(T2* estimates, T* hypers, T* mask, int R, size_t N) {
    for (size_t i = blockIdx.x; i < N; i += (gridDim.x)) {
		estimates[i] = 0;
		float sum = 0;
		for(size_t j = threadIdx.x; j< R; j+=blockDim.x){
			T2 m = max(hypers[i * R + j], mask[j]);
			sum+=m;
		}
		__syncthreads();
		sum = reduce_block(sum);
		if (threadIdx.x == 0)  estimates[i] = sum;
	
    }
}
void maxsum_char_dev(float* estimates, char* hypers, char* mask, size_t N, int R){
	maxsum_kernel<<<1024,32>>>(estimates,hypers,mask,R,N);
}
void maxsum_float_dev(float* estimates, float* hypers, float* mask, size_t N, int R){
	maxsum_kernel<<<1024,32>>>(estimates,hypers,mask,R,N);
}
size_t count(float* start, float* end, float item){
	return thrust::count(
		thrust::device_ptr<float>(start), 
		thrust::device_ptr<float>(end), 
		-1.0f);
}
