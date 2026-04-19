#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

inline void cuda_check(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), file, line);
    std::exit(1);
  }
}
#define CUDA_CK(x) cuda_check((x), __FILE__, __LINE__)
