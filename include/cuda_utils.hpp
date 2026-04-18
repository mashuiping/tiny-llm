#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdio>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            throw std::runtime_error("CUDA error"); \
        } \
    } while (0)

#define TILE_SIZE 16
#define BLOCK_SIZE 256

inline int get_num_blocks(int n, int block_size = BLOCK_SIZE) {
    return (n + block_size - 1) / block_size;
}
