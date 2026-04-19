NVCC = nvcc
NVCC_FLAGS = -std=c++17 -O3 -use_fast_math
CUDA_ARCH = -arch=sm_80

INCLUDES = -Iinclude
KERNELS = kernels/gemm.cu kernels/attention.cu kernels/softmax.cu kernels/layernorm.cu kernels/embedding.cu kernels/utils.cu

SRC = main.cpp core/tensor.cpp core/autograd.cpp core/optimizer.cpp layers/embedding.cpp layers/linear.cpp layers/layernorm.cpp layers/transformer.cpp layers/model.cpp data/poetry_dataset.cpp data/tokenizer.cpp train/trainer.cpp generate/generator.cpp

.PHONY: all train generate clean sync

all: train generate

# train and generate targets intentionally use identical sources but produce different binaries
train: $(SRC) $(KERNELS)
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $(INCLUDES) -o train $(SRC) $(KERNELS) -lcublas

generate: $(SRC) $(KERNELS)
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $(INCLUDES) -o generate $(SRC) $(KERNELS) -lcublas

sync:
	./tools/k8s/sync-to-build-gpu.sh

clean:
	rm -f train generate *.o
