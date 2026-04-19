# Demos build — does not include parent core/, kernels/, etc.
# Paths are anchored to this Makefile so `make -f demos/Makefile` from repo root
# and `make` from inside demos/ both work.

THIS_MK := $(abspath $(lastword $(MAKEFILE_LIST)))
DEMO_DIR := $(dir $(THIS_MK))

NVCC := nvcc
NVCC_FLAGS := -std=c++17 -O3 --use_fast_math
CUDA_ARCH ?= -arch=sm_80
COMMON_INC := -I$(DEMO_DIR)common

BUILD_DIR := $(DEMO_DIR)build
$(shell mkdir -p $(BUILD_DIR))

DEMO1 := $(BUILD_DIR)/demo-1
DEMO1_SRC := $(DEMO_DIR)phase01_infra/main.cu $(DEMO_DIR)phase01_infra/gemm.cu $(DEMO_DIR)common/cpu_gemm.cpp

DEMO2_SM := $(BUILD_DIR)/demo-2-softmax
DEMO2_SM_SRC := $(DEMO_DIR)phase02_ops_autograd/sm.cu

DEMO2_LN := $(BUILD_DIR)/demo-2-layernorm
DEMO2_LN_SRC := $(DEMO_DIR)phase02_ops_autograd/ln.cu

DEMO2_ATTN := $(BUILD_DIR)/demo-2-attention
DEMO2_ATTN_SRC := $(DEMO_DIR)phase02_ops_autograd/attn.cu

DEMO3 := $(BUILD_DIR)/demo-3
DEMO3_SRC := $(DEMO_DIR)phase03_model/main.cpp $(DEMO_DIR)phase03_model/model.cpp

DEMO4 := $(BUILD_DIR)/demo-4
DEMO4_SRC := \
	$(DEMO_DIR)phase04_train_generate/main.cpp \
	$(DEMO_DIR)phase04_train_generate/tok.cpp \
	$(DEMO_DIR)phase04_train_generate/dataset.cpp \
	$(DEMO_DIR)phase04_train_generate/micro_lm.cpp \
	$(DEMO_DIR)phase04_train_generate/train.cpp \
	$(DEMO_DIR)phase04_train_generate/gen.cpp

DEMO5 := $(BUILD_DIR)/demo-5
DEMO5_SRC := $(DEMO_DIR)phase05_extensions_opt/main.cu $(DEMO_DIR)phase05_extensions_opt/fp16_gemm.cu

.PHONY: all demo-1 demo-2 demo-2-softmax demo-2-layernorm demo-2-attention demo-3 demo-4 demo-5 clean

all: demo-1 demo-2 demo-3 demo-4 demo-5

demo-2: demo-2-softmax demo-2-layernorm demo-2-attention

demo-1: $(DEMO1_SRC) $(THIS_MK)
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $(COMMON_INC) -o $(DEMO1) $(DEMO1_SRC)

demo-2-softmax: $(DEMO2_SM_SRC) $(THIS_MK)
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $(COMMON_INC) -o $(DEMO2_SM) $(DEMO2_SM_SRC)

demo-2-layernorm: $(DEMO2_LN_SRC) $(THIS_MK)
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $(COMMON_INC) -o $(DEMO2_LN) $(DEMO2_LN_SRC)

demo-2-attention: $(DEMO2_ATTN_SRC) $(THIS_MK)
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $(COMMON_INC) -o $(DEMO2_ATTN) $(DEMO2_ATTN_SRC)

demo-3: $(DEMO3_SRC) $(THIS_MK)
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $(COMMON_INC) -o $(DEMO3) $(DEMO3_SRC)

demo-4: $(DEMO4_SRC) $(THIS_MK)
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $(COMMON_INC) -o $(DEMO4) $(DEMO4_SRC)

demo-5: $(DEMO5_SRC) $(THIS_MK)
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $(COMMON_INC) -o $(DEMO5) $(DEMO5_SRC)

clean:
	rm -rf $(BUILD_DIR)
