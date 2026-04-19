# Demos build — does not include parent core/, kernels/, etc.
NVCC := nvcc
NVCC_FLAGS := -std=c++17 -O3 --use_fast_math
CUDA_ARCH ?= -arch=sm_80
COMMON_INC := -Idemos/common

BUILD_DIR := demos/build
$(shell mkdir -p $(BUILD_DIR))

DEMO1 := $(BUILD_DIR)/demo-1
DEMO1_SRC := demos/phase01_infra/main.cu demos/phase01_infra/gemm.cu demos/common/cpu_gemm.cpp

DEMO2_SM := $(BUILD_DIR)/demo-2-softmax
DEMO2_SM_SRC := demos/phase02_ops_autograd/sm.cu

DEMO2_LN := $(BUILD_DIR)/demo-2-layernorm
DEMO2_LN_SRC := demos/phase02_ops_autograd/ln.cu

DEMO2_ATTN := $(BUILD_DIR)/demo-2-attention
DEMO2_ATTN_SRC := demos/phase02_ops_autograd/attn.cu

DEMO3 := $(BUILD_DIR)/demo-3
DEMO3_SRC := demos/phase03_model/main.cpp demos/phase03_model/model.cpp

DEMO4 := $(BUILD_DIR)/demo-4
DEMO4_SRC := \
	demos/phase04_train_generate/main.cpp \
	demos/phase04_train_generate/tok.cpp \
	demos/phase04_train_generate/dataset.cpp \
	demos/phase04_train_generate/micro_lm.cpp \
	demos/phase04_train_generate/train.cpp \
	demos/phase04_train_generate/gen.cpp

DEMO5 := $(BUILD_DIR)/demo-5
DEMO5_SRC := demos/phase05_extensions_opt/main.cu demos/phase05_extensions_opt/fp16_gemm.cu

.PHONY: all demo-1 demo-2 demo-2-softmax demo-2-layernorm demo-2-attention demo-3 demo-4 demo-5 clean

all: demo-1 demo-2 demo-3 demo-4 demo-5

demo-2: demo-2-softmax demo-2-layernorm demo-2-attention

demo-1: $(DEMO1_SRC) demos/Makefile
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $(COMMON_INC) -o $(DEMO1) $(DEMO1_SRC)

demo-2-softmax: $(DEMO2_SM_SRC) demos/Makefile
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $(COMMON_INC) -o $(DEMO2_SM) $(DEMO2_SM_SRC)

demo-2-layernorm: $(DEMO2_LN_SRC) demos/Makefile
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $(COMMON_INC) -o $(DEMO2_LN) $(DEMO2_LN_SRC)

demo-2-attention: $(DEMO2_ATTN_SRC) demos/Makefile
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $(COMMON_INC) -o $(DEMO2_ATTN) $(DEMO2_ATTN_SRC)

demo-3: $(DEMO3_SRC) demos/Makefile
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $(COMMON_INC) -o $(DEMO3) $(DEMO3_SRC)

demo-4: $(DEMO4_SRC) demos/Makefile
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $(COMMON_INC) -o $(DEMO4) $(DEMO4_SRC)

demo-5: $(DEMO5_SRC) demos/Makefile
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $(COMMON_INC) -o $(DEMO5) $(DEMO5_SRC)

clean:
	rm -rf $(BUILD_DIR)
