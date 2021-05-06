# Compiler
CC = g++
NVCC = nvcc

# Compiler flags:
CFLAGS = -std=c++17 -Wall
NVCCFLAGS = -std=c++14

# The build target 
TARGET = runner

all: gpu

gpu:
	$(NVCC) $(NVCCFLAGS) -o runner.o -c runner.cpp -x cu --expt-relaxed-constexpr
	$(NVCC) $(NVCCFLAGS) -o runner_kernel.o -c nearest_neighbor_gpu.cu --expt-relaxed-constexpr
	$(NVCC) $(NVCCFLAGS) runner.o runner_kernel.o -o runner

clean:
	$(RM) $(TARGET)
