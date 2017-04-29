CUDA_DIR=/usr/local/cuda-8.0
NVCC=$(CUDA_DIR)/bin/nvcc
CC=g++

FFMPEG=/home/peiya/ffmpeg-3.2.4/ffmpeg
CXX_FLAGS = -G -std=c++11 -arch sm_30 -O2 -I../utils 

all:
	$(NVCC) $(CXX_FLAGS) main.cu counting.cu -o lab2


