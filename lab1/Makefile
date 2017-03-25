CUDA_DIR=/usr/local/cuda-8.0
NVCC=$(CUDA_DIR)/bin/nvcc

FFMPEG=/home/peiya/ffmpeg-3.2.4/ffmpeg
#/usr/local/cuda-8.0/bin/nvcc -ccbin g++ -I../../common/inc -m64 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_60,code=compute_60 -o simpleTexture3D_kernel.o -c simpleTexture3D_kernel.cu
INCLUDES = -I$(CUDA_DIR)/samples/common/inc/
LIBS = -lGL -lGLU -lX11 -lglut
CXX_FLAGS = -std=c++11 $(INCLUDES) $(LIBS)

all:
	$(NVCC) $(CXX_FLAGS) main.cu lab1.cu Perlin.cpp -o lab1
run:
	./lab1
	$(FFMPEG) -i result.y4m -y -r 24 output.mp4
clean:
	rm lab1
