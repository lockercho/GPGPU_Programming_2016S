CUDA_DIR=/usr/local/cuda-8.0
NVCC=$(CUDA_DIR)/bin/nvcc
CC=g++

FFMPEG=/home/peiya/ffmpeg-3.2.4/ffmpeg
CXX_FLAGS = -std=c++11

all:
	$(NVCC) $(CXX_FLAGS) main.cu lab1.cu Perlin3D.cpp Particle.cpp -o lab1

gravity: 
	$(CC) $(CXX_FLAGS) gravity.cpp Particle.cpp -o gravity

run:
	./lab1
	$(FFMPEG) -i result.y4m -y -r 24 output.mp4
	git add .
	git commit -m "run"
	git push origin master

clean:
	rm lab1
