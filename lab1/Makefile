CUDA_DIR=/usr/local/cuda-8.0
NVCC=$(CUDA_DIR)/bin/nvcc

FFMPEG=/home/peiya/ffmpeg-3.2.4/ffmpeg
CXX_FLAGS = -std=c++11

all:
	$(NVCC) $(CXX_FLAGS) main.cu lab1.cu FractalNoise.cpp Perlin.cpp -o lab1
run:
	./lab1
	$(FFMPEG) -i result.y4m -y -r 24 output.mp4
	git add output.mp4
	git commit -m "run"
	git push

clean:
	rm lab1
