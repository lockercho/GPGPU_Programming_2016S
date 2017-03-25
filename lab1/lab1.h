#pragma once
#include <cstdint>
#include <memory>
#include "FractalNoise.h"
using std::unique_ptr;

struct Lab1VideoInfo {
	unsigned w, h, n_frame;
	unsigned fps_n, fps_d;
};

class Lab1VideoGenerator {
	struct Impl;
	unique_ptr<Impl> impl;
	FractalNoise * noiseMaker;
	int W = 640;
	int H = 480;
	int NFRAME = 24;
	float rotMat[2][2];
	float * loose_noise;
	float * dense_noise;
public:
	Lab1VideoGenerator();
	~Lab1VideoGenerator();
	void get_info(Lab1VideoInfo &info);
	void Generate(uint8_t *yuv);
	float getNoise(float * noiseArr, int x, int y); 
	void setRotMatrix(int degree);
	void generateNoise(float * noiseArr, float freq);
};
